import logging
import os
import time
from typing import Dict, Optional
from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from dotenv import load_dotenv
from utils import setup_logging
from tasks import llm_handle_message
from brain import set_use_gemini, get_use_gemini

from search_document.combine_search import CombinedSearch
from search_document.rerank import BGEReranker

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

# When CELERY_ENABLED=false the app runs fully synchronous — no Redis/Celery required.
CELERY_ENABLED = os.getenv("CELERY_ENABLED", "true").strip().lower() == "true"
if not CELERY_ENABLED:
    logger.info("[startup] CELERY_ENABLED=false — running in uvicorn-only (sync) mode")

# init retriever and reranker
combined_search_instance = CombinedSearch()
_RERANKER_PATH = os.getenv(
    "RERANKER_MODEL_PATH",
    "BAAI/bge-reranker-v2-m3",
)
reranker_instance = BGEReranker(model_name=_RERANKER_PATH, use_fp16=True)


app = FastAPI()

# define class name
class CompleteRequest(BaseModel):
    bot_id: Optional[str] = 'bot_Legal_VN'
    user_id: str
    user_message: str
    sync_request: Optional[bool] = False
    use_model: Optional[str] = None  # "gemini" or "finetuned"; None = use current global

class ModelSwitchRequest(BaseModel):
    model: str  # "gemini" or "finetuned"

class RetrievalRequest(BaseModel):
    query: str
    top_k_search: int = 30
    top_k_rerank: int = 5


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/model/status")
async def model_status():
    return {"use_gemini": get_use_gemini(), "model": "gemini" if get_use_gemini() else "finetuned"}

@app.post("/model/switch")
async def model_switch(request: ModelSwitchRequest):
    if request.model not in ("gemini", "finetuned"):
        raise HTTPException(status_code=400, detail="model must be 'gemini' or 'finetuned'")
    set_use_gemini(request.model == "gemini")
    return {"model": request.model, "use_gemini": get_use_gemini()}

@app.post("/retrieval")
async def retrieval(request: RetrievalRequest):
    try:
        # Lấy dữ liệu từ body
        query = request.query
        top_k_search = request.top_k_search
        top_k_rerank = request.top_k_rerank
        # Thực hiện tìm kiếm bằng CombinedSearch
        search_results = combined_search_instance.search(query_text=query, top_k=top_k_search)

        # Thực hiện rerank kết quả tìm kiếm
        reranked_results = reranker_instance.rerank(query=query, documents=search_results, topk=top_k_rerank)

        return {
            "results": reranked_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.post("/chat/complete")
async def complete(data: CompleteRequest):
    bot_id = data.bot_id
    user_id = data.user_id
    user_message = data.user_message
    logger.info(f"Complete chat from user {user_id} to {bot_id}: {user_message}")

    if not user_message or not user_id:
        raise HTTPException(status_code=400, detail="User id and user message are required")

    # Per-request model override
    if data.use_model == "finetuned":
        set_use_gemini(False)
    elif data.use_model == "gemini":
        set_use_gemini(True)

    # Run sync when Celery is disabled OR caller explicitly requests sync
    if not CELERY_ENABLED or data.sync_request:
        response = llm_handle_message(bot_id, user_id, user_message)
        return {"response": str(response)}
    else:
        task = llm_handle_message.delay(bot_id, user_id, user_message)
        return {"task_id": task.id}


@app.get("/chat/complete_v2/{task_id}")
async def get_response(task_id: str):
    if not CELERY_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Task polling is unavailable: CELERY_ENABLED=false. Use sync_request=true instead.",
        )
    start_time = time.time()
    timeout = 60  # Timeout sau 60 giây
    polling_interval = 0.1  # Thời gian chờ giữa mỗi lần kiểm tra (100ms)
    
    while True:
        # Lấy trạng thái task từ Celery
        task_result = AsyncResult(task_id)
        task_status = task_result.status
        
        # Ghi log trạng thái task
        logger.info(f"Task ID: {task_id}, Status: {task_status}")
        
        # Nếu task đã hoàn tất, trả về kết quả
        if task_status not in ('PENDING', 'STARTED'):
            return {
                "task_id": task_id,
                "task_status": task_status,
                "task_result": task_result.result
            }
        
        # Kiểm tra timeout
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            logger.warning(f"Task {task_id} timed out after {timeout} seconds.")
            return {
                "task_id": task_id,
                "task_status": task_status,
                "error_message": "Service timeout, please retry."
            }
        
        # Chờ trước khi kiểm tra lại
        await asyncio.sleep(polling_interval)

if __name__ == "__main__":
    import uvicorn
    _debug = os.getenv("DEBUG", "false").strip().lower() == "true"
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8002")),
        workers=1,  # reload requires workers=1
        reload=_debug,
        log_level="debug" if _debug else "info",
    )

