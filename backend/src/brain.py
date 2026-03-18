import json
import logging
import os
import time
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", default=None)
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "1TuanPham/T-VisStar-7B-v0.1")

# Global state: True = use Gemini, False = use finetuned model via vLLM
USE_GEMINI = bool(GEMINI_API_KEY and GEMINI_API_KEY != "your-gemini-api-key-here")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# vLLM client (OpenAI-compatible API)
vllm_client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")

from tavily_search import search

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def _convert_messages_to_gemini(messages):
    """Convert OpenAI-style messages to Gemini contents format."""
    system_instruction = None
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_instruction = content
        elif role == "assistant":
            contents.append({"role": "model", "parts": [content]})
        else:
            contents.append({"role": "user", "parts": [content]})
    return system_instruction, contents


def _gemini_call(messages, model_name="gemini-2.0-flash"):
    """Call Gemini API with retry logic. Returns text or raises on failure."""
    system_instruction, contents = _convert_messages_to_gemini(messages)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction,
    )
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.generate_content(contents)
            return response.text
        except Exception as e:
            last_error = e
            logger.warning(f"Gemini attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    raise last_error


def _vllm_call(messages):
    """Call the finetuned model served via vLLM (OpenAI-compatible API)."""
    response = vllm_client.chat.completions.create(
        model=VLLM_MODEL_NAME,
        messages=messages,
    )
    return response.choices[0].message.content


def chat_complete(messages=()):
    """Unified chat completion: tries Gemini if enabled, falls back to vLLM."""
    global USE_GEMINI
    if USE_GEMINI:
        try:
            return _gemini_call(messages)
        except Exception as e:
            logger.error(f"Gemini failed after {MAX_RETRIES} retries, falling back to vLLM: {e}")
            USE_GEMINI = False
            return _vllm_call(messages)
    else:
        return _vllm_call(messages)


def set_use_gemini(value: bool):
    """Allow external code (API endpoint) to toggle the model."""
    global USE_GEMINI
    USE_GEMINI = value
    logger.info(f"USE_GEMINI set to {value}")


def get_use_gemini() -> bool:
    return USE_GEMINI


def gen_doc_prompt(docs):
    """
    """
    doc_prompt = "Dưới đây là tài liệu về các điều luật liên quan đến câu hỏi của người dùng:"
    for i,doc in enumerate(docs):
        doc_prompt += f"{i}. {doc} \n"
    doc_prompt += "Kết thúc phần các tài liệu liên quan."

    return doc_prompt


def generate_conversation_text(conversations):
    conversation_text = ""
    for conversation in conversations:
        logger.info("Generate conversation: {}".format(conversation))
        role = conversation.get("role", "user")
        content = conversation.get("content", "")
        conversation_text += f"{role}: {content}\n"
    return conversation_text

# Dựa vào history và câu hỏi hiện tại => Viết lại câu hỏi.
def detect_user_intent(history, message):
    # Convert history to list messages
    history_messages = generate_conversation_text(history)
    logger.info(f"History messages: {history_messages}")
    # Update documents to prompt
    user_prompt = f"""
    Dựa vào lịch sử hội thoại và câu hỏi mới nhất của người dùng, hãy viết lại câu hỏi mới nhất
    thành một câu hỏi độc lập, rõ ràng bằng tiếng Việt. Người dùng có thể chuyển đổi giữa các chủ đề
    pháp luật khác nhau như luật giao thông, kinh tế, v.v., vì vậy hãy xác định chính xác ý định
    hiện tại của người dùng để diễn đạt lại câu hỏi một cách chính xác nhất.
    Câu hỏi được viết lại phải rõ ràng, đầy đủ và có thể hiểu được mà không cần ngữ cảnh bổ sung.

    Lịch sử hội thoại:
    {history_messages}

    Câu hỏi gốc: {message}

    Câu trả lời:
    """
    gemini_messages = [
        {"role": "system", "content": "Bạn là một trợ lý thông minh chuyên về pháp luật Việt Nam."},
        {"role": "user", "content": user_prompt}
    ]
    logger.info(f"Rephrase input messages: {gemini_messages}")
    return chat_complete(gemini_messages)


# Classify xem câu query thuộc loại nào?
def detect_route(history, message):
    logger.info(f"Detect route on history messages: {history}")
    # Update documents to prompt
    user_prompt = f"""
    Dựa vào lịch sử hội thoại và tin nhắn mới nhất của người dùng, hãy phân loại câu hỏi vào một trong hai loại sau:

    1. Câu hỏi liên quan đến pháp luật tại Việt Nam, các tình huống thực tế liên quan đến luật.
    Ví dụ:
       - Nếu xe máy không đội mũ bảo hiểm thì bị phạt bao nhiêu tiền?
       - Nếu ô tô đi ngược chiều thì bị phạt thế nào?
       - Lập kế hoạch đấu giá quyền khai thác khoáng sản dựa trên các căn cứ nào?
       - Mục đích của bảo hiểm tiền gửi là gì?
       - Bao nhiêu tuổi đi nghĩa vụ quân sự?
    => Nhãn: "legal"

    2. Câu hỏi thông thường, không liên quan đến pháp luật.
    Ví dụ:
       - Xin chào, tôi cần bạn hỗ trợ.
       - Hôm nay thời tiết thế nào?
    => Nhãn: "chitchat"

    Chỉ trả về nhãn phân loại, không giải thích thêm.

    Lịch sử hội thoại:
    {history}

    Tin nhắn mới nhất:
    {message}

    Phân loại (chọn "chitchat" hoặc "legal"):
    """
    gemini_messages = [
        {"role": "system", "content": "Bạn là một trợ lý thông minh giúp phân loại câu hỏi của người dùng."},
        {"role": "user", "content": user_prompt}
    ]
    logger.info(f"Route output: {gemini_messages}")
    raw = chat_complete(gemini_messages)
    return raw.strip().strip('"').lower()

# define agent for process search internet + gen response
def get_legal_agent_anwer(messages):
    logger.info(f"Call tavily tool search")
    # Extract the last user message as the search query
    query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            query = msg.get("content", "")
            break
    observation = search(query)
    logger.info(f"Search observation: {observation[:200]}")
    full_message = messages + [{
            "role": "user",
            "content": f"Dưới đây là kết quả tìm kiếm từ internet liên quan đến câu hỏi của bạn. Hãy dựa vào đó để trả lời bằng tiếng Việt:\n{observation}"
    }]
    response = chat_complete(full_message)
    return response



if __name__ == "__main__":
    history = [{"role": "system", "content": "Bạn là một trợ lý thông minh chuyên về pháp luật Việt Nam."}]
    message = "Xin chào"
    output_detect = detect_route(history, message)
    print(output_detect)