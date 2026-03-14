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
    Based on the following conversation history and the latest user query, rewrite the latest query as 
    a standalone question in Vietnamese. The user may switch between different legal topics, such as 
    traffic laws, economic regulations, etc., so ensure the intent of the user is accurately
    identified at the current moment to rephrase the query as precisely as possible. 
    The rewritten question should be clear, complete, and understandable without additional context.

    Chat History:
    {history_messages}

    Original Question: {message}

    Answer:
    """
    gemini_messages = [
        {"role": "system", "content": "You are an amazing virtual assistant"},
        {"role": "user", "content": user_prompt}
    ]
    logger.info(f"Rephrase input messages: {gemini_messages}")
    return chat_complete(gemini_messages)


# Classify xem câu query thuộc loại nào?
def detect_route(history, message):
    logger.info(f"Detect route on history messages: {history}")
    # Update documents to prompt
    user_prompt = f"""
    Given the following chat history and the user's latest message. Hãy phân loại xu hướng mong muốn trong tin nhắn của user là loại nào trong 2 loại sau. \n
    1. Mong muốn hỏi các thông tin liên quan đến luật pháp tại Việt Nam, các tình huống thực tế gặp phải liên quan đến luật 
    Ví dụ: -  Nếu xe máy không đội mũ bảo hiểm thì bị phạt bao nhiêu tiền?
           -  Nếu ô tô đi ngược chiều thì bị phạt thế nào?
           -  Lập kế hoạch đấu giá quyền khai thác khoáng sản dựa trên các căn cứ nào ?
           -  Mục đích của bảo hiểm tiền gửi là gì ?
    => Loại này có nhãn là : "legal"
    2. Mong muốn chitchat thông thường.
    Ví dụ:  - Hi, xin chào, tôi cần bạn hỗ trợ,....
            - Chủ tịch nước Việt Nam là ai ,....
    => Loại này có nhãn là : "chitchat"
    Provide only the classification label as your response.

    Chat History:
    {history}

    Latest User Message:
    {message}

    Classification (choose either "chitchat" or "legal"):
    """
    gemini_messages = [
        {"role": "system", "content": "You are a highly intelligent assistant that helps classify customer queries"},
        {"role": "user", "content": user_prompt}
    ]
    logger.info(f"Route output: {gemini_messages}")
    return chat_complete(gemini_messages)

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
            "content": observation
    }]
    response = chat_complete(full_message)
    return response



if __name__ == "__main__":
    history = [{"role": "system", "content": "You are an amazing virtual assistant"}]
    message = "Xin chào"
    output_detect = detect_route(history, message)
    print(output_detect)