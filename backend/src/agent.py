import os
import logging
from dotenv import load_dotenv, find_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "1TuanPham/T-VisStar-7B-v0.1")

# Tavily search tool (reads TAVILY_API_KEY from env automatically)
tavily_tool = TavilySearchResults(max_results=3)


def _build_agent(use_gemini: bool):
    """Build a LangGraph ReAct agent with either Gemini or vLLM backend."""
    if use_gemini:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_API_KEY,
        )
    else:
        llm = ChatOpenAI(
            base_url=VLLM_BASE_URL,
            api_key="not-needed",
            model=VLLM_MODEL_NAME,
        )
    return create_react_agent(llm, [tavily_tool])


def _convert_to_langchain_messages(messages):
    """Convert dict messages to LangChain message objects."""
    lc_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))
    return lc_messages


def react_agent_handle(history, question):
    from brain import get_use_gemini
    # Build agent dynamically based on current model selection
    agent = _build_agent(use_gemini=get_use_gemini())

    # Build context from history
    history_text = ""
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        history_text += f"{role}: {content}\n"

    full_input = f"Chat History:\n{history_text}\nCurrent Question: {question}" if history_text else question
    result = agent.invoke({"messages": [("user", full_input)]})
    return result["messages"][-1].content