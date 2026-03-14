
# LLM chat bot - Vietnamese Q&A System in Vietnamese's legal document 
In this project, I build a complete Q&A chatbot related to Vietnamese's legal document

Link dataset : [Link data](https://drive.google.com/drive/folders/1HyF8-EfL4w0G3spBbhcc0jTOqdc4XUhB)

# Table of content

<!--ts-->
   * [Project structure](#project-structure)
   * [Getting started](#getting-started)
      * [Prepare enviroment](#prepare-enviroment)
      * [Running application docker container in local](#running-application-docker-container-in-local)
   * [Application services](#application-services)
      * [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
        * [System overview](#system-overview)
        * [Build Vector Database and Elasticsearch](#buiding-vectorDB-and-elasticsearch)
        * [RAG flow answering](#rag-flow-answering)
        * [Dual LLM Backend](#dual-llm-backend)
        * [Finetune rerank model](#finetune-rerank-model)
        * [Finetune LLM for answer generation](#finetune-LLM)
   * [Demo](#demo)
<!--te-->

# Project structure
```bash
├── backend                                   
│   ├── .env                                    # Environment variables (API keys, endpoints)
│   ├── requirements.txt                        # backend dependencies for the backend 
│   ├── entrypoint.sh                           # script run backend  
│   ├── src                                     # Source code for the backend
│   │   ├── search_document                             
│   │   │   ├── combine_search.py              # ensemble result from Bge-m3, e5
│   │   │   ├── rerank.py                      # reranking with finetuned BGE-reranker-v2-m3
│   │   │   ├── search_elastic.py              # Search using elasticsearch
│   │   │   ├── search_with_bge.py             # Search using Bge-m3 (Qdrant Cloud)
│   │   │   └── search_with_e5.py              # Search using Multilingual-e5-large (Qdrant Cloud)
│   │   ├── agent.py                            # LangChain ReAct agent with Tavily search (Gemini/vLLM)
│   │   ├── app.py                              # Entry point for the FastAPI backend + model switch API
│   │   ├── brain.py                            # Dual LLM logic (Gemini + vLLM fallback), routing, rewriting
│   │   ├── cache.py                            # Redis cache for conversation IDs
│   │   ├── database.py                         # Celery task queue configuration
│   │   ├── models.py                           # MongoDB models for chat history
│   │   ├── tavily_search.py                    # Tavily internet search tool
│   │   ├── schemas.py                          # Pydantic data schemas for API endpoints
│   │   ├── tasks.py                            # Celery tasks: routing, RAG, generation
│   │   └── utils.py                            # Utility functions (logging, ID generation)                  
├── chatbot-ui                                  # Frontend chatbot application (Streamlit)
│   ├── chat_interface.py                       # Chatbot interface with model selector sidebar
│   ├── config.toml                             # Configuration file for chatbot                  
│   ├── entrypoint.sh                           # Entrypoint script for chatbot
│   ├── requirements.txt                        # Python dependencies for chatbot
├── finetune_llm                                # Directory for finetune llm
│   ├── download_model.py                       # download base model          
│   ├── finetune.py                             # finetune LLM for answer generation (QLoRA + SFTTrainer)
│   ├── gen_data.py                             # Generate training data via GPT-4o-mini
│   ├── merge_with_base.py                      # Merge finetuned LoRA weights with base model
│   ├── test_model.py                           # Run inference with merged model via vLLM
│   ├── evaluate_finetuned_model.py             # Evaluate correctness with LlamaIndex evaluator
│   └── requirements.txt                        # Finetuning dependencies
├── images                                      # Directory for storing image assets
├── retrieval                                   # Retrieval folder
│   ├── FlagEmbedding                           # folder include code finetune
│   ├── hard_negative_bge_round1.py             # search using bge-m3
│   ├── hard_negative_e5.py                     # search using e5
│   ├── create_data_rerank.py                   # create data for reranking   
│   ├── finetune.sh                             # Script to finetuning bge-reranker-v2-m3
│   └── setup_env.sh                            # Script to create env
```
# Getting started

To get started with this project, we need to do the following

## Prepare enviroment 
Install all dependencies dedicated to the project in local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -r chatbot-ui/requirements.txt
```

## Configure environment variables

Copy or edit `backend/.env` with your credentials:

```bash
GEMINI_API_KEY = your-gemini-api-key-here        # Google Gemini API key
TAVILY_API_KEY = your-tavily-api-key-here        # Tavily search API key
QDRANT_URL = https://your-cluster.cloud.qdrant.io:6333   # Qdrant Cloud endpoint
QDRANT_API_KEY = your-qdrant-api-key-here        # Qdrant Cloud API key
VLLM_BASE_URL = http://localhost:8000/v1         # vLLM server endpoint (for finetuned model)
VLLM_MODEL_NAME = 1TuanPham/T-VisStar-7B-v0.1   # Model name served by vLLM
```

## Start application

**Backend (FastAPI + Celery worker):**
```bash
sh backend/entrypoint.sh
```

**Chatbot UI (Streamlit):**
```bash
sh chatbot-ui/entrypoint.sh
```

**(Optional) Serve finetuned model via vLLM:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model path/to/merged_model \
    --host 0.0.0.0 --port 8000
```

> **Note:** External services (Redis, MongoDB, Qdrant Cloud, Elasticsearch) must be running/accessible before starting the backend.

# Application services 

## RAG (Retrieval-Augmented Generation) 

### System overview

![rag_system](images/rag_flow.jpg)

### Build Vector Database and Elasticsearch 
- Because the size of each rule is quite long, the first step will need to be chunked into smaller parts. Then these chunks will be passed through 2 embedding models, Bge-m3 and Multilingual-e5-large, and finally these embedding vectors  will be stored in Qdrant.
- Additionally, these rules are also saved to elasticsearch to enhance the accuracy of retrieval based on lexical matching. 

### RAG flow answering

- **Routing user's intent**: Initially, based on the current query and chat history, determine whether this user's intent is chitchat or law topic. The `Gemini 2.0 Flash` model (or the finetuned model via vLLM) combined with `few-shot prompting` is used to perform this intent determination. If the user's intent is chitchat, it will go through the LLM to return the final answer. Else, going to query reflection step.

- **Query reflection**: Chat history and current query will be rewritten into a single sentence with more complete meaning for easier retrieval. The model used in this step is `Gemini 2.0 Flash` (with automatic fallback to the finetuned vLLM model).

- **Retrieval Relevant Documents**: The rewritten query will be passed through two embedding models, Bge-m3 and Multilingual-e5-large, then **Qdrant Cloud** will be used to retrieve semantically related documents. Besides, Elasticsearch is also used for retrieval based on lexical matching. Finally, to avoid losing relevant documents during retrieval, all retrieved documents were merged and duplicates were removed.

- **Reranking**: If reranking is not used, the number of retrieved documents is quite large. If this entire number of documents is put into the LLM, it may exceed the model's input token limit and be expensive. If the number of documents is small (small top_k), it may lead to the loss of related documents. The top-k documents retrieved from the previous step will be passed through the rerank model to re-rank the scores and get the top5 documents with the highest scores.

- **Generating Final Answer**: The LLM combines the top5 documents after reranking step with the user's query and chat history to generate a response. In the prompt for the LLM I specified that it will return 'no' if the retrieved document does not contain the answer, so if the response is different from 'no' then it will be the final answer. If the response is 'no' then it will call the search tool in the next step to get more information.

- **Tool call and Generation**: Use the **LangChain ReAct agent** with the Tavily search tool to search for content on the internet related to the query and then feed this content back to the LLM to generate answers. The agent supports both Gemini and the finetuned vLLM model as its backbone.

### Dual LLM Backend

The system supports two LLM backends that can be switched at runtime:

| Backend | Model | Description |
|---------|-------|-------------|
| **Gemini (Cloud API)** | `gemini-2.0-flash` | Google's cloud-hosted model via `google-generativeai` SDK. Default when `GEMINI_API_KEY` is configured. |
| **Finetuned Model (vLLM)** | `1TuanPham/T-VisStar-7B-v0.1` | Self-hosted finetuned Vietnamese legal QA model served via vLLM's OpenAI-compatible API. |

**How it works:**
- A global `USE_GEMINI` flag in `brain.py` controls which backend is active.
- On startup, if a valid `GEMINI_API_KEY` is detected, Gemini is used by default.
- If Gemini fails (API key missing, timeout, etc.), the system **automatically retries up to 3 times** with exponential backoff, then **falls back to the finetuned vLLM model**.
- Users can manually switch models from the **Streamlit sidebar** ("Gemini (Cloud API)" / "Finetuned Model (vLLM)").
- The backend also exposes REST endpoints for model management:
  - `GET /model/status` — returns the currently active model
  - `POST /model/switch` — switch between `"gemini"` and `"finetuned"`
  - `POST /chat/complete` — accepts optional `use_model` field per request

### Key technology migrations

| Component | Before | After |
|-----------|--------|-------|
| **LLM** | OpenAI GPT-4o-mini | Google Gemini 2.0 Flash + finetuned model via vLLM |
| **Agent framework** | LlamaIndex ReActAgent | LangChain ReAct agent (`create_react_agent` + `AgentExecutor`) |
| **Vector database** | Qdrant (local, `localhost:6333`) | Qdrant Cloud (authenticated via `QDRANT_URL` + `QDRANT_API_KEY`) |
| **Agent LLM** | LlamaIndex OpenAI LLM | `ChatGoogleGenerativeAI` / `ChatOpenAI` (vLLM) via LangChain |

### Finetune rerank model
Create enviroment 
```bash
cd retrieval
sh setup_env.sh
```
#### Create data finetune
- Train data should be a json file, where each line is a dict like this:

```shell
{"query": str, "pos": List[str], "neg":List[str]}
```
`query` is the query, and `pos` is a list of positive texts, `neg` is a list of negative texts. 
- For each embedding model => will take the top 25 chunks with the highest similarity to each query. If the chunk is in the labeled data, it will be assigned as positive and vice versa, it will be negative => Then the results of these embedding models will be summarized.

- Follow the steps below to create the training dataset

```bash
Step1: cd retrieval
Step2: CUDA_VISIBLE_DEVICES=0 python create_data_rerank.py
```

#### Finetune BGE-v2-m3
Finetune BGE-v2-m3 with parameters: 

    - epochs: 6
    - learning_rate: 1e-5
    - batch_size = 2

Run script for training
```bash
sh finetune.sh
```
### Finetune LLM for answer generation
#### Create + format training data
- The training data will be in conversational format.
```shell
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```
- Follow the steps below to create the training + test dataset
```bash
Step1: cd finetune_llm
Step2: python gen_data.py
```
- The number of training dataset are 10000 samples and the number of test dataset are 1000 samples

#### Finetune LLM
- The base model I used for finetune is [1TuanPham/T-VisStar-7B-v0.1](https://huggingface.co/1TuanPham/T-VisStar-7B-v0.1). This model ranks quite high on the VMLU Leaderboard of Fine-tuned Models
- I used the [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) from trl library to finetune this model. Beside, I user [QLora](https://arxiv.org/abs/2305.14314) technique to reduce the memory footprint of large language models during finetuning, without sacrificing performance by using quantization.

Run script for training
```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py
```
- Below are the results for the training process on WanDB
![Tracking training](images/tracking_finetune_llm.png)

- Merge weight with model base
Run scrip for merge
```bash
python merge_with_base.py
```

### Evaluate 

The evaluation metrics currently in use are:

- **Recall@k**: Evaluate the accuracy of information retrieval
- **Correctness**:The metric evaluates the answer generated by the system to match a given query reference answer.

The golden dataset I chose for evaluation consists of 1000 samples. Each sample includes 3 fields: query, related_documents, answer


**Recall@k**
|Model               | K=3    | K =5   | K=10    |
|-----------------   |--------|--------|---------|
|BGE-m3              | 55.11% | 63.43% | 72.18%  |
|E5                  | 54.61% | 63.53% | 72.02%  |
|Elasticsearch       | 42.54% | 49.61% | 56.85%  |
|Ensemble            | 68.38% | 74.85% | 80.66%  |
|Ensemble + rerank   | 79.82% | 82,82% | 87.66%  |

**Correctness**

Score is rated on a 5-point scale and has an accuracy of 4.27/5
# DEMO       
![demo](images/demo.png)