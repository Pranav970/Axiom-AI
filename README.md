# Axiom-AI
Production-Grade Agentic RAG &amp; Multi-Agent Orchestration (LangGraph, Pydantic, ChromaDB)

---

markdown
# 🤖 Production Agentic RAG Framework

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-🚀-orange?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![Anthropic Claude 3.5](https://img.shields.io/badge/Claude%203.5%20Sonnet-🎨-purple?style=for-the-badge)](https://www.anthropic.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-⚡-blue?style=for-the-badge)](https://www.trychroma.com/)

An enterprise-grade, asynchronous **Corrective Retrieval-Augmented Generation (CRAG)** engine powered by **LangGraph**, **FastAPI**, and **Claude 3.5 Sonnet**. This framework features self-correcting evaluation loops, semantic document chunking, and SQLite-backed persistent multi-turn conversation memory.

---

## 🧭 Architecture Flow

The workflow models complex multi-step reasoning as a stateful, cyclic directed graph via **LangGraph**:

```
    🚀 START
       │
       ▼
┌──────────────┐

──>│   Retrieve   │
│  └──────┬───────┘
│          │
│          ▼
│   ┌──────────────┐
│   │    Grade     │
│   └──────┬───────┘
│          │
│          ├─────── [ Verdict: Relevant ] ───────► ┌──────────────┐
│          │                                       │   Generate   │ ──► 🏁 END
│    [ Irrelevant & ]                              └──────▲───────┘
│    [ Rewrites left ]                                    │
│          │                                        [ Web Docs ]
│          ▼                                              │
│   ┌──────────────┐                               ┌──────────────┐
└───│   Rewrite    │                               │  Web Search  │
    └──────────────┘                               └──────▲───────┘
│
[ Irrelevant & Exhausted ]

```

### ⚡ Key Features
* **Self-Correcting LLM-as-a-Judge (`grade_node`)**: Strict Pydantic-enforced verification evaluates document relevance before generation.
* **Adaptive Query Reformulation (`rewrite_node`)**: Automatically optimizes terms to boost vector recall if initial hits fail.
* **Web Fallback Loop (`web_search_node`)**: Seamless fallback mechanism to pull outside web-corpus contexts to completely eliminate hallucinations.
* **Threaded Session Memory**: Native SQLite checkpointing persists complete conversational graphs across stateless HTTP requests using a single `session_id`.
* **Hybrid Ingestion Engine**: Couples **Semantic Chunking** with a structural recursive fallback safety net to enforce bounded context sizing.

---

## 📂 Project Structure

```text
agentic-rag/
├── app/
│   ├── main.py                 # FastAPI Web API Gateway
│   ├── config.py               # LRU-cached configuration management
│   ├── agent/
│   │   ├── state.py            # TypedDict defining Global Graph State
│   │   ├── nodes.py            # Async Functional Nodes (Execution)
│   │   ├── edges.py            # Pure routing logic functions
│   │   └── graph.py            # StateGraph compilation & orchestration
│   ├── tools/
│   │   ├── schemas.py          # Structured Pydantic contracts for LLM outputs
│   │   └── web_search.py       # Asynchronous Web search component
│   ├── memory/
│   │   └── checkpointer.py     # SQLite application lifespan lifecycle hook
│   └── vectorstore/
│       ├── store.py            # ChromaDB engine & embedding setup
│       └── ingest.py           # Command Line Bulk Ingestion Processor
├── data/
│   └── sample.md               # Local knowledge base documentation source
└── storage/                    # Local storage (Git-ignored)

```

---

## ⚙️ Quick Start

### 1. Clone & Set Up Environment

```bash
git clone [https://github.com/your-username/agentic-rag.git](https://github.com/your-username/agentic-rag.git)
cd agentic-rag

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install strictly pinned modules
pip install -r requirements.txt

```

### 2. Configure Credentials

Create a `.env` file from the provided template:

```bash
cp .env.example .env

```

Open `.env` and fill in your keys:

```env
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxx
CLAUDE_MODEL=claude-3-5-sonnet-20241022
CHROMA_PERSIST_DIR=./storage/chroma
MEMORY_DB_PATH=./storage/memory.sqlite

```

### 3. Run Knowledge Base Ingestion

Populate your Vector database with semantic chunking from your local files:

```bash
python -m app.vectorstore.ingest --path ./data --reset

```

### 4. Boot Up Server Gateway

Spin up the high-performance ASGI production server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

```

The Interactive Swagger API sandbox will automatically map to: `http://localhost:8000/docs`

---

## 🛰️ Core API Interaction Specifications

### 📥 1. Execute Multi-Turn Chat

`POST /chat`

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What is Corrective RAG and what happens if local search is irrelevant?",
       "session_id": "session_user_01"
     }'

```

**Response Example (`200 OK`)**:

```json
{
  "session_id": "session_user_01",
  "answer": "Corrective Retrieval-Augmented Generation (CRAG) evaluates document quality using a grader. If local retrieval is deemed irrelevant, the framework rewrites the query and falls back to an external web search.",
  "confidence": "high",
  "sources": [
    "./data/sample.md"
  ],
  "retrieval_verdict": "relevant",
  "relevance_score": 0.95,
  "used_web_search": false,
  "rewrite_count": 0
}

```

### 🔍 2. Inspect Session Timeline History

`GET /session/{session_id}/history`

Allows you to verify exactly what state transitions are saved in the persistent SQLite graph database.

```bash
curl -X GET "http://localhost:8000/session/session_user_01/history"

```

---

## 🛠️ Production Verification Testing

To trigger the **fallback path** (Irrelevant -> Rewrite -> Web Search) to test the graph routing robustness, ask the endpoint something outside your local `./data/sample.md` dataset:

```json
{
  "question": "What are the core benchmarks of Anthropic Claude 3.5 Sonnet?",
  "session_id": "session_user_01"
}

```

**Engine Trace Observation**:

1. `retrieve` matches nothing of statistical importance.
2. `grade` outputs a score below `GRADE_THRESHOLD` ➡️ returns `irrelevant`.
3. `rewrite` steps in to optimize keywords.
4. `websearch` pulls matched context strings out of the search runner.
5. `generate` provides a fully structured `FinalAnswer` with active web URLs cited!

---

🏁 **Developed with ❤️ using Python, LangGraph and Claude 3.5.**

***








