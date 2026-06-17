# Production Agentic RAG

CRAG-style agent: Retrieve -> Grade -> (Generate | Rewrite -> Retrieve | WebSearch -> Generate),
backed by ChromaDB, Claude 3.5 Sonnet, Pydantic-validated tool I/O,
and a persistent LangGraph SQLite checkpointer for per-session memory.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then edit ANTHROPIC_API_KEY
```

## Ingest Sample data
```bash
python -m app.vectorstore.ingest --path ./data --reset
```

## Run the API 
```bash
uvicorn app.main:app --reload --port 8000
```
------

## Try it 👉
# First turn — start a new session
```bash
curl -s -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"question":"What is Corrective RAG?"}' | jq
```
# Re-use the returned session_id to continue the conversation
```bash
curl -s -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"question":"How does it relate to LangGraph?","session_id":"<paste-id>"}' | jq
```
# Force the CRAG fallback path (no local docs match)
```bash
curl -s -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"question":"Latest pricing for Anthropic Claude API?"}' | jq
```
# Inspect persisted memory
```bash
curl -s http://localhost:8000/session/<paste-id>/history | jq
```
---

## 3. Execution Instructions 

```bash
# 1. Clone / create the project and enter it
cd agentic-rag

# 2. Create an isolated environment
python -m venv .venv
source .venv/bin/activate                # macOS/Linux
# .venv\Scripts\activate                 # Windows PowerShell

# 3. Install dependencies (pinned for reproducibility)
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure secrets
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...

# 5. Ingest the sample corpus (semantic chunking into ChromaDB)
python -m app.vectorstore.ingest --path ./data --reset

# 6. Start the API
uvicorn app.main:app --reload --port 8000

# 7. Smoke-test
curl -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"question":"What is Corrective RAG?"}'
```

