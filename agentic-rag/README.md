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

## Try it
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
