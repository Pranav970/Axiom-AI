# Production Agentic RAG

CRAG-style agent: Retrieve -> Grade -> (Generate | Rewrite -> Retrieve | WebSearch -> Generate),
backed by ChromaDB, Claude 3.5 Sonnet, Pydantic-validated tool I/O,
and a persistent LangGraph SQLite checkpointer for per-session memory.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then edit ANTHROPIC_API_KEY
