# Agentic RAG Framework — Internal Knowledge Base

## Corrective RAG (CRAG)
Corrective Retrieval-Augmented Generation introduces a lightweight
retrieval evaluator that scores the relevance of retrieved documents
against the user question. When the score is low, the system either
rewrites the query for another retrieval pass or falls back to an
external knowledge source such as web search. This avoids hallucination
when the local vector store does not contain a satisfactory answer.

## LangGraph
LangGraph models LLM agents as directed graphs with explicit state,
cycles, and persistence. Each node is a pure function that receives
the current state and returns a delta. Conditional edges enable
routing logic such as "if retrieval is irrelevant, go to web search".
The Checkpointer interface persists state per `thread_id`, enabling
multi-turn conversations across separate process invocations.

## ChromaDB
ChromaDB is an open-source, AI-native embedding database. In this
project it is used with persistent local storage and semantic
chunking so that retrieved chunks respect natural document
boundaries rather than fixed character windows.

## Pydantic for Structured Output
Pydantic schemas enforce that LLM outputs — grades, rewritten
queries, final answers — conform to strict types before they are
trusted by downstream nodes. Combined with Claude's structured
output mode, this eliminates brittle string parsing.
