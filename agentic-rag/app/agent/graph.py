"""StateGraph assembly.

Topology:

    START
      |
      v
   retrieve --> grade --+--(relevant)----> generate --> END
                        |
                        +--(irrelevant, attempts left)--> rewrite --> retrieve
                        |
                        +--(irrelevant, exhausted)------> websearch --> generate
"""
from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.agent.edges import route_after_grade
from app.agent.nodes import (
    generate_node,
    grade_node,
    retrieve_node,
    rewrite_node,
    web_search_node,
)
from app.agent.state import GraphState


def build_graph(checkpointer: BaseCheckpointSaver):
    g = StateGraph(GraphState)

    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("websearch", web_search_node)
    g.add_node("generate", generate_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")

    g.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "websearch": "websearch",
        },
    )

    # After rewriting, retrieve again with the improved query
    g.add_edge("rewrite", "retrieve")
    # After web search, go straight to generation (web docs are merged in)
    g.add_edge("websearch", "generate")
    g.add_edge("generate", END)

    return g.compile(checkpointer=checkpointer)
