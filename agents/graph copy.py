# agents/graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from agents.nodes import retriever_node, summarizer_node, qa_node, visualizer_node


class SessionState(TypedDict, total=False):
    session: str
    query: str
    docs: List[str]
    context: str
    summary: str
    visualization: str
    answer: str


def build_summary_graph():
    graph = StateGraph(SessionState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("visualizer", visualizer_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "summarizer")
    graph.add_edge("summarizer", "visualizer")
    graph.add_edge("visualizer", END)

    return graph.compile()


def build_qa_graph():
    graph = StateGraph(SessionState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("qa", qa_node)
    graph.add_node("visualizer", visualizer_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "qa")
    graph.add_edge("qa", "visualizer")
    graph.add_edge("visualizer", END)

    return graph.compile()
