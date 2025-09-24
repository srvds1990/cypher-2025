# agents/nodes.py
from typing import Dict

def retriever_node(state: Dict) -> Dict:
    session = state.get("session", "Unknown session")
    docs = [f"Dummy document content for session: {session}"]
    state["docs"] = docs
    state["context"] = "\n".join(docs)
    return state


def summarizer_node(state: Dict) -> Dict:
    docs = state.get("docs", [])
    state["summary"] = f"📑 Summary for {state.get('session', 'Unknown')}\n\n" \
                       f"This is a dummy summary.\n\n" + "\n".join(docs)
    state["visualization"] = "📊 Dummy visualization goes here."
    state["answer"] = state["summary"]  # so UI fallback works
    return state


def qa_node(state: Dict) -> Dict:
    question = state.get("query", "")
    docs = state.get("docs", [])
    state["answer"] = f"🤖 Q: {question}\n\nAnswer: This is a dummy QA result using {len(docs)} docs."
    return state


def visualizer_node(state: Dict) -> Dict:
    return state
