# agents/nodes.py
from typing import Dict, List
from db.vector_store import semantic_search_by_text
from utils.chunking import chunk_text
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

# Init OpenAI Client
client_oa = OpenAI(api_key=OPENAI_API_KEY)
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")


def retriever_node(state: Dict) -> Dict:
    """Retrieve relevant transcript chunks for a query or session."""
    session = state.get("session")
    query = state.get("query", " ")
    where = {"session": session} if session else None
    q = query if query.strip() else session or " "
    res = semantic_search_by_text(q, top_k=8, where=where)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    state["docs"] = docs
    state["metas"] = metas
    state["context"] = "\n\n---\n\n".join(docs[:8]) if docs else ""
    return state


def summarizer_node(state: Dict) -> Dict:
    """Generate a structured summary of the retrieved session context."""
    context = state.get("context", "")
    system = (
        "You are an assistant that summarizes conference session transcripts for a professional audience. "
        "Produce: a 1-line title, a 60-90 word summary, 3 clear one-line key takeaways, and 3 practical actions."
    )
    user_prompt = f"Context:\n{context}\n\nPlease produce: title, summary (60-90 words), 3 key takeaways, 3 practical actions."

    try:
        resp = client_oa.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        out = resp.choices[0].message.content.strip()
    except Exception as e:
        out = f"⚠️ Failed to call LLM: {e}"

    state["summary"] = out
    state["answer"] = out

    # Extract keywords for visualization
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=20)
        X = vec.fit_transform(state.get("docs", []))
        terms = vec.get_feature_names_out()
        scores = X.sum(axis=0).A1
        kw = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
        state["keywords"] = kw[:12]
    except Exception:
        state["keywords"] = []
    return state


def qa_node(state: Dict) -> Dict:
    """Answer user questions using only the retrieved context."""
    context = state.get("context", "")
    question = state.get("query", "")
    system = "You are a helpful assistant answering user questions using the provided context only. If not present, say you couldn't find the answer."
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer concisely and cite context if helpful."

    try:
        resp = client_oa.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=350,
        )
        out = resp.choices[0].message.content.strip()
    except Exception as e:
        out = f"⚠️ Failed to call LLM: {e}"

    state["answer"] = out
    return state


def visualizer_node(state: Dict) -> Dict:
    """Prepare data for a simple keyword frequency visualization."""
    kw = state.get("keywords", [])
    state["viz_fig_data"] = {
        "terms": [t for t, s in kw],
        "scores": [s for t, s in kw],
    }
    return state
