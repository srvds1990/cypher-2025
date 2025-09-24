# db/vector_store.py
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings

# Load env variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in .env")

# ✅ Initialize OpenAI client (new API)
client_oa = OpenAI(api_key=OPENAI_API_KEY)

# Configuration
CHROMA_DIR = os.getenv("CHROMA_DIR", "./db/chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "cypher_sessions")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# ✅ Use NEW Chroma PersistentClient
client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)
)

# Ensure collection exists
collection = client.get_or_create_collection(name=CHROMA_COLLECTION)


def upsert_document_chunks(ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]):
    """Upsert chunks with embeddings into Chroma."""
    if not texts:
        return {"upserted": 0}

    all_embeddings = []
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            resp = client_oa.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            all_embeddings.extend([d.embedding for d in resp.data])
        except Exception as e:
            print(f"❌ Embedding batch failed: {e}")
            raise

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=all_embeddings,
        metadatas=metadatas
    )
    return {"upserted": len(texts)}


def upsert_full_document(doc_id_prefix: str, session: str, text: str, chunker_fn):
    """Split text into chunks and insert into Chroma with metadata."""
    chunks = chunker_fn(text)
    ids = [f"{doc_id_prefix}_{i}" for i in range(len(chunks))]
    metadatas = [{"session": session, "chunk_index": i} for i in range(len(chunks))]
    res = upsert_document_chunks(ids, chunks, metadatas)
    return {"ingested_chunks": res["upserted"], "first_id": ids[0] if ids else None}


def semantic_search_by_text(query: str, top_k: int = 6, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Query documents from Chroma by semantic similarity."""
    try:
        qemb_resp = client_oa.embeddings.create(model=EMBEDDING_MODEL, input=[query])
        qemb = qemb_resp.data[0].embedding
    except Exception as e:
        print(f"❌ Failed to embed query: {e}")
        raise

    results = collection.query(
        query_embeddings=[qemb],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )
    return results


def list_sessions() -> List[str]:
    """Return list of distinct session names from Chroma metadata."""
    try:
        res = collection.get(include=["metadatas"])
    except Exception:
        return []
    sessions = set()
    for m in res.get("metadatas", []):
        if isinstance(m, dict) and "session" in m:
            sessions.add(m["session"])
    return sorted(list(sessions))
