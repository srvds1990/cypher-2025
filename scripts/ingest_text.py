# scripts/ingest_text.py
import nltk

# Auto-download punkt if missing
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.vector_store import upsert_full_document
from utils.chunking import chunk_text
import uuid

def ingest_file(path, session_name):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    prefix = f"{session_name.replace(' ','_')}_{uuid.uuid4().hex[:8]}"
    res = upsert_full_document(prefix, session_name, text, chunk_text)
    print("Ingested:", res)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/ingest_text.py path/to/transcript.txt \"Session Title\"")
        sys.exit(1)
    ingest_file(sys.argv[1], sys.argv[2])
