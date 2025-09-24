# utils/chunking.py
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt", quiet=True)

from typing import List

def chunk_text(text: str, chunk_size_chars: int = 1200, overlap_chars: int = 200) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []
    cur = ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= chunk_size_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)

    if overlap_chars > 0 and len(chunks) > 1:
        new_chunks = []
        for i, c in enumerate(chunks):
            prev = chunks[i-1] if i>0 else ""
            overlap = prev[-overlap_chars:] if prev else ""
            new_chunks.append((overlap + " " + c).strip())
        return new_chunks
    return chunks
