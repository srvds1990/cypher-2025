# db/vector_store.py
def list_sessions():
    """Return dummy session names for testing UI."""
    return [
        "Generative AI: Job Enabler or Disruptor?",
        "Applied GenAI for Business",
        "Responsible AI and Ethics",
        "AI in Healthcare: Opportunities & Challenges"
    ]

def query_documents(session: str, query: str):
    """Return dummy docs for a session."""
    return [f"This is a mock document for {session}. Query was: {query}"]
