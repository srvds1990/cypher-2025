# scripts/delete_session.py
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from db import vector_store

def delete_session(old_session_name: str):
    coll = vector_store.collection
    print("Attempting to delete session:", old_session_name)
    # Try direct delete by metadata (preferred)
    try:
        coll.delete(where={"session": old_session_name})
        print("✅ Deleted by metadata using collection.delete(where=...)")
        return
    except Exception as e:
        print("⚠️ collection.delete(where=...) not supported or failed:", e)

    # Fallback: fetch all items and delete matching ids
    try:
        res = coll.get(include=["ids", "metadatas"])
        ids = res.get("ids", [])
        metas = res.get("metadatas", [])
        ids_to_delete = [i for i, m in zip(ids, metas)
                         if isinstance(m, dict) and m.get("session") == old_session_name]

        if not ids_to_delete:
            print("⚠️ No items found for session:", old_session_name)
            return

        coll.delete(ids=ids_to_delete)
        print(f"✅ Deleted {len(ids_to_delete)} items for session '{old_session_name}'")
    except Exception as ex:
        print("❌ Fallback delete failed:", ex)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/delete_session.py \"Old Session Name\"")
        sys.exit(1)
    delete_session(sys.argv[1])
