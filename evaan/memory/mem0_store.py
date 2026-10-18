from mem0 import Memory
from config import MEM0_CONFIG, USER_ID


try:
    m = Memory.from_config(MEM0_CONFIG)
    print("Mem0 memory system loaded.")
except Exception as error:
    m = None
    print(f"Mem0 is unavailable; continuing without semantic memory ({error}).")
print()


def remember_turn(user_text, assistant_text):
    """Store a conversation turn without interrupting the chat if memory fails."""
    if m is None:
        return False

    try:
        m.add(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            user_id=USER_ID,
            infer=False,
        )
        return True
    except Exception as error:
        print(f"[Memory warning] Could not save this turn: {error}")
        return False


def get_relevant_memory(query, limit=3):
    """Return relevant memories, or an empty string when retrieval is unavailable."""
    if m is None:
        return ""

    try:
        results = m.search(
            query=query,
            filters={"user_id": USER_ID},
            limit=limit,
        )
        memories = results.get("results", []) if isinstance(results, dict) else []
        return "\n".join(
            item["memory"] for item in memories
            if isinstance(item, dict) and item.get("memory")
        )
    except Exception as error:
        print(f"[Memory warning] Could not retrieve memories: {error}")
        return ""
