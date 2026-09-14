from mem0 import Memory
from config import MEM0_CONFIG, USER_ID


m = Memory.from_config(MEM0_CONFIG)
print("Mem0 memory system loaded.")
print()


def remember_turn(user_text, assistant_text):
    m.add(
        [{"role": "user", "content": user_text}, {"role": "assistant", "content": assistant_text}],
        user_id=USER_ID,
        infer=False
    )

def get_relevant_memory(query, limit=3): 
    results = m.search(
    query=query,
    filters={"user_id": USER_ID},
    limit=limit
    )
    if not results.get("results"):
        return ""
    return "\n".join(r["memory"] for r in results["results"])
