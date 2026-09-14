import psycopg2
from config import DB_CONFIG, MAX_HISTORY_STORED


# 5. MEMORY (Postgres-backed) 

def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def load_memory():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT role, content FROM evaan_history ORDER BY id ASC LIMIT %s", (MAX_HISTORY_STORED,))
    messages = [{"role": r, "content": c} for r, c in cur.fetchall()]
    cur.execute("SELECT mood, recovery_counter FROM evaan_state WHERE id = 1")
    row = cur.fetchone()
    mood, recovery = row if row else ("happy", 0)
    cur.close()
    conn.close()
    print(f"Loaded {len(messages)} saved messages (mood: {mood})")
    return messages, mood, recovery


def save_memory(history, mood, recovery_counter):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("TRUNCATE evaan_history")
    trimmed_history = [m for m in history if m["role"] != "system"][-MAX_HISTORY_STORED:]
    for msg in trimmed_history:
        cur.execute("INSERT INTO evaan_history (role, content) VALUES (%s, %s)", (msg["role"], msg["content"]))
    cur.execute("UPDATE evaan_state SET mood = %s, recovery_counter = %s WHERE id = 1", (mood, recovery_counter))
    conn.commit()
    cur.close()
    conn.close()


def clear_memory():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("TRUNCATE evaan_history")
    cur.execute("UPDATE evaan_state SET mood = 'happy', recovery_counter = 0 WHERE id = 1")
    conn.commit()
    cur.close()
    conn.close()
    print("Memory cleared — Evaan starts fresh and happy again.")


def get_user_name():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT user_name FROM evaan_state WHERE id = 1")
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row and row[0] else None


def set_user_name(name):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE evaan_state SET user_name = %s WHERE id = 1", (name,))
    conn.commit()
    cur.close()
    conn.close()