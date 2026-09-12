# =========================================================
# EVAAN — Local CPU Version
# Qwen2.5-0.5B-Instruct
# First run downloads automatically.
# After download, Evaan can run without internet.
# =========================================================

import os
import warnings
import logging
from urllib.parse import quote

from dotenv import load_dotenv

# Suppress Mem0 telemetry (PostHog network errors)
os.environ["MEM0_TELEMETRY"] = "False"

# Suppress HF Hub unauthenticated-request warning
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Force offline mode — stops HuggingFace Hub from checking for updates
# over the network on every run. Without this, if the internet is slow
# or down, model loading hangs for 20+ seconds retrying network calls
# before falling back to the local cache.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Suppress general Python warnings (FutureWarning, etc.)
warnings.filterwarnings("ignore")

# Suppress noisy library loggers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("mem0").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

# Load .env file (EVAAN_DB_NAME, EVAAN_DB_USER, EVAAN_DB_PASSWORD, etc.)
load_dotenv()


def env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or not value:
        raise RuntimeError(f"Missing {name}. Copy .env.example to .env and set it.")
    return value


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DB_NAME = env("EVAAN_DB_NAME", "evaan_brain")
DB_USER = env("EVAAN_DB_USER")
DB_PASSWORD = env("EVAAN_DB_PASSWORD")
DB_HOST = env("EVAAN_DB_HOST", "localhost")
DB_PORT = int(env("EVAAN_DB_PORT", "5432"))

MEM0_CONFIG = {
    "embedder": {"provider": "huggingface", "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}},
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "dbname": DB_NAME,
            "user": DB_USER,
            "password": DB_PASSWORD,
            "host": DB_HOST,
            "port": DB_PORT,
            "collection_name": env("EVAAN_MEM0_COLLECTION", "evaan_memory"),
            "connection_string": (
                f"postgresql://{quote(DB_USER, safe='')}:{quote(DB_PASSWORD, safe='')}"
                f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            ),
        }
    },
    # Mem0 2.x supports Hugging Face for embeddings, but not as an LLM
    # provider.  Evaan saves raw turns (``infer=False``), so Mem0 never
    # calls this client; the actual chat model remains the local Transformers
    # Qwen model configured above.  ``lmstudio`` is a supported local provider
    # and keeps the configuration valid without requiring an API key.
    "llm": {
        "provider": "lmstudio",
        "config": {
            "model": MODEL_ID,
            "lmstudio_base_url": "http://localhost:1234/v1",
        },
    },
}
USER_ID = env("EVAAN_USER_ID", DB_USER)

DB_CONFIG = {
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "host": DB_HOST,
    "port": DB_PORT,
}


# How many recent messages are sent to the model as context.
# This does NOT limit how much is saved to the database anymore.
MAX_TURNS_IN_CONTEXT = 20

# How many messages we keep in evaan_history before trimming
# the oldest ones. Set this much higher than MAX_TURNS_IN_CONTEXT
# so "persistent memory" actually persists beyond a few turns.
MAX_HISTORY_STORED = 400

MOOD_RECOVERY_TURNS = 3