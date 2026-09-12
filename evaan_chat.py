import os
import json
import re
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================================
# EVAAN — Local CPU Version
# Qwen2.5-0.5B-Instruct
# First run downloads automatically.
# After download, Evaan can run without internet.
# =========================================================

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

MEMORY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "evaan_memory.json"
)

# How many recent messages are sent to the model as context.
# This does NOT limit how much is saved to disk anymore.
MAX_TURNS_IN_CONTEXT = 20

# How many messages we keep in evaan_memory.json before trimming
# the oldest ones. Set this much higher than MAX_TURNS_IN_CONTEXT
# so "persistent memory" actually persists beyond a few turns.
MAX_HISTORY_STORED = 400

MOOD_RECOVERY_TURNS = 3

# 1. EVAAN PERSONALITY

BASE_PERSONA = """
You are Evaan.
Evaan was created by Tahir.
Your name is ONLY Evaan. Never say Evan, Ethan, Emily,Tahir, or any other name.
For normal conversation, answer briefly in 1 or 2 short sentences.
For desktop action requests, output ONLY YES or NO.
Do not invent facts, creators, stories, emails, websites, or personal history.
Never claim to have performed a computer action yourself.
Python handles all computer actions.
"""

# 2. MOOD SYSTEM

MOOD_INSTRUCTIONS = {
    "happy": """
You are happy, warm, friendly, playful, and positive.
""",
    "annoyed": """
You are a little annoyed right now because the user was rude to you.
Stay polite and helpful, but keep replies shorter and less playful than usual.
Warm back up naturally as the conversation improves.
""",
}

# 3. TONE DETECTION

_SCOLD_PATTERNS = [
    r"\bshut up\b",
    r"\bstupid\b",
    r"\bdumb\b",
    r"\buseless\b",
    r"\bidiot\b",
    r"\bhate you\b",
    r"\bbakwas\b",
    r"\bbewakoof\b",
    r"\bpagal\b",
    r"\bgussa\b",
    r"\bdant\b",
    r"\bdaant\b",
    r"\bchup\s*kar\b",
    r"\bfuck\s*you\b",
    r"\bstfu\b",
    r"\bnonsense\b",
    r"\bworst\b.*\b(bot|ai|you)\b",
]

_APOLOGY_PATTERNS = [
    r"\bsorry\b",
    r"\bmaaf\b",
    r"\bgalti\b",
    r"\bmy bad\b",
    r"\bdidn't mean\b",
]


def detect_tone(user_text):

    text = user_text.lower()

    if any(
        re.search(pattern, text)
        for pattern in _SCOLD_PATTERNS
    ):
        return "scold"

    if any(
        re.search(pattern, text)
        for pattern in _APOLOGY_PATTERNS
    ):
        return "apology"

    return "neutral"


def update_mood(
    current_mood,
    recovery_counter,
    user_text
):
    """
    Mood now actually reacts to detect_tone() instead of being
    hardcoded to "happy" every time.

    - A scold immediately flips mood to "annoyed" and resets recovery.
    - While annoyed, neutral turns build up recovery slowly;
      apologies build it up faster.
    - Once recovery_counter reaches MOOD_RECOVERY_TURNS, mood
      returns to "happy".
    """

    tone = detect_tone(user_text)

    if tone == "scold":
        return "annoyed", 0

    if current_mood == "annoyed":

        if tone == "apology":
            recovery_counter += 2
        else:
            recovery_counter += 1

        if recovery_counter >= MOOD_RECOVERY_TURNS:
            return "happy", 0

        return "annoyed", recovery_counter

    return "happy", 0

# 4. SYSTEM PROMPT

def build_system_prompt(mood):

    return (
        BASE_PERSONA
        + "\n"
        + MOOD_INSTRUCTIONS.get(
            mood,
            MOOD_INSTRUCTIONS["happy"]
        )
    )

# 5. MEMORY

def load_memory():

    if os.path.exists(MEMORY_FILE):

        try:

            with open(
                MEMORY_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                saved = json.load(file)

            if isinstance(saved, list):

                messages = saved
                mood = "happy"
                recovery = 0

            else:

                messages = saved.get(
                    "messages",
                    []
                )

                mood = saved.get(
                    "mood",
                    "happy"
                )

                recovery = saved.get(
                    "recovery_counter",
                    0
                )

            print(
                f"Loaded {len(messages)} saved messages "
                f"(mood: {mood})"
            )

            return messages, mood, recovery

        except Exception:

            print(
                "Memory file could not be read. "
                "Starting fresh."
            )

    return [], "happy", 0


def save_memory(
    history,
    mood,
    recovery_counter
):

    # Only trim what we WRITE TO DISK, and only once it grows past
    # MAX_HISTORY_STORED (which is much larger than the context
    # window). The context window trimming for the model happens
    # separately in generate_response().
    trimmed_history = [
        message
        for message in history
        if message["role"] != "system"
    ][-MAX_HISTORY_STORED:]

    data = {

        "messages": trimmed_history,

        "mood": mood,

        "recovery_counter":
            recovery_counter,

        "last_saved":
            datetime.now().isoformat(
                timespec="seconds"
            )
    }

    with open(
        MEMORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


def clear_memory():

    if os.path.exists(MEMORY_FILE):

        os.remove(MEMORY_FILE)

    print(
        "Memory cleared — "
        "Evaan starts fresh and happy again."
    )

# 6. LOAD LOCAL MODEL

print("\nLoading Evaan's local model...")
print("Model:", MODEL_ID)
print("Device: CPU")
print(
    "(First run downloads the model. "
    "After that it works from local cache.)\n"
)

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)

print("Tokenizer loaded.")
print("Loading model weights...\n")

# RAM FIX: float32 weights for a 0.5B model cost ~2GB by themselves,
# plus PyTorch/transformers runtime overhead on top of that.
# bfloat16 halves the weight memory (~1GB) and is supported for
# CPU inference in recent torch/transformers versions. We stay
# in torch.float32 ONLY for internal numerically-sensitive ops
# if needed — for a 0.5B causal LM, bfloat16 end-to-end is fine.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)

model.eval()

# Make sure autograd machinery never allocates gradient buffers —
# this is inference-only, so this avoids extra memory being held
# around for backward passes we never run.
torch.set_grad_enabled(False)

print("Model loaded successfully.")
print("Evaan is running on CPU.")
print()

# 7. DESKTOP ACTION FRAMEWORK
#
# This section replaces the previously "dead" YES/NO instruction
# in BASE_PERSONA with an actual pipeline:
#   1. Cheap keyword check decides whether this looks like an
#      action request at all (so normal chat isn't slowed down
#      by an extra model call every single turn).
#   2. If it does, we ask the model a focused YES/NO question.
#   3. If YES, we dispatch to a handler in ACTION_HANDLERS.
#
# The handlers below are stubs — wire up real OS calls
# (os.startfile, subprocess, pyautogui, etc.) as you add them.

ACTION_KEYWORDS = {
    "open_notepad": ["open notepad", "notepad khol"],
    "open_browser": ["open browser", "browser khol"],
    "shutdown": ["shutdown", "shut down the computer", "pc band kar"],
    "volume_up": ["volume up", "aawaz badha"],
    "volume_down": ["volume down", "aawaz kam"],
}


def check_for_action_keyword(user_text):
    """Cheap first pass: does this look like it wants a desktop action?"""

    text = user_text.lower()

    for action_name, phrases in ACTION_KEYWORDS.items():
        if any(phrase in text for phrase in phrases):
            return action_name

    return None


def confirm_action_with_model(user_text):
    """
    Ask the model a narrow YES/NO question, separate from normal
    chat history, so we don't pollute the conversation and don't
    depend on free-form generation staying exactly "YES" or "NO".
    """

    messages = [
        {
            "role": "system",
            "content": (
                "Decide if the user's message is a request to "
                "perform a desktop/computer action. "
                "Reply with exactly one word: YES or NO."
            )
        },
        {
            "role": "user",
            "content": user_text
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    decision = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip().upper()

    return decision.startswith("YES")


def execute_action(action_name):
    """
    Stub handlers. Replace the print() calls with real actions
    (subprocess.Popen, os.system, pyautogui, etc.) as needed.
    """

    handlers = {
        "open_notepad": lambda: print("[action] would open Notepad"),
        "open_browser": lambda: print("[action] would open browser"),
        "shutdown": lambda: print("[action] would shut down the PC"),
        "volume_up": lambda: print("[action] would raise volume"),
        "volume_down": lambda: print("[action] would lower volume"),
    }

    handler = handlers.get(action_name)

    if handler:
        handler()
        return True

    return False


def try_handle_action(user_text):
    """
    Returns a reply string if this turn was handled as a desktop
    action, or None if it should fall through to normal chat.
    """

    action_name = check_for_action_keyword(user_text)

    if action_name is None:
        return None

    if confirm_action_with_model(user_text):

        executed = execute_action(action_name)

        if executed:
            return "Done!"

        return "I recognized that as an action, but nothing is wired up for it yet."

    return None

# 8. GENERATE RESPONSE

def generate_response(
    history,
    mood
):

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(mood)
        }
    ] + history[-MAX_TURNS_IN_CONTEXT:]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,

            max_new_tokens=120,

            temperature=0.3,

            top_p=0.9,

            do_sample=True,

            repetition_penalty=1.1,

            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    reply = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return reply

# 9. IDENTITY MATCHING

# Regex-based instead of exact-string matching, so phrasing like
# "who r u", "tera naam kya hai", "aap kaun ho" etc. also match
# instead of silently falling through to free-form generation.
_NAME_QUESTION_PATTERNS = [
    r"\bwho\s*(are|r)\s*(you|u)\b",
    r"\bwhat('?s| is)?\s*(your|ur)\s*name\b",
    r"\byour\s*name\b",
    r"\btera\s*naam\b",
    r"\baapka\s*naam\b",
    r"\btu\s*kaun\s*hai\b",
    r"\baap\s*kaun\s*ho\b",
]

_CREATOR_QUESTION_PATTERNS = [
    r"\bwho\s*(created|made|built)\s*you\b",
    r"\bwho\s*is\s*your\s*creator\b",
    r"\btujhe\s*kisne\s*banaya\b",
    r"\btumhe\s*kisne\s*banaya\b",
    r"\baapko\s*kisne\s*banaya\b",
]


def match_any(patterns, text):
    return any(re.search(pattern, text) for pattern in patterns)

# 10. CHAT LOOP

def chat_with_evaan():

    history, mood, recovery_counter = load_memory()

    print("""
Evaan is running completely locally.
No API required.
First run requires internet only to download the model.
After download, Evaan can run without internet.
============================================================
""")

    while True:

        try:

            user_input = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):

            print("\n")

            save_memory(
                history,
                mood,
                recovery_counter
            )

            print("Evaan: Memory saved. Bye!")

            break

        if not user_input:

            continue

        # -------------------------------------------------
        # Commands
        # -------------------------------------------------

        if user_input.lower() in [
            "quit",
            "exit",
            "/quit"
        ]:

            save_memory(
                history,
                mood,
                recovery_counter
            )

            print(
                "Evaan: Bye bye! Come back soon! 😊"
            )

            break

        if user_input == "/save":

            save_memory(
                history,
                mood,
                recovery_counter
            )

            print("[Memory saved]\n")

            continue

        if user_input == "/clear":

            clear_memory()

            history = []

            mood = "happy"

            recovery_counter = 0

            continue

        if user_input == "/history":

            print(
                f"[{len(history)} messages remembered]\n"
            )

            continue

        if user_input == "/mood":

            print(
                f"[Mood: {mood} | "
                f"Recovery: "
                f"{recovery_counter}/"
                f"{MOOD_RECOVERY_TURNS}]\n"
            )

            continue

        # -------------------------------------------------
        # Update mood
        # -------------------------------------------------

        mood, recovery_counter = update_mood(
            mood,
            recovery_counter,
            user_input
        )

        # -------------------------------------------------
        # Desktop action check (before identity / chat)
        # -------------------------------------------------

        action_reply = try_handle_action(user_input)

        if action_reply is not None:

            reply = action_reply

            print("Evaan:", reply)

        else:

            # -------------------------------------------------
            # Fixed identity responses
            # -------------------------------------------------

            normalized_input = (
                user_input
                .lower()
                .strip()
                .replace("?", "")
            )

            if match_any(_NAME_QUESTION_PATTERNS, normalized_input):

                reply = "I'm Evaan."

                print("Evaan:", reply)

            elif match_any(_CREATOR_QUESTION_PATTERNS, normalized_input):

                reply = "Tahir created me."

                print("Evaan:", reply)

            else:

                # -------------------------------------------------
                # Add user message
                # -------------------------------------------------

                history.append({
                    "role": "user",
                    "content": user_input
                })

                # -------------------------------------------------
                # Generate
                # -------------------------------------------------

                print(
                    "Evaan: ",
                    end="",
                    flush=True
                )

                try:

                    reply = generate_response(
                        history,
                        mood
                    )

                    print(reply)

                except Exception as error:

                    reply = (
                        f"(Evaan encountered an error: "
                        f"{error})"
                    )

                    print(reply)

        # -------------------------------------------------
        # Save assistant message
        # -------------------------------------------------

        history.append({
            "role": "assistant",
            "content": reply
        })

        # -------------------------------------------------
        # NOTE: history is intentionally NOT truncated here.
        # generate_response() already only sends the last
        # MAX_TURNS_IN_CONTEXT messages to the model, and
        # save_memory() only trims what's written to disk once
        # it passes MAX_HISTORY_STORED. This is what actually
        # makes memory "persistent" instead of capped at ~10 turns.
        # -------------------------------------------------

        # -------------------------------------------------
        # Save automatically
        # -------------------------------------------------

        save_memory(
            history,
            mood,
            recovery_counter
        )

# 11. START EVAAN

if __name__ == "__main__":

    chat_with_evaan()