import re

from model import tokenizer, model, generate_response
from persona import build_system_prompt
from mood import update_mood
from memory.db import load_memory, save_memory, clear_memory, get_user_name, set_user_name
from memory.mem0_store import remember_turn, get_relevant_memory
from actions import try_handle_action
from identity import (
    match_any,
    _NAME_QUESTION_PATTERNS,
    _CREATOR_QUESTION_PATTERNS,
    _TELL_NAME_PATTERNS,
    _ASK_USER_NAME_PATTERNS,
)
from config import MOOD_RECOVERY_TURNS

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
            save_memory(history, mood, recovery_counter)
            print("Evaan: Memory saved. Bye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() in ["quit", "exit", "/quit"]:
            save_memory(history, mood, recovery_counter)
            print("Evaan: Bye bye! Come back soon! 😊")
            break

        if user_input == "/save":
            save_memory(history, mood, recovery_counter)
            print("[Memory saved]\n")
            continue

        if user_input == "/clear":
            clear_memory()
            history = []
            mood = "happy"
            recovery_counter = 0
            continue

        if user_input == "/history":
            print(f"[{len(history)} messages remembered]\n")
            continue

        if user_input == "/mood":
            print(f"[Mood: {mood} | Recovery: {recovery_counter}/{MOOD_RECOVERY_TURNS}]\n")
            continue

        # Update mood
        mood, recovery_counter = update_mood(mood, recovery_counter, user_input)

        # Desktop action check (before identity / chat)
        action_reply = try_handle_action(user_input)

        if action_reply is not None:
            reply = action_reply
            print("Evaan:", reply)

        else:
            normalized_input = user_input.lower().strip().replace("?", "")

            # -------------------------------------------------
            # User name tracking (deterministic, checked first)
            # -------------------------------------------------
            name_match = None
            for pattern in _TELL_NAME_PATTERNS:
                found = re.search(pattern, normalized_input)
                if found:
                    name_match = found.group(1).capitalize()
                    break

            if name_match:
                set_user_name(name_match)
                reply = f"Nice to meet you, {name_match}! I'll remember that."
                print("Evaan:", reply)

            elif match_any(_ASK_USER_NAME_PATTERNS, normalized_input):
                known_name = get_user_name()
                reply = f"Your name is {known_name}." if known_name else "I don't think you've told me your name yet!"
                print("Evaan:", reply)

            elif match_any(_NAME_QUESTION_PATTERNS, normalized_input):
                reply = "I'm Evaan."
                print("Evaan:", reply)

            elif match_any(_CREATOR_QUESTION_PATTERNS, normalized_input):
                reply = "Tahir created me."
                print("Evaan:", reply)

            else:
                history.append({"role": "user", "content": user_input})
                relevant_memory = get_relevant_memory(user_input)

                print("Evaan: ", end="", flush=True)

                try:
                    reply = generate_response(history, mood, relevant_memory)
                    print(reply)
                except Exception as error:
                    reply = f"(Evaan encountered an error: {error})"
                    print(reply)

        history.append({"role": "assistant", "content": reply})
        remember_turn(user_input, reply)

        # NOTE: history is intentionally NOT truncated here.
        # generate_response() already only sends the last
        # MAX_TURNS_IN_CONTEXT messages to the model, and
        # save_memory() only trims what's written to the database once
        # it passes MAX_HISTORY_STORED. This is what actually
        # makes memory "persistent" instead of capped at ~10 turns.

        save_memory(history, mood, recovery_counter)