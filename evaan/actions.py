import torch
from model import tokenizer, model


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
