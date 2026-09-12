import re


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
