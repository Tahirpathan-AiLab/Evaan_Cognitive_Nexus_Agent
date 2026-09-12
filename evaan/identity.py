import re


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

_TELL_NAME_PATTERNS = [
    r"\bmy\s*name\s*is\s+(\w+)",
    r"\bmera\s*naam\s+(\w+)\s*hai",
    r"\bmai\s*(\w+)\s*hu\b",
]

_ASK_USER_NAME_PATTERNS = [
    r"\bwhat\s*(is|'s)?\s*my\s*name\b",
    r"\bmera\s*naam\s*kya\s*hai\b",
]


def match_any(patterns, text):
    return any(re.search(pattern, text) for pattern in patterns)
