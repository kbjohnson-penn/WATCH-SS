import string

filler_sounds = [
    "ah",
    "eh",
    "er",
    "hm",
    "huh",
    "mm",
    "uh",
    "um",
]

uncommon_letters = list(filter(lambda c: c not in ["a", "i", "o"], list(string.ascii_lowercase)))

keywords = filler_sounds + uncommon_letters