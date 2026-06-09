import re


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _normalize_word(word: str) -> str:
    found = _WORD_RE.findall(word.lower())
    return "".join(found)


def _normalized_words(tokens: list[str]) -> list[str]:
    return [word for word in (_normalize_word(token) for token in tokens) if word]


def _collapse_adjacent_repeats(tokens: list[str], max_phrase_words: int = 10) -> list[str]:
    if len(tokens) < 2:
        return tokens

    tokens = tokens[:]
    changed = True
    passes = 0
    while changed and passes < 8:
        changed = False
        passes += 1
        i = 0
        while i < len(tokens):
            max_size = min(max_phrase_words, (len(tokens) - i) // 2)
            removed = False
            for size in range(max_size, 0, -1):
                left = _normalized_words(tokens[i : i + size])
                right = _normalized_words(tokens[i + size : i + 2 * size])
                if left and left == right:
                    del tokens[i + size : i + 2 * size]
                    changed = True
                    removed = True
                    break
            if not removed:
                i += 1
    return tokens


def _trim_to_last_words(tokens: list[str], max_words: int) -> list[str]:
    if max_words <= 0:
        return tokens
    word_count = 0
    cut_index = 0
    for index in range(len(tokens) - 1, -1, -1):
        if _normalize_word(tokens[index]):
            word_count += 1
        if word_count >= max_words:
            cut_index = index
            break
    if word_count < max_words:
        return tokens
    return tokens[cut_index:]


def _same_text(left: str, right: str) -> bool:
    return _normalized_words(left.split()) == _normalized_words(right.split())


def clean_subtitle_window(text: str, max_words: int = 46) -> str:
    text = " ".join(str(text).split()).strip()
    if not text:
        return ""

    tokens = text.split()
    tokens = _collapse_adjacent_repeats(tokens)
    tokens = _trim_to_last_words(tokens, max_words)
    tokens = _collapse_adjacent_repeats(tokens)
    return " ".join(tokens).strip()


class SubtitleBuffer:
    """Live-caption stabilizer.

    This intentionally does not append every Whisper result forever. Whisper often
    re-transcribes the same rolling audio window, so appending creates duplicated
    lyrics. The buffer keeps the current rolling caption window and lets the
    overlay replace its text, similar to mobile live captions.
    """

    def __init__(self, max_lines: int, max_words: int | None = None) -> None:
        self.max_lines = max_lines
        self.max_words = max_words or max(28, max_lines * 12)
        self.current_text = ""
        self.last_text = ""

    def add(self, text: str) -> bool:
        cleaned = clean_subtitle_window(text, self.max_words)
        self.last_text = cleaned
        if not cleaned:
            return False
        if _same_text(cleaned, self.current_text):
            return False
        self.current_text = cleaned
        return True

    def render(self) -> str:
        return self.current_text
