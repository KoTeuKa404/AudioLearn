from collections import deque
import re


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _normalize_word(word: str) -> str:
    found = _WORD_RE.findall(word.lower())
    return "".join(found)


def _normalized_words(tokens: list[str]) -> list[str]:
    return [word for word in (_normalize_word(token) for token in tokens) if word]


def _collapse_adjacent_repeats(tokens: list[str], max_phrase_words: int = 12) -> list[str]:
    if len(tokens) < 2:
        return tokens

    tokens = tokens[:]
    changed = True
    passes = 0
    while changed and passes < 6:
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


def _remove_prefix_overlap(text_tokens: list[str], recent_tokens: list[str], max_overlap: int = 18) -> list[str]:
    if not text_tokens or not recent_tokens:
        return text_tokens

    text_norm = _normalized_words(text_tokens)
    recent_norm = _normalized_words(recent_tokens)
    if not text_norm or not recent_norm:
        return text_tokens

    max_size = min(max_overlap, len(text_norm), len(recent_norm))
    overlap_size = 0
    for size in range(max_size, 0, -1):
        if recent_norm[-size:] == text_norm[:size]:
            overlap_size = size
            break

    if overlap_size <= 0:
        return text_tokens

    words_to_skip = overlap_size
    result = []
    for token in text_tokens:
        if words_to_skip > 0 and _normalize_word(token):
            words_to_skip -= 1
            continue
        result.append(token)
    return result


def _is_duplicate_of_recent(text_tokens: list[str], recent_tokens: list[str]) -> bool:
    text_norm = _normalized_words(text_tokens)
    recent_norm = _normalized_words(recent_tokens)
    if not text_norm or not recent_norm:
        return False

    if len(text_norm) <= 2:
        return False

    window = " ".join(recent_norm[-80:])
    phrase = " ".join(text_norm)
    return phrase in window


def clean_subtitle_chunk(text: str, recent_text: str = "") -> str:
    text = " ".join(str(text).split()).strip()
    if not text:
        return ""

    tokens = text.split()
    recent_tokens = recent_text.split()

    tokens = _collapse_adjacent_repeats(tokens)
    tokens = _remove_prefix_overlap(tokens, recent_tokens)
    tokens = _collapse_adjacent_repeats(tokens)

    if _is_duplicate_of_recent(tokens, recent_tokens):
        return ""

    return " ".join(tokens).strip()


class SubtitleBuffer:
    def __init__(self, max_lines: int) -> None:
        self.lines = deque(maxlen=max_lines)
        self.last_text = ""

    def add(self, text: str) -> bool:
        recent_text = self.render()
        text = clean_subtitle_chunk(text, recent_text)
        self.last_text = text
        if not text:
            return False
        if self.lines and text == self.lines[-1]:
            return False
        self.lines.append(text)
        return True

    def render(self) -> str:
        return "\n".join(self.lines)
