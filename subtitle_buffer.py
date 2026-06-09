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


def _contains_phrase(haystack_tokens: list[str], needle_tokens: list[str]) -> bool:
    haystack = _normalized_words(haystack_tokens)
    needle = _normalized_words(needle_tokens)
    if not haystack or not needle or len(needle) > len(haystack):
        return False
    for index in range(0, len(haystack) - len(needle) + 1):
        if haystack[index : index + len(needle)] == needle:
            return True
    return False


def _longest_suffix_prefix_overlap(
    left_tokens: list[str],
    right_tokens: list[str],
    max_overlap: int = 32,
) -> int:
    left_norm = _normalized_words(left_tokens)
    right_norm = _normalized_words(right_tokens)
    max_size = min(max_overlap, len(left_norm), len(right_norm))
    for size in range(max_size, 0, -1):
        if left_norm[-size:] == right_norm[:size]:
            return size
    return 0


def _drop_words_from_start(tokens: list[str], word_count: int) -> list[str]:
    if word_count <= 0:
        return tokens
    result = []
    remaining = word_count
    for token in tokens:
        if remaining > 0 and _normalize_word(token):
            remaining -= 1
            continue
        result.append(token)
    return result


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
    """Live-caption stabilizer with short visible history.

    Whisper re-transcribes overlapping audio windows. Appending every returned
    chunk duplicates lyrics, while replacing the whole caption removes history.
    This buffer merges new chunks into a rolling transcript using suffix/prefix
    overlap and renders only the last visible words.
    """

    def __init__(self, max_lines: int, max_words: int | None = None) -> None:
        self.max_lines = max_lines
        self.max_words = max_words or max(32, max_lines * 14)
        self.history_words_limit = self.max_words * 3
        self.tokens: list[str] = []
        self.current_text = ""
        self.last_text = ""

    def add(self, text: str) -> bool:
        cleaned = clean_subtitle_window(text, self.max_words)
        if not cleaned:
            self.last_text = ""
            return False

        new_tokens = cleaned.split()
        new_tokens = _collapse_adjacent_repeats(new_tokens)

        if _contains_phrase(self.tokens[-self.history_words_limit :], new_tokens):
            self.last_text = ""
            return False

        overlap = _longest_suffix_prefix_overlap(self.tokens, new_tokens)
        tail_tokens = _drop_words_from_start(new_tokens, overlap)

        if not tail_tokens:
            self.last_text = ""
            return False

        if overlap == 0 and len(self.tokens) >= 8:
            recent = self.tokens[-self.max_words :]
            if _contains_phrase(recent, tail_tokens[: max(3, min(8, len(tail_tokens)))]):
                self.last_text = ""
                return False

        self.tokens.extend(tail_tokens)
        self.tokens = _collapse_adjacent_repeats(self.tokens)
        self.tokens = _trim_to_last_words(self.tokens, self.history_words_limit)

        visible_tokens = _trim_to_last_words(self.tokens, self.max_words)
        visible_text = " ".join(visible_tokens).strip()
        self.last_text = " ".join(tail_tokens).strip()

        if not visible_text or _same_text(visible_text, self.current_text):
            return False

        self.current_text = visible_text
        return True

    def render(self) -> str:
        return self.current_text
