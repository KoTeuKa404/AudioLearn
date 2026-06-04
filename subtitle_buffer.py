from collections import deque


class SubtitleBuffer:
    def __init__(self, max_lines: int) -> None:
        self.lines = deque(maxlen=max_lines)

    def add(self, text: str) -> bool:
        text = " ".join(text.split())
        if not text:
            return False
        if self.lines and text == self.lines[-1]:
            return False
        self.lines.append(text)
        return True

    def render(self) -> str:
        return "\n".join(self.lines)
