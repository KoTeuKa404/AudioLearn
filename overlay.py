import ctypes
import os
import queue
import tkinter as tk
import tkinter.font as tkfont


class SubtitleOverlay:
    def __init__(self, ui_config: dict, text_queue: queue.Queue, stop_event) -> None:
        self.ui_config = ui_config
        self.text_queue = text_queue
        self.stop_event = stop_event
        self.current_text = ""
        self.placeholder_text = str(ui_config.get("placeholder_text", "")).strip()
        self.windowed = bool(ui_config.get("windowed", False))
        self.force_topmost = bool(ui_config.get("force_topmost", False))
        self.draggable = bool(ui_config.get("draggable", False))
        self._drag_start = None
        self.scroll_text = ""
        self.last_chunk = ""

        self.root = tk.Tk()
        if not self.windowed:
            self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", float(ui_config.get("opacity", 0.8)))

        bg = ui_config.get("background_color", "#000000")
        fg = ui_config.get("text_color", "#FFFFFF")
        last_fg = ui_config.get("last_text_color", "#FFD54A")

        self.root.configure(bg=bg)

        font_family = ui_config.get("font_family", "Segoe UI")
        font_size = int(ui_config.get("font_size", 22))
        padding = int(ui_config.get("padding_px", 16))

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width_ratio = float(ui_config.get("width_ratio", 0.9))
        width = int(screen_w * width_ratio)
        self.font = tkfont.Font(family=font_family, size=font_size)
        line_height = int(self.font.metrics("linespace"))
        self.max_lines = int(ui_config.get("max_lines", 3))
        height = int(line_height * self.max_lines + padding * 2)
        x = int((screen_w - width) / 2)
        y = int(screen_h - height - int(ui_config.get("bottom_margin_px", 80)))

        self.root.geometry(f"{width}x{height}+{x}+{y}")

        self.wrap_length = width - padding * 2
        avg_char_width = self.font.measure("n") or 10
        self.wrap_chars = max(10, int(self.wrap_length / avg_char_width))

        self.text = tk.Text(
            self.root,
            bg=bg,
            font=self.font,
            wrap="word",
            height=self.max_lines,
            width=self.wrap_chars,
            bd=0,
            highlightthickness=0,
            relief="flat",
            cursor="arrow",
        )
        self.text.pack(fill="both", expand=True, padx=padding, pady=padding)
        self.text.tag_configure("base", foreground=fg, justify="left")
        self.text.tag_configure("last", foreground=last_fg, justify="left")
        limit_override = ui_config.get("scroll_char_limit")
        self.scroll_char_limit = int(limit_override) if limit_override else int(
            self.wrap_chars * self.max_lines * 4
        )
        if self.placeholder_text:
            self._render_text(self.placeholder_text, "")
            self.current_text = self.placeholder_text

        self.root.update_idletasks()
        if ui_config.get("click_through", False) and not self.windowed:
            self._enable_click_through()
        elif self.draggable:
            self._enable_dragging()

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _enable_click_through(self) -> None:
        if os.name != "nt":
            return
        hwnd = self.root.winfo_id()
        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x00080000
        WS_EX_TRANSPARENT = 0x00000020
        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(
            hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT
        )

    def _enable_dragging(self) -> None:
        self.root.bind("<Button-1>", self._start_drag)
        self.root.bind("<B1-Motion>", self._on_drag)
        self.text.bind("<Button-1>", self._start_drag)
        self.text.bind("<B1-Motion>", self._on_drag)

    def _start_drag(self, event) -> None:
        self._drag_start = (event.x_root, event.y_root)

    def _on_drag(self, event) -> None:
        if self._drag_start is None:
            return
        x_start, y_start = self._drag_start
        dx = event.x_root - x_start
        dy = event.y_root - y_start
        self._drag_start = (event.x_root, event.y_root)
        x = self.root.winfo_x() + dx
        y = self.root.winfo_y() + dy
        self.root.geometry(f"+{x}+{y}")

    def _poll_queue(self) -> None:
        updated = False
        try:
            while True:
                text = self.text_queue.get_nowait()
                if text and self._append_scroll_text(text):
                    updated = True
        except queue.Empty:
            pass

        if updated:
            self._render_scroll_text()
            self.root.update_idletasks()
            if self.force_topmost:
                self.root.lift()
                self.root.attributes("-topmost", True)

        if not self.stop_event.is_set():
            self.root.after(100, self._poll_queue)
        else:
            self.close()

    def run(self) -> None:
        self._poll_queue()
        self.root.mainloop()

    def _append_scroll_text(self, text: str) -> bool:
        chunk = " ".join(text.split())
        if not chunk:
            return False
        self.last_chunk = chunk
        if self.scroll_text:
            self.scroll_text = f"{self.scroll_text} {chunk}"
        else:
            self.scroll_text = chunk
        if self.scroll_char_limit > 0 and len(self.scroll_text) > self.scroll_char_limit:
            excess = len(self.scroll_text) - self.scroll_char_limit
            cut = self.scroll_text.find(" ", excess)
            if cut == -1:
                self.scroll_text = self.scroll_text[-self.scroll_char_limit :]
            else:
                self.scroll_text = self.scroll_text[cut + 1 :]
        return True

    def _fit_lines(self, text: str) -> list[str]:
        if not text:
            return []
        lines: list[str] = []
        for raw in text.splitlines():
            wrapped = textwrap.wrap(raw, width=self.wrap_chars) or [""]
            lines.extend(wrapped)
        if len(lines) > self.max_lines:
            lines = lines[-self.max_lines :]
        return lines

    def _render_scroll_text(self) -> None:
        if self.scroll_text == self.current_text:
            return
        self._render_text(self.scroll_text, self.last_chunk)
        self.current_text = self.scroll_text

    def _render_text(self, display_text: str, highlight_chunk: str) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        if display_text:
            self.text.insert("end", display_text, ("base",))
            if highlight_chunk:
                start = display_text.rfind(highlight_chunk)
                if start != -1:
                    end = start + len(highlight_chunk)
                    self.text.tag_add("last", f"1.0 + {start} chars", f"1.0 + {end} chars")
        self.text.configure(state="disabled")
        self.text.see("end")

    def close(self) -> None:
        if not self.stop_event.is_set():
            self.stop_event.set()
        if self.root is not None:
            self.root.destroy()
            self.root = None
