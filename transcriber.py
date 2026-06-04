import logging
import os
import queue
import threading
import time
from typing import Callable

import numpy as np

from audio_utils import float32_to_wav_bytes, resample_linear


class TranscribeWorker(threading.Thread):
    def __init__(
        self,
        segment_queue: queue.Queue,
        text_callback: Callable[[str], None],
        transcriber,
        config: dict,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.segment_queue = segment_queue
        self.text_callback = text_callback
        self.transcriber = transcriber
        self.stop_event = stop_event
        self.target_rate = int(config["audio"].get("target_sample_rate", 16000))
        self.min_silence_rms = float(config["transcription"].get("min_silence_rms", 0.0))
        self.language = config["transcription"].get("language", "auto")
        self.debug_audio = bool(config["transcription"].get("debug_audio", False))
        self.debug_interval = float(config["transcription"].get("debug_audio_interval", 5.0))
        self.debug_text = bool(config["transcription"].get("debug_text", False))
        self.debug_text_max_chars = int(
            config["transcription"].get("debug_text_max_chars", 160)
        )
        self.drop_old_segments = bool(config["transcription"].get("drop_old_segments", True))
        self.streaming = bool(config["transcription"].get("streaming", False))
        self.stream_window_seconds = float(
            config["transcription"].get("stream_window_seconds", 6.0)
        )
        self.stream_update_seconds = float(
            config["transcription"].get("stream_update_seconds", 0.7)
        )
        self._stream_buffer = np.zeros(0, dtype=np.float32)
        self._last_debug_time = 0.0
        self._last_empty_time = 0.0

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                segment, sample_rate = self.segment_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if self.drop_old_segments:
                dropped = 0
                while True:
                    try:
                        segment, sample_rate = self.segment_queue.get_nowait()
                        dropped += 1
                    except queue.Empty:
                        break
                if dropped and self.debug_audio:
                    logging.info("Dropped %s queued segments to reduce latency", dropped)

            if sample_rate != self.target_rate:
                segment = resample_linear(segment, sample_rate, self.target_rate)
                sample_rate = self.target_rate

            if segment.size:
                rms = float(np.sqrt(np.mean(segment**2)))
                peak = float(np.max(np.abs(segment)))
            else:
                rms = 0.0
                peak = 0.0

            if self.streaming:
                if self.min_silence_rms > 0 and rms < self.min_silence_rms:
                    self._stream_buffer = np.zeros(0, dtype=np.float32)
                    continue
                if segment.size:
                    if self._stream_buffer.size:
                        self._stream_buffer = np.concatenate([self._stream_buffer, segment])
                    else:
                        self._stream_buffer = segment.copy()
                max_samples = int(self.stream_window_seconds * sample_rate)
                if max_samples > 0 and self._stream_buffer.size > max_samples:
                    self._stream_buffer = self._stream_buffer[-max_samples:]
                min_samples = int(sample_rate * max(0.2, self.stream_update_seconds))
                if self._stream_buffer.size < min_samples:
                    continue
                audio_for_transcribe = self._stream_buffer
            else:
                if self.min_silence_rms > 0 and rms < self.min_silence_rms:
                    continue
                audio_for_transcribe = segment

            if self.debug_audio:
                now = time.time()
                if now - self._last_debug_time >= self.debug_interval:
                    logging.info(
                        "Audio level rms=%.5f peak=%.5f sr=%s", rms, peak, sample_rate
                    )
                    self._last_debug_time = now

            try:
                text = self.transcriber.transcribe(audio_for_transcribe, sample_rate, self.language)
            except Exception as exc:
                logging.exception("Transcription error: %s", exc)
                continue

            if not text and self.debug_audio:
                now = time.time()
                if now - self._last_empty_time >= self.debug_interval:
                    logging.info("No text for segment (rms=%.5f peak=%.5f)", rms, peak)
                    self._last_empty_time = now
            if text:
                if self.debug_text:
                    preview = text
                    if len(preview) > self.debug_text_max_chars:
                        preview = preview[: self.debug_text_max_chars].rstrip() + "..."
                    logging.info("Transcript: %s", preview)
                self.text_callback(text)


def build_transcriber(config: dict):
    engine = str(config["transcription"].get("engine", "faster-whisper")).lower()
    if engine in ("faster-whisper", "whisper"):
        return FasterWhisperTranscriber(
            model_name=config["transcription"].get("model", "base"),
            device=config["transcription"].get("device", "cpu"),
            compute_type=config["transcription"].get("compute_type", "int8"),
            vad_filter=bool(config["transcription"].get("vad_filter", True)),
        )
    if engine in ("openai", "api"):
        online_cfg = config.get("online", {})
        if not online_cfg.get("enable", False):
            raise ValueError("online.enable must be true for openai engine")
        return OpenAITranscriber(
            api_key_env=online_cfg.get("api_key_env", "OPENAI_API_KEY"),
            model=online_cfg.get("model", "gpt-4o-mini-transcribe"),
        )
    raise ValueError(f"Unsupported engine: {engine}")


class FasterWhisperTranscriber:
    def __init__(self, model_name: str, device: str, compute_type: str, vad_filter: bool) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.vad_filter = vad_filter

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str) -> str:
        lang = None if language == "auto" else language
        segments, _info = self.model.transcribe(
            audio,
            language=lang,
            task="transcribe",
            beam_size=1,
            vad_filter=self.vad_filter,
            condition_on_previous_text=False,
        )
        texts = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                texts.append(text)
        return " ".join(texts).strip()


class OpenAITranscriber:
    def __init__(self, api_key_env: str, model: str) -> None:
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"Missing API key in env var: {api_key_env}")
        self.api_key = api_key
        self.model = model

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str) -> str:
        import requests

        wav_bytes = float32_to_wav_bytes(audio, sample_rate)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"model": self.model}
        if language and language != "auto":
            data["language"] = language

        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers=headers,
            data=data,
            files=files,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("text", "")).strip()
