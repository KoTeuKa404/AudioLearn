import copy
import json
import os

DEFAULT_CONFIG = {
    "audio": {
        "source": "mix",
        "mic_device": "default",
        "system_device": "default",
        "capture_sample_rate": 48000,
        "target_sample_rate": 16000,
        "block_ms": 30,
        "queue_max": 20,
        "gain": 1.0,
    },
    "transcription": {
        "engine": "faster-whisper",
        "model": "base",
        "language": "auto",
        "device": "cpu",
        "compute_type": "int8",
        "chunk_seconds": 2.5,
        "overlap_seconds": 0.5,
        "min_silence_rms": 0.01,
        "vad_filter": True,
        "debug_audio": False,
        "debug_audio_interval": 5.0,
        "debug_text": False,
        "debug_text_max_chars": 160,
        "drop_old_segments": False,
        "streaming": False,
        "stream_window_seconds": 6.0,
        "stream_update_seconds": 0.7,
    },
    "ui": {
        "font_family": "Segoe UI",
        "font_size": 20,
        "max_lines": 4,
        "opacity": 0.75,
        "width_ratio": 0.85,
        "bottom_margin_px": 90,
        "padding_px": 14,
        "text_color": "#FFFFFF",
        "last_text_color": "#FFD54A",
        "background_color": "#000000",
        "click_through": True,
        "placeholder_text": "Listening...",
        "windowed": False,
        "force_topmost": False,
        "draggable": True,
    },
    "online": {
        "enable": False,
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini-transcribe",
    },
}


def deep_update(base: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value


def load_config(path: str) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            overrides = json.load(handle)
        if isinstance(overrides, dict):
            deep_update(config, overrides)
    return config
