import logging
import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from audio_utils import resample_linear


def format_device_list() -> str:
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    lines = []
    for idx, dev in enumerate(devices):
        hostapi_name = hostapis[dev["hostapi"]]["name"]
        lines.append(
            f"{idx}: {dev['name']} (in:{dev['max_input_channels']} out:{dev['max_output_channels']}) "
            f"hostapi:{hostapi_name} default_sr:{int(dev['default_samplerate'])}"
        )
    return "\n".join(lines)


def resolve_device_index(query: Optional[str], kind: str) -> Optional[int]:
    if query is None:
        return None
    if isinstance(query, int):
        return query
    text = str(query).strip()
    if text == "" or text.lower() == "default":
        return None
    if text.isdigit():
        return int(text)
    target = text.lower()
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if target in dev["name"].lower():
            if kind == "mic" and dev["max_input_channels"] > 0:
                return idx
            if kind == "system" and (
                dev["max_output_channels"] > 0 or dev["max_input_channels"] > 0
            ):
                return idx
    return None


def get_default_device_index(kind: str) -> Optional[int]:
    default_in, default_out = sd.default.device
    if kind == "mic":
        if default_in is not None and default_in >= 0:
            return default_in
        return None
    if default_out is not None and default_out >= 0:
        return default_out
    return None


def get_default_device_rate(kind: str, device_query: Optional[str]) -> int:
    idx = resolve_device_index(device_query, kind)
    if idx is None:
        idx = get_default_device_index(kind)
    devices = sd.query_devices()
    if idx is None:
        return int(devices[0]["default_samplerate"])
    return int(devices[idx]["default_samplerate"])


class AudioStreamSource:
    def __init__(
        self,
        kind: str,
        device_index: Optional[int],
        sample_rate: int,
        blocksize: int,
        gain: float,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        loopback: bool,
    ) -> None:
        self.kind = kind
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.gain = float(gain)
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.loopback = loopback
        self.stream = None

    def start(self) -> None:
        device = self.device_index
        if device is None:
            device = get_default_device_index(self.kind)
        extra_settings = _build_wasapi_loopback_settings(self.loopback)
        loopback_supported = self.loopback and extra_settings is not None
        if self.loopback and extra_settings is None:
            logging.warning(
                "WASAPI loopback is not supported by the installed sounddevice; "
                "use a Stereo Mix input device or upgrade sounddevice."
            )
            if self.kind == "system":
                fallback = _find_input_device_by_name(["stereo mix", "loopback"])
                if fallback is None:
                    fallback = _find_any_input_device()
                if fallback is not None:
                    device = fallback
        if device is not None and not loopback_supported:
            device_info = sd.query_devices(device)
            if device_info.get("max_input_channels", 0) <= 0:
                raise ValueError(
                    "Selected device has no input channels. "
                    "Pick a mic/Stereo Mix device or enable WASAPI loopback."
                )
        self.stream = sd.InputStream(
            device=device,
            channels=1,
            dtype="float32",
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            callback=self._callback,
            extra_settings=extra_settings,
        )
        self.stream.start()

    def _callback(self, indata, frames, time, status) -> None:
        if self.stop_event.is_set():
            return
        data = indata.copy()
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        data = data.astype(np.float32, copy=False)
        if self.gain != 1.0:
            data = np.clip(data * self.gain, -1.0, 1.0)
        try:
            self.output_queue.put_nowait(data)
        except queue.Full:
            pass

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


class AudioMixer(threading.Thread):
    def __init__(
        self,
        mic_queue: queue.Queue,
        system_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.mic_queue = mic_queue
        self.system_queue = system_queue
        self.output_queue = output_queue
        self.stop_event = stop_event

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                mic = self.mic_queue.get(timeout=0.1)
            except queue.Empty:
                mic = None
            try:
                sys = self.system_queue.get(timeout=0.1)
            except queue.Empty:
                sys = None

            if mic is None and sys is None:
                continue
            if mic is None:
                mixed = sys
            elif sys is None:
                mixed = mic
            else:
                min_len = min(len(mic), len(sys))
                if len(mic) != len(sys):
                    mic = mic[:min_len]
                    sys = sys[:min_len]
                mixed = mic + sys

            mixed = np.clip(mixed, -1.0, 1.0)
            try:
                self.output_queue.put_nowait(mixed)
            except queue.Full:
                pass


def _build_wasapi_loopback_settings(loopback: bool):
    if not loopback or not hasattr(sd, "WasapiSettings"):
        return None
    try:
        return sd.WasapiSettings(loopback=True)
    except TypeError:
        try:
            settings = sd.WasapiSettings()
        except TypeError:
            return None
        if hasattr(settings, "loopback"):
            try:
                settings.loopback = True
                return settings
            except Exception:
                return None
    return None


def _find_input_device_by_name(names: list[str]) -> Optional[int]:
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) <= 0:
            continue
        name = str(dev.get("name", "")).lower()
        if any(token in name for token in names):
            return idx
    return None


def _find_any_input_device() -> Optional[int]:
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            return idx
    return None


class AudioSegmenter(threading.Thread):
    def __init__(
        self,
        input_queue: queue.Queue,
        capture_rate: int,
        target_rate: int,
        chunk_seconds: float,
        overlap_seconds: float,
        output_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.capture_rate = capture_rate
        self.target_rate = target_rate
        self.chunk_samples = max(1, int(chunk_seconds * capture_rate))
        self.overlap_samples = max(0, int(overlap_seconds * capture_rate))
        if self.overlap_samples >= self.chunk_samples:
            self.overlap_samples = max(0, self.chunk_samples // 2)
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.buffer = np.zeros(0, dtype=np.float32)

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self.input_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame is None:
                continue
            self.buffer = np.concatenate([self.buffer, frame])
            while len(self.buffer) >= self.chunk_samples:
                segment = self.buffer[: self.chunk_samples]
                keep_from = max(0, self.chunk_samples - self.overlap_samples)
                self.buffer = self.buffer[keep_from:]
                if self.capture_rate != self.target_rate:
                    segment = resample_linear(segment, self.capture_rate, self.target_rate)
                    out_rate = self.target_rate
                else:
                    out_rate = self.capture_rate
                try:
                    self.output_queue.put_nowait((segment, out_rate))
                except queue.Full:
                    pass


class AudioPipeline:
    def __init__(self, config: dict, stop_event: threading.Event) -> None:
        audio_cfg = config["audio"]
        self.stop_event = stop_event
        self.source = audio_cfg.get("source", "mic")
        self.queue_max = int(audio_cfg.get("queue_max", 20))

        capture_rate = audio_cfg.get("capture_sample_rate")
        if self.source == "mix" and not capture_rate:
            raise ValueError("capture_sample_rate is required for audio.source = mix")
        if not capture_rate:
            if self.source == "system":
                capture_rate = get_default_device_rate("system", audio_cfg.get("system_device"))
            else:
                capture_rate = get_default_device_rate("mic", audio_cfg.get("mic_device"))
        self.capture_rate = int(capture_rate)
        self.target_rate = int(audio_cfg.get("target_sample_rate", self.capture_rate))

        block_ms = float(audio_cfg.get("block_ms", 30))
        self.blocksize = max(1, int(self.capture_rate * block_ms / 1000.0))
        gain = float(audio_cfg.get("gain", 1.0))

        self.frames_queue: queue.Queue = queue.Queue(maxsize=self.queue_max)
        self.segment_queue: queue.Queue = queue.Queue(maxsize=self.queue_max)

        self.sources = []
        self.mixer = None

        if self.source == "mix":
            mic_queue = queue.Queue(maxsize=self.queue_max)
            sys_queue = queue.Queue(maxsize=self.queue_max)

            mic_index = resolve_device_index(audio_cfg.get("mic_device"), "mic")
            sys_index = resolve_device_index(audio_cfg.get("system_device"), "system")

            self.sources.append(
                AudioStreamSource(
                    kind="mic",
                    device_index=mic_index,
                    sample_rate=self.capture_rate,
                    blocksize=self.blocksize,
                    gain=gain,
                    output_queue=mic_queue,
                    stop_event=self.stop_event,
                    loopback=False,
                )
            )
            self.sources.append(
                AudioStreamSource(
                    kind="system",
                    device_index=sys_index,
                    sample_rate=self.capture_rate,
                    blocksize=self.blocksize,
                    gain=gain,
                    output_queue=sys_queue,
                    stop_event=self.stop_event,
                    loopback=True,
                )
            )
            self.mixer = AudioMixer(mic_queue, sys_queue, self.frames_queue, self.stop_event)
        else:
            kind = "system" if self.source == "system" else "mic"
            device_index = resolve_device_index(
                audio_cfg.get("system_device") if kind == "system" else audio_cfg.get("mic_device"),
                kind,
            )
            self.sources.append(
                AudioStreamSource(
                    kind=kind,
                    device_index=device_index,
                    sample_rate=self.capture_rate,
                    blocksize=self.blocksize,
                    gain=gain,
                    output_queue=self.frames_queue,
                    stop_event=self.stop_event,
                    loopback=(kind == "system"),
                )
            )

        trans_cfg = config["transcription"]
        if trans_cfg.get("streaming", False):
            chunk_seconds = float(trans_cfg.get("stream_update_seconds", 0.7))
            overlap_seconds = 0.0
        else:
            chunk_seconds = float(trans_cfg.get("chunk_seconds", 5.0))
            overlap_seconds = float(trans_cfg.get("overlap_seconds", 1.0))
        self.segmenter = AudioSegmenter(
            input_queue=self.frames_queue,
            capture_rate=self.capture_rate,
            target_rate=self.target_rate,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            output_queue=self.segment_queue,
            stop_event=self.stop_event,
        )

    def start(self) -> None:
        for source in self.sources:
            source.start()
        if self.mixer is not None:
            self.mixer.start()
        self.segmenter.start()

    def stop(self) -> None:
        for source in self.sources:
            source.stop()
        if self.mixer is not None:
            self.mixer.join(timeout=1)
        self.segmenter.join(timeout=1)
