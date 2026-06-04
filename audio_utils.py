import io
import wave

import numpy as np


def to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    return np.mean(samples, axis=1)


def resample_linear(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return samples
    if len(samples) == 0:
        return samples.astype(np.float32)
    ratio = target_sr / float(orig_sr)
    target_len = int(len(samples) * ratio)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.arange(len(samples), dtype=np.float32)
    x_new = np.linspace(0, len(samples) - 1, num=target_len, dtype=np.float32)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def float32_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    samples = np.clip(samples, -1.0, 1.0)
    pcm16 = (samples * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()
