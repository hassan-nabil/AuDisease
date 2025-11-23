from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf

from data_pipeline import _clean_column_names, load_raw_data


def get_feature_columns() -> List[str]:
    """
    Return the ordered list of feature column names used by the tabular model.

    This mirrors the logic used in the training pipeline: drop the 'name'
    column and keep everything except the 'status' target.
    """
    df = load_raw_data()
    df = _clean_column_names(df)
    df = df.drop(columns=["name"])
    cols = [c for c in df.columns if c != "status"]
    return cols


def _read_wav_mono_from_path(path: Path) -> tuple[np.ndarray, int]:
    """
    Read an audio file from disk and return (mono_waveform, sample_rate).

    Uses soundfile, which supports WAV (including WAVE_FORMAT_EXTENSIBLE / 65534)
    and several other common formats.
    """
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def _read_wav_mono_from_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Read audio data from bytes and return (mono_waveform, sample_rate).
    """
    # soundfile can read from a file-like object
    import io

    bio = io.BytesIO(audio_bytes)
    audio, sr = sf.read(bio, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def _extract_basic_features(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract a small, robust set of audio features from a mono waveform.

    These are generic time- and frequency-domain statistics that work across
    different speakers and recording conditions:
    - duration (seconds)
    - mean energy
    - zero-crossing rate
    - spectral centroid
    - spectral bandwidth
    - mean amplitude
    - standard deviation of amplitude
    """
    if waveform.size == 0 or sr <= 0:
        return np.zeros(7, dtype=np.float32)

    x = waveform.astype(np.float32)
    x = x - float(x.mean())

    duration = x.size / float(sr)
    energy = float(np.mean(x**2))

    # Zero-crossing rate
    zc = np.mean(np.abs(np.sign(x[1:]) - np.sign(x[:-1])) > 0).item()

    # Frequency-domain stats on a single Hann-windowed frame
    max_len = min(x.size, 16000)
    frame = x[:max_len]
    if frame.size < 256:
        return np.array(
            [duration, energy, 0.0, 0.0, 0.0, float(x.mean()), float(x.std())],
            dtype=np.float32,
        )

    window = np.hanning(frame.size).astype(np.float32)
    frame = frame * window
    spec = np.fft.rfft(frame)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(frame.size, d=1.0 / sr)

    mag_sum = float(mag.sum()) or 1.0
    centroid = float((freqs * mag).sum() / mag_sum)
    bandwidth = float(
        np.sqrt(((freqs - centroid) ** 2 * mag).sum() / mag_sum)
    )

    mean_amp = float(x.mean())
    std_amp = float(x.std())

    return np.array(
        [duration, energy, zc, centroid, bandwidth, mean_amp, std_amp],
        dtype=np.float32,
    )


def extract_audio_features_from_wav_file(path: Path) -> np.ndarray:
    """
    Convenience helper for training: extract features from a WAV file on disk.
    """
    waveform, sr = _read_wav_mono_from_path(path)
    return _extract_basic_features(waveform, sr)


def extract_audio_features_from_wav_bytes(audio_bytes: bytes) -> np.ndarray:
    """
    Extract features from WAV bytes (used at prediction time).
    """
    waveform, sr = _read_wav_mono_from_bytes(audio_bytes)
    return _extract_basic_features(waveform, sr)



