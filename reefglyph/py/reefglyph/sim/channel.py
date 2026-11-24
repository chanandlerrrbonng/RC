"""M6: channel simulator (band-limit, multipath, noise, ppm, delay, clipping)."""
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from ..spec import FS_AUDIO


def channel(x, snr_db=15, ppm=0, delay_s=0.0, taps=None, ir=None, clip=None, seed=0):
    """taps: [(delay_s, gain), ...]; ir: measured impulse response @48 kHz (M6 replay).
    snr_db is broadband (24 kHz) SNR; in-band SNR is ~9 dB higher."""
    rng = np.random.default_rng(seed)
    y = np.asarray(x, np.float64)
    if ir is not None: y = fftconvolve(y, ir)
    if taps:
        h = np.zeros(int(max(d for d, _ in taps) * FS_AUDIO) + 1); h[0] = 1
        for d, g in taps: h[int(d * FS_AUDIO)] += g
        y = fftconvolve(y, h)
    sos = butter(6, [900, 4100], btype="band", fs=FS_AUDIO, output="sos"); y = sosfilt(sos, y)
    if ppm:                                    # clock mismatch via resampling
        t = np.arange(len(y)) * (1 + ppm * 1e-6); y = np.interp(t, np.arange(len(y)), y, right=0)
    y = np.concatenate([np.zeros(int(delay_s * FS_AUDIO)), y])
    act = np.abs(y) > 1e-4
    p = np.mean(y[act] ** 2) if np.any(act) else 1
    y = y + rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), len(y))
    if clip: y = np.clip(y, -clip, clip)
    return y.astype(np.float32)


def write_wav(path, x):
    from scipy.io import wavfile
    wavfile.write(path, FS_AUDIO, (np.clip(x, -1, 1) * 32767).astype(np.int16))
