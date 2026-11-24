"""M5a: CSS modem. Chirps, sync, SNR estimate, modulate/demodulate."""
import numpy as np
from scipy.signal import resample_poly
from ..spec import *
from . import fec
from .framing import unpack_frame


def base_chirp(sf):
    N = 2 ** sf; n = np.arange(N)
    return np.exp(1j * (np.pi * n**2 / N - np.pi * n))


def sym_wave(k, sf, down=False):
    c = np.roll(base_chirp(sf), -int(k))      # base[(n+k) % N]
    return np.conj(c) if down else c


def preamble(sf):
    return np.concatenate([sym_wave(0, sf)] * N_UP + [sym_wave(0, sf, True)] * N_DOWN)


def to_audio(bb):
    x = resample_poly(bb, OSR, 1)
    t = np.arange(len(x)) / FS_AUDIO
    a = np.real(x * np.exp(2j * np.pi * FC * t))
    return (a / (np.max(np.abs(a)) + 1e-12) * 0.8).astype(np.float32)


def to_baseband(a):
    t = np.arange(len(a)) / FS_AUDIO
    return resample_poly(a * np.exp(-2j * np.pi * FC * t) * 2, 1, OSR)


def _frame_cfg(kind, i):
    """(codeword length n, bits per frame)."""
    if kind == "ack": return 8, 28
    return (8 if i == 0 else 7), FRAME_BITS


def _nsym(kind, i, sf):
    return ack_symbols(sf) if kind == "ack" else frame_symbols(sf, i == 0)


def modulate(frame_bits, sf=DEFAULT_SF, kind="data"):
    """frame_bits: list of 140-bit arrays (or [ack 28-bit]). Returns float32 audio @48 kHz."""
    parts = [preamble(sf)]
    for i, b in enumerate(frame_bits):
        n, _ = _frame_cfg(kind, i)
        syms = fec.encode_bits(np.asarray(b, np.uint8), sf, n)
        parts += [sym_wave(s, sf) for s in syms]
    parts.append(np.zeros(2 ** sf * 2, complex))
    return to_audio(np.concatenate(parts))


def _peak(seg, ref):
    N = len(ref); m = np.abs(np.fft.fft(seg * ref))
    p = int(np.argmax(m)); md = np.median(m) + 1e-12
    l, r = m[(p - 1) % N], m[(p + 1) % N]
    den = l - 2 * m[p] + r
    d = 0.5 * (l - r) / den if den != 0 else 0.0
    return p + d, m[p] / md


def _sw(v, N):  # wrap to signed
    return (v + N / 2) % N - N / 2


def sync(bb, sf, thr=7.0):
    """Returns dict(start, cfo (bins), snr_db (uncalibrated peak/median), snr4) or None."""
    N = 2 ** sf; up = base_chirp(sf); cu, cd = np.conj(up), up
    nw = len(bb) // N
    pk = [_peak(bb[g * N:(g + 1) * N], cu) for g in range(nw)]
    g0 = None
    for g in range(nw - 4):
        w = pk[g:g + 4]
        if all(x[1] > thr for x in w) and all(abs(_sw(x[0] - w[0][0], N)) < 2.5 for x in w):
            g0 = g; break
    if g0 is None: return None
    a = int(round(g0 * N - _sw(pk[g0][0], N)))
    while a < 0: a += N
    iD = None
    for i in range(0, 16):
        s = a + i * N
        if s + N > len(bb): return None
        _, ru = _peak(bb[s:s + N], cu); _, rd = _peak(bb[s:s + N], cd)
        if rd > ru and rd > thr: iD = i; break
    if iD is None or iD < 1: return None
    bu = _sw(_peak(bb[a + (iD - 1) * N:a + iD * N], cu)[0], N)
    bd = _sw(_peak(bb[a + iD * N:a + (iD + 1) * N], cd)[0], N)
    c = _sw((bu + bd) / 2, N / 2)        # CFO in bins
    tau = (bu - bd) / 2                   # window-late timing error in samples
    start = a + (iD + 2) * N - int(round(tau))
    ups = [_peak(bb[a + k * N:a + (k + 1) * N], cu)[1] for k in range(max(0, iD - 6), iD)]
    snr_db = 20 * np.log10(np.mean(ups))
    return dict(start=start, cfo=c, tau=tau, snr_db=float(snr_db),
                snr4=int(np.clip(round(snr_db / 2), 0, 15)))


def _track(hs, hd, d0, s):
    """Predicted timing offset D(s) (samples): true symbol start minus nominal start."""
    if len(hs) < 12: return d0 if len(hs) < 3 else float(np.median(hd))
    b1, a0 = np.polyfit(hs[-48:], hd[-48:], 1)
    return a0 + b1 * s


def demodulate(audio, sf=DEFAULT_SF, kind="data", max_frames=9):
    """Returns (list[Frame] | ack bits, info). Frames with bad CRC come back ok=False.
    Timing drift (clock ppm) is tracked from the fractional part of every dechirp peak:
    delta = wrap(f - applied + D_hat) = D_hat - D, independent of the decoded symbol."""
    N = 2 ** sf
    bb = to_baseband(np.asarray(audio, np.float64))
    s = sync(bb, sf)
    if s is None: return [], None
    bb = bb * np.exp(-2j * np.pi * s["cfo"] * np.arange(len(bb)) / N)       # CFO derotation
    cu = np.conj(base_chirp(sf))
    d0 = -(s["tau"] - round(s["tau"]))
    hs, hd = [], []
    sym_i = 0; out = []
    nframes = 1 if kind == "ack" else max_frames
    fi = 0
    while fi < nframes:
        n, nbits = _frame_cfg(kind, fi)
        bins, ok_win = [], True
        for j in range(_nsym(kind, fi, sf)):
            si = sym_i + j
            dh = _track(hs, hd, d0, si); ap = int(round(dh))
            st = s["start"] + si * N + ap
            if st < 0 or st + N > len(bb): ok_win = False; break
            f, _ = _peak(bb[st:st + N], cu)
            fc = f - ap + dh                                   # timing-corrected (fractional) bin
            bins.append(int(round(fc)) % N)
            hs.append(si); hd.append(dh - _sw(fc, 1.0))        # _sw(.,1): wrapped fractional part
        if not ok_win: break
        bits = fec.decode_syms(bins, sf, n, nbits)
        if kind == "ack": return bits, s
        fr = unpack_frame(bits); out.append(fr)
        if fr.ok and fi == 0: nframes = min(fr.n, max_frames)
        sym_i += len(bins); fi += 1
    return out, s
