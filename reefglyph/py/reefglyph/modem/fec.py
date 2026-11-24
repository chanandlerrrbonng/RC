"""Whitening, Hamming(7,4)/(8,4), block interleaver, Gray mapping."""
import numpy as np
from ..spec import bits_to_int as b2i, int_to_bits as i2b


def whiten(bits):
    s, out = 0xFF, []
    for _ in range(len(bits)):
        out.append(s & 1)
        fb = ((s >> 7) ^ (s >> 5) ^ (s >> 4) ^ (s >> 3)) & 1
        s = ((s << 1) | fb) & 0xFF
    return np.asarray(bits, np.uint8) ^ np.array(out, np.uint8)


def _cw(nib, n):
    d = [(nib >> 3) & 1, (nib >> 2) & 1, (nib >> 1) & 1, nib & 1]
    p1, p2, p3 = d[0] ^ d[1] ^ d[3], d[0] ^ d[2] ^ d[3], d[1] ^ d[2] ^ d[3]
    w = d + [p1, p2, p3]
    if n == 8:
        w.append(sum(w) & 1)
    return w


_ENC = {n: np.array([_cw(i, n) for i in range(16)], np.uint8) for n in (7, 8)}
_DEC = {}
for _n in (7, 8):
    _tab = np.zeros(2**_n, np.uint8)
    for _w in range(2**_n):
        _wb = i2b(_w, _n)
        _tab[_w] = np.argmin([(_wb != _ENC[_n][k]).sum() for k in range(16)])
    _DEC[_n] = _tab


def gray_enc(v): return v ^ (v >> 1)


def gray_dec(g):
    v = 0
    while g:
        v ^= g; g >>= 1
    return v


def encode_bits(bits, sf, n):
    """bits (multiple of 4) -> symbol ints (bins) for one frame."""
    w = whiten(bits)
    nibs = w.reshape(-1, 4)
    cws = np.array([_ENC[n][b2i(x)] for x in nibs])               # (ncw, n)
    pad = (-len(cws)) % sf
    cws = np.vstack([cws, np.zeros((pad, n), np.uint8)])
    syms = []
    for blk in cws.reshape(-1, sf, n):
        for j in range(n):                                          # symbol j = bit j of each codeword
            syms.append(gray_dec(b2i(blk[:, j])))
    return np.array(syms)


def decode_syms(syms, sf, n, nbits):
    cws = []
    vals = [gray_enc(int(s)) for s in syms]
    for k in range(0, len(vals), n):
        cols = np.array([i2b(v, sf) for v in vals[k:k + n]])      # (n, sf)
        cws.append(cols.T)                                          # (sf, n)
    cws = np.vstack(cws)[: nbits // 4]
    nibs = [_DEC[n][b2i(c)] for c in cws]
    bits = np.concatenate([i2b(int(x), 4) for x in nibs])
    return whiten(bits)
