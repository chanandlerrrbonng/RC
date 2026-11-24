"""M4: CRC, frames, burst assembly, ACK, ARQ state."""
import numpy as np
from dataclasses import dataclass
from ..spec import *


def crc16(bits):
    crc = 0xFFFF
    for b in bits:
        top = ((crc >> 15) & 1) ^ int(b); crc = (crc << 1) & 0xFFFF
        if top: crc ^= 0x1021
    return crc


def crc8(bits):
    crc = 0
    for b in bits:
        top = ((crc >> 7) & 1) ^ int(b); crc = (crc << 1) & 0xFF
        if top: crc ^= 0x07
    return crc


@dataclass
class Frame:
    layer: int; idx: int; n: int; burst: int; payload: np.ndarray; ok: bool = True


def pack_frame(f):
    hdr = np.concatenate([int_to_bits(f.layer, 2), int_to_bits(f.idx, 4), int_to_bits(f.n, 4), int_to_bits(f.burst, 6)])
    pay = np.zeros(PAY_BITS, np.uint8); pay[:len(f.payload)] = f.payload
    body = np.concatenate([hdr, pay])
    return np.concatenate([body, int_to_bits(crc16(body), 16)])


def unpack_frame(bits):
    body, crc = bits[:124], bits_to_int(bits[124:])
    ok = crc16(body) == crc
    return Frame(bits_to_int(bits[0:2]), bits_to_int(bits[2:6]), bits_to_int(bits[6:10]),
                 bits_to_int(bits[10:16]), bits[16:124].copy(), ok)


def packetize(layers, burst_id, only=None):
    """layers: {0:bits,1:bits,2:bits}; L0 always first. only: set of frame indices to (re)send.
    NOTE: the first frame of any burst is modulated with Hamming(8,4) *by position*."""
    chunks = []
    for L in sorted(layers):
        b = np.asarray(layers[L], np.uint8)
        for k in range(LAYER_FRAMES[L]):
            chunks.append((L, b[k * PAY_BITS:(k + 1) * PAY_BITS]))
    n = len(chunks)
    return [pack_frame(Frame(L, i, n, burst_id, p)) for i, (L, p) in enumerate(chunks)
            if only is None or i in only]


class Reassembler:
    """Keeps partial layers across attempts (same burst id)."""
    def __init__(self): self.got = {}; self.n = 9; self.burst = None

    def add(self, frames):
        for f in frames:
            if f.ok:
                self.n, self.burst = f.n, f.burst
                self.got[f.idx] = (f.layer, f.payload)

    def bitmap(self): return [int(i in self.got) for i in range(9)]
    def missing(self): return [i for i in range(self.n) if i not in self.got]

    def layers(self):
        """Return {layer: bits} for layers whose frames all arrived."""
        out, base = {}, 0
        for L in (0, 1, 2):
            idx = range(base, base + LAYER_FRAMES[L]); base += LAYER_FRAMES[L]
            if base > self.n: break
            if all(i in self.got for i in idx):
                out[L] = np.concatenate([self.got[i][1] for i in idx])[:LAYER_BITS[L]]
        return out


def pack_ack(burst_id, bitmap, snr4):
    bm = np.zeros(9, np.uint8); bm[:len(bitmap)] = bitmap
    body = np.concatenate([int_to_bits(burst_id, 6), bm, int_to_bits(snr4, 4)])   # 19 bits
    pad = np.concatenate([body, np.zeros(5, np.uint8)])                           # 24 bits for CRC calc
    return np.concatenate([body, int_to_bits(crc8(pad), 8), [0]]).astype(np.uint8)


def unpack_ack(bits):
    body = bits[:19]
    ok = crc8(np.concatenate([body, np.zeros(5, np.uint8)])) == bits_to_int(bits[19:27])
    return dict(ok=ok, burst=bits_to_int(bits[0:6]), bitmap=list(bits[6:15]), snr4=bits_to_int(bits[15:19]))


ACK_START_S, ACK_WINDOW_S, MAX_ATTEMPTS = 0.3, 3.0, 3
