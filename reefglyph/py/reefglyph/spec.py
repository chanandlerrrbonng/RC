"""M1: frozen constants. Treat as frozen after Day 1."""
import numpy as np

FS_AUDIO = 48000; BW = 3000; FC = 2500; DEFAULT_SF = 7
OSR = FS_AUDIO // BW            # 16
N_UP, N_DOWN = 8, 2
PREAMBLE_SYMS = N_UP + N_DOWN
HDR_BITS, PAY_BITS, CRC_BITS = 16, 108, 16
FRAME_BITS = HDR_BITS + PAY_BITS + CRC_BITS   # 140
L0_BITS, L1_BITS, L2_BITS = 80, 175, 576
LAYER_BITS = {0: L0_BITS, 1: L1_BITS, 2: L2_BITS}
LAYER_FRAMES = {0: 1, 1: 2, 2: 6}
ACK_BITS = 28


def frame_symbols(sf=DEFAULT_SF, first=False):
    n = 8 if first else 7
    return -(-35 // sf) * n


def ack_symbols(sf=DEFAULT_SF):
    return -(-7 // sf) * 8          # 7 codewords, Hamming(8,4)


def airtime(burst, sf=DEFAULT_SF):
    """burst: 'short' | 'long' | 'ack' (or number of data frames incl. F0)"""
    if burst == "ack":
        syms = PREAMBLE_SYMS + ack_symbols(sf)
    else:
        nf = {"short": 1, "long": 9}.get(burst, burst)
        syms = PREAMBLE_SYMS + frame_symbols(sf, True) + (nf - 1) * frame_symbols(sf)
    return syms * (2 ** sf) / BW


def bits_to_int(b):
    v = 0
    for x in b:
        v = (v << 1) | int(x)
    return v


def int_to_bits(v, n):
    return np.array([(int(v) >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)
