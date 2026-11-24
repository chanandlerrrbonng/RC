"""M3: pack/unpack of L0, L1, L2."""
import numpy as np
from dataclasses import dataclass, field
from ..spec import bits_to_int as b2i, int_to_bits as i2b


def _q(v, lo, hi, bits):
    return int(np.clip(round((v - lo) / (hi - lo) * (2**bits - 1)), 0, 2**bits - 1))


def _dq(q, lo, hi, bits):
    return lo + q / (2**bits - 1) * (hi - lo)


def lab_q(lab):  # L 0-100 (6b), a,b -> 5b, step 4
    L, a, b = lab
    return [_q(L, 0, 100, 6), int(np.clip(round(a / 4) + 16, 0, 31)), int(np.clip(round(b / 4) + 16, 0, 31))]


def lab_dq(q):
    return (_dq(q[0], 0, 100, 6), (q[1] - 16) * 4.0, (q[2] - 16) * 4.0)


@dataclass
class CoralRecord:
    transect: int = 0; coral: int = 0; depth_m: float = 0.0; time_s: float = 0.0
    bleach: float = 0.0; health: int = 1; morph: int = 3     # 0 branching 1 massive 2 plate 3 other
    colors: list = field(default_factory=lambda: [(50, 0, 0), (50, 0, 0)])
    patch: int = 0; evidence: int = 0; texture: int = 0
    efd: np.ndarray = None      # (8,4) in unit box
    locus: tuple = (0.5, 0.5); aspect: float = 1.0
    thumb: np.ndarray = None    # 16x16 ints 0..3
    palette: np.ndarray = None  # (4,3) uint8 RGB


def pack_L0(r):
    f = [(r.transect, 6), (r.coral, 8), (_q(r.depth_m, 0, 63.5, 7), 7), (_q(r.time_s, 0, 5115, 10), 10),
         (_q(r.bleach, 0, 1, 6), 6), (r.health - 1, 3), (r.morph, 2)]
    for c in r.colors:
        q = lab_q(c); f += [(q[0], 6), (q[1], 5), (q[2], 5)]
    f += [(r.patch, 1), (r.evidence, 1), (int(np.clip(r.texture, 0, 15)), 4)]
    out = np.concatenate([i2b(v, n) for v, n in f]); assert len(out) == 80
    return out


def unpack_L0(bits):
    p = [0]

    def rd(n):
        v = b2i(bits[p[0]:p[0] + n]); p[0] += n; return v
    r = CoralRecord()
    r.transect, r.coral = rd(6), rd(8)
    r.depth_m = _dq(rd(7), 0, 63.5, 7); r.time_s = _dq(rd(10), 0, 5115, 10)
    r.bleach = _dq(rd(6), 0, 1, 6); r.health = rd(3) + 1; r.morph = rd(2)
    r.colors = [lab_dq([rd(6), rd(5), rd(5)]) for _ in range(2)]
    r.patch, r.evidence = rd(1), rd(1)
    r.texture = rd(4)
    return r


EFD_RANGE = 0.5


def pack_L1(r):
    out = [i2b(_q(v, -EFD_RANGE, EFD_RANGE, 5), 5) for v in np.asarray(r.efd).ravel()]
    out += [i2b(_q(v, 0, 1, 5), 5) for v in r.locus]
    out.append(i2b(_q(np.log2(np.clip(r.aspect, .25, 4)), -2, 2, 5), 5))
    out = np.concatenate(out); assert len(out) == 175
    return out


def unpack_L1(bits, r=None):
    r = r or CoralRecord()
    v = [_dq(b2i(bits[i * 5:i * 5 + 5]), -EFD_RANGE, EFD_RANGE, 5) for i in range(32)]
    r.efd = np.array(v).reshape(8, 4)
    r.locus = tuple(_dq(b2i(bits[160 + 5 * i:165 + 5 * i]), 0, 1, 5) for i in range(2))
    r.aspect = float(2 ** _dq(b2i(bits[170:175]), -2, 2, 5))
    return r


def pack_L2(r):
    px = np.concatenate([i2b(int(v), 2) for v in np.asarray(r.thumb).ravel()])
    pal = []
    for R, G, B in np.asarray(r.palette):
        pal.append(i2b(((int(R) >> 3) << 11) | ((int(G) >> 2) << 5) | (int(B) >> 3), 16))
    out = np.concatenate([px] + pal); assert len(out) == 576
    return out


def unpack_L2(bits, r=None):
    r = r or CoralRecord()
    r.thumb = np.array([b2i(bits[2 * i:2 * i + 2]) for i in range(256)]).reshape(16, 16)
    pal = []
    for i in range(4):
        v = b2i(bits[512 + 16 * i:528 + 16 * i])
        pal.append([((v >> 11) & 31) << 3, ((v >> 5) & 63) << 2, (v & 31) << 3])
    r.palette = np.array(pal, dtype=np.uint8)
    return r
