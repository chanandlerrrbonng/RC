"""M2: classical-CV coral analysis. analyze(bgr, rect) -> CoralRecord."""
import json, os
import numpy as np, cv2, pyefd
from ..codec.layers import CoralRecord

PARAMS = dict(L_t=60.0, C_t=25.0, tau=0.35, norm=0, seg="gc")   # overwritten by data/params.json
MAXS = 320                                                       # analysis resolution (longest side)
HEALTH_EDGES = None                               # 5 ascending edges from the calibration split


def load_params(path="data/params.json"):
    """Called automatically at import if data/params.json exists (run from project root)."""
    global HEALTH_EDGES
    if os.path.exists(path):
        d = json.load(open(path))
        PARAMS.update({k: d[k] for k in d if k in PARAMS})
        HEALTH_EDGES = d.get("health_edges")


load_params()


def to_lab(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128


def gray_world(bgr):
    m = bgr.reshape(-1, 3).mean(0); return np.clip(bgr * (m.mean() / m), 0, 255).astype(np.uint8)


def median_cut(px, k):
    boxes = [px]
    while len(boxes) < k:
        i = int(np.argmax([len(b) > 1 and np.ptp(b, 0).max() for b in boxes])); b = boxes.pop(i)
        ch = int(np.argmax(np.ptp(b, 0))); b = b[np.argsort(b[:, ch])]; h = max(1, len(b) // 2)
        boxes += [b[:h], b[h:]] if len(b) > 1 else [b, b]
    return np.array([b.mean(0) for b in boxes])


def segment(bgr, rect):
    mask = np.zeros(bgr.shape[:2], np.uint8); bg = np.zeros((1, 65)); fg = np.zeros((1, 65))
    cv2.grabCut(bgr, mask, tuple(int(v) for v in rect), bg, fg, 4, cv2.GC_INIT_WITH_RECT)
    return ((mask == 1) | (mask == 3)).astype(np.uint8)


def rect_mask(bgr, rect):
    m = np.zeros(bgr.shape[:2], np.uint8); x, y, w, h = (int(v) for v in rect)
    m[y:y + h, x:x + w] = 1; return m


def pixel_stats(bgr, mask):
    """Eroded-mask L and chroma (brightest 1 % dropped) + the eroded mask."""
    er = cv2.erode(mask, np.ones((7, 7), np.uint8))           # ~3 px erosion
    if er.sum() < 20: er = mask
    if er.sum() == 0: er = np.ones_like(mask)
    L, a, b = to_lab(bgr); sel = er > 0
    Ls, C = L[sel], np.hypot(a[sel], b[sel])
    keep = Ls <= np.percentile(Ls, 99)
    return Ls[keep], C[keep], er


def bleach_from_stats(Ls, C, p=PARAMS):
    return float(np.mean((Ls > p["L_t"]) & (C < p["C_t"])))


def health_x(Ls, C):
    return float(Ls.mean() - C.mean())                         # pale + low chroma -> high


def health_score(x):
    if HEALTH_EDGES is None: return 3
    return int(np.searchsorted(HEALTH_EDGES, x)) + 1           # 1..6


def bleach_index(bgr, mask, p=PARAMS):
    Ls, C, er = pixel_stats(bgr, mask)
    return bleach_from_stats(Ls, C, p), er


def analyze(bgr, rect, normalise=None, params=PARAMS, with_evidence=True):
    """rect = (x, y, w, h). Returns CoralRecord."""
    s = MAXS / max(bgr.shape[:2])
    if s < 1:
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        rect = tuple(int(v * s) for v in rect)
    mask = rect_mask(bgr, rect) if params.get("seg") == "rect" else segment(bgr, rect)
    if (params.get("norm", 0) if normalise is None else normalise): bgr = gray_world(bgr)
    Ls, C, er = pixel_stats(bgr, mask)
    bi = bleach_from_stats(Ls, C, params)
    r = CoralRecord(bleach=bi, health=min(6, max(1, health_score(health_x(Ls, C)))), morph=3)
    r.evidence = int(bi > params["tau"])
    px = bgr[er > 0].astype(np.float32)
    cols = median_cut(px, 2)
    r.colors = [tuple(float(v[0, 0]) for v in to_lab(c.reshape(1, 1, 3).astype(np.uint8))) for c in cols]
    if not with_evidence: return r
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return r
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    box = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.float32)
    S = 64; dst = np.float32([[0, S - 1], [0, 0], [S - 1, 0], [S - 1, S - 1]])
    M = cv2.getPerspectiveTransform(box, dst)
    wm = cv2.warpPerspective(mask * 255, M, (S, S)); wi = cv2.warpPerspective(bgr, M, (S, S))
    r.aspect = float(np.clip(w / max(h, 1), 0.25, 4))
    cs, _ = cv2.findContours((wm > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs: return r
    cc = max(cs, key=cv2.contourArea)[:, 0, :].astype(float) / (S - 1)
    r.efd = np.clip(pyefd.elliptic_fourier_descriptors(cc, order=8, normalize=False), -.5, .5)
    r.locus = tuple(float(v) for v in np.clip(pyefd.calculate_dc_coefficients(cc), 0, 1))
    th = cv2.resize(wi, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
    thm = cv2.resize(wm, (16, 16)).reshape(-1) > 127
    pal = median_cut(th[thm] if thm.sum() >= 4 else th, 4)
    d = ((th[:, None, :] - pal[None]) ** 2).sum(-1)
    r.thumb = d.argmin(1).reshape(16, 16)
    r.palette = pal[:, ::-1].round().astype(np.uint8)          # RGB
    return r