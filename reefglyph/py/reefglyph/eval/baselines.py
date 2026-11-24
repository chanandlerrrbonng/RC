"""M9 baselines. Run from project root:  uv run python -m reefglyph.eval.baselines [N]
Uses the same held-out split as calibrate.py (N images max, default 200). Writes data/baselines.csv and .png.
Baselines see a 128x128 squashed copy of each crop; the decoded image goes through the same M2 detector.
B1 header size and B3 coefficient cost are estimates, not real bitstreams."""
import glob, sys, numpy as np, cv2, pywt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from ..analysis import coral

BUDGETS, OUT = [600, 1500, 3000, 6000], 128
up = lambda im: cv2.resize(im, (OUT, OUT), interpolation=cv2.INTER_LINEAR)
_h = {}


def jpeg_hdr(q):   # header bytes ~ size of a flat 8x8 JPEG minus its few entropy bytes
    if q not in _h:
        _h[q] = len(cv2.imencode(".jpg", np.full((8, 8, 3), 128, np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])[1]) - 8
    return _h[q]


def b1(src, bits):   # headerless JPEG: largest size with quality>=25 whose entropy bytes fit
    for s in (128, 96, 64, 48, 32, 24, 16):
        im = cv2.resize(src, (s, s), interpolation=cv2.INTER_AREA)
        for q in range(90, 4, -5):
            buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1]
            if (len(buf) - jpeg_hdr(q)) * 8 <= bits and q >= 25:
                return up(cv2.imdecode(buf, 1))
    return up(cv2.resize(src, (16, 16), interpolation=cv2.INTER_AREA))


def b2(src, bits):   # downscaled palette image: k colours, log2(k) bits/pixel + 16-bit palette entries
    k = 4 if bits < 1000 else 8 if bits < 4000 else 16
    s = max(4, int(np.sqrt((bits - 16 * k) / np.log2(k))))
    px = cv2.resize(src, (s, s), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
    pal = coral.median_cut(px, k)
    idx = ((px[:, None] - pal[None]) ** 2).sum(-1).argmin(1)
    return up(pal.round().astype(np.uint8)[idx].reshape(s, s, 3))


def b3(src, bits):   # Haar, 32x32 YCrCb, keep K largest coeffs; cost/coef ~ sign + 5b magnitude + position
    s = 32
    im = cv2.cvtColor(cv2.resize(src, (s, s), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2YCrCb).astype(np.float32) - 128
    A = [pywt.coeffs_to_array(pywt.wavedec2(im[..., c], "haar", level=3)) for c in range(3)]
    flat = np.concatenate([a.ravel() for a, _ in A]); N = len(flat); step = np.abs(flat).max() / 31 + 1e-9
    K = next((k for k in range(N, 0, -1) if k * (6 + np.log2(N / k) + 1.44) <= bits), 1)
    r = np.zeros(N, np.float32); keep = np.argsort(-np.abs(flat))[:K]
    r[keep] = np.round(flat[keep] / step) * step
    ch = []
    for c, (a, sl) in enumerate(A):
        co = pywt.array_to_coeffs(r[c * a.size:(c + 1) * a.size].reshape(a.shape), sl, output_format="wavedec2")
        ch.append(pywt.waverec2(co, "haar"))
    out = np.clip(np.stack(ch, -1) + 128, 0, 255).astype(np.uint8)
    return up(cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR))


def bleach(im):
    h, w = im.shape[:2]
    return coral.analyze(im, (w // 10, h // 10, w * 8 // 10, h * 8 // 10), with_evidence=False).bleach


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    files, y = [], []
    for lab, d in ((0, "healthy"), (1, "bleached")):      # same order and filtering as calibrate.load()
        for f in sorted(glob.glob(f"data/crops/{d}/*")):
            if cv2.imread(f) is not None: files.append(f); y.append(lab)
    y = np.array(y); _, ih = train_test_split(np.arange(len(y)), test_size=.5, stratify=y, random_state=0)
    ih = np.random.default_rng(0).permutation(ih)[:n]; tau = coral.PARAMS["tau"]
    rows = {}
    for c, i in enumerate(ih):
        im = cv2.imread(files[i]); src = cv2.resize(im, (OUT, OUT), interpolation=cv2.INTER_AREA)
        rows.setdefault(("ReefGlyph L0", 76), []).append((round(bleach(im) * 63) / 63, y[i]))   # 6-bit bleach index
        for name, fn in (("B1 JPEG", b1), ("B2 palette", b2), ("B3 Haar", b3)):
            for b in BUDGETS: rows.setdefault((name, b), []).append((bleach(fn(src, b)), y[i]))
        if c % 20 == 0: print(f"{c}/{len(ih)}")
    acc = lambda v: np.mean((np.array(v)[:, 0] > tau) == np.array(v)[:, 1])
    with open("data/baselines.csv", "w") as fo:
        fo.write("method,bits,auc,accuracy\n")
        for (m, b), v in rows.items():
            s, t = np.array(v).T; line = f"{m},{b},{roc_auc_score(t, s):.3f},{acc(v):.3f}"
            fo.write(line + "\n"); print(line)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for m in ("B1 JPEG", "B2 palette", "B3 Haar"):
        plt.plot(BUDGETS, [acc(rows[(m, b)]) for b in BUDGETS], "o-", label=m)
    plt.plot([76], [acc(rows[("ReefGlyph L0", 76)])], "r*", ms=14, label="ReefGlyph L0 (76 b)")
    plt.xscale("log"); plt.xlabel("bits"); plt.ylabel("detector accuracy"); plt.legend()
    plt.savefig("data/baselines.png", dpi=150); print("saved data/baselines.csv, data/baselines.png")
