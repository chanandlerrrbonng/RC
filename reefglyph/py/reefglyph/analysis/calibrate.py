"""Run from project root:  uv run python -m reefglyph.analysis.calibrate
Delete data/stats_cache.pkl only if you change how images are loaded."""
import glob, itertools, json, os, pickle, time, numpy as np, cv2
from sklearn.metrics import precision_recall_fscore_support as prf, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from . import coral

CACHE, NPX = "data/stats_cache.pkl", 3000
RECALL_TARGET = 0.90


def load():
    if os.path.exists(CACHE): return pickle.load(open(CACHE, "rb"))
    rng = np.random.default_rng(0); V, y, t0 = {}, [], time.time()
    for lab, d in ((0, "healthy"), (1, "bleached")):
        for f in sorted(glob.glob(f"data/crops/{d}/*")):
            im = cv2.imread(f)
            if im is None: continue
            s = coral.MAXS / max(im.shape[:2])
            if s < 1: im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            h, w = im.shape[:2]; rect = (w // 10, h // 10, w * 8 // 10, h * 8 // 10)
            masks = {"gc": coral.segment(im, rect), "rect": coral.rect_mask(im, rect)}
            for norm in (0, 1):
                img = coral.gray_world(im) if norm else im
                for seg, m in masks.items():
                    Ls, C, _ = coral.pixel_stats(img, m)
                    k = rng.choice(len(Ls), min(NPX, len(Ls)), replace=False)
                    V.setdefault((seg, norm), []).append((Ls[k], C[k]))
            y.append(lab)
    print(f"{len(y)} images, {(time.time() - t0) / max(1, len(y)):.2f} s/image (all variants)")
    out = (V, np.array(y)); pickle.dump(out, open(CACHE, "wb")); return out


def frac(S, Lt, Ct): return np.array([np.mean((L > Lt) & (C < Ct)) for L, C in S])


def tune(S, y):
    best = (-1, None)
    for Lt, Ct in itertools.product(range(0, 95, 5), list(range(4, 65, 4)) + [999]):
        a = roc_auc_score(y, frac(S, Lt, Ct))
        if a > best[0]: best = (a, (Lt, Ct))
    return best


if __name__ == "__main__":
    V, y = load()
    print(f"class balance: {y.mean():.2f} bleached, n={len(y)}")
    ic, ih = train_test_split(np.arange(len(y)), test_size=.5, stratify=y, random_state=0)
    pick = lambda S, i: [S[j] for j in i]
    for k, S in V.items():                                   # ceiling check, no tuning
        fL = [L.mean() for L, C in S]; fC = [-C.mean() for L, C in S]
        print(f"{k} single-feature AUC  meanL {roc_auc_score(y, fL):.3f}  -meanChroma {roc_auc_score(y, fC):.3f}")
    res = {}
    for k, S in V.items():
        a, (Lt, Ct) = tune(pick(S, ic), y[ic])
        ah = roc_auc_score(y[ih], frac(pick(S, ih), Lt, Ct))
        res[k] = (a, ah, Lt, Ct); print(f"{k}: cal AUC {a:.3f} held-out AUC {ah:.3f}  L_t={Lt} C_t={Ct}")
    k = max(res, key=lambda k: res[k][0]); a, ah, Lt, Ct = res[k]; S = V[k]
    sc = frac(pick(S, ic), Lt, Ct); fpr, tpr, th = roc_curve(y[ic], sc)
    tau = float(th[np.where(tpr >= RECALL_TARGET)[0][0]]) - 1e-9   # lowest-FPR threshold with >=90% recall on cal split
    sh = frac(pick(S, ih), Lt, Ct)
    p, r, f, _ = prf(y[ih], sh > tau, average="binary", zero_division=0)
    print(f"CHOSEN {k}: tau={tau:.3f}  HELD-OUT precision {p:.3f} recall {r:.3f} F1 {f:.3f} AUC {ah:.3f}")
    if ah < .75: print("WARNING: AUC < 0.75 -> look at the data/segmentation, not the thresholds")
    if Lt == 0: print("NOTE: L_t=0 -> lightness unused; report this as a chroma-only detector")
    if Ct == 999: print("NOTE: C_t=999 -> chroma unused")
    if Lt == 90 or Ct == 4: print("WARNING: L_t/C_t at grid edge")
    np.savez("data/scores.npz", s=sh, y=y[ih])                      # held-out scores for eval/triage.py
    xs = [coral.health_x(L, C) for L, C in pick(S, ic)]
    edges = [float(v) for v in np.quantile(xs, [i / 6 for i in range(1, 6)])]
    txs = []
    for j in ic:
        S_j = V[k][j]; L_j, C_j = S_j
        C_2d = np.hypot(np.zeros_like(C_j), C_j).astype(np.float32).reshape(-1, 1)
        gx = cv2.Scharr(C_2d, cv2.CV_32F, 1, 0); gy = cv2.Scharr(C_2d, cv2.CV_32F, 0, 1)
        txs.append(float(np.hypot(gx, gy).mean()))
    texture_edges = [float(v) for v in np.quantile(txs, [(i + 1) / 16 for i in range(15)])]
    json.dump(dict(L_t=Lt, C_t=Ct, tau=tau, health_edges=edges, texture_edges=texture_edges, norm=k[1], seg=k[0]),
              open("data/params.json", "w")); print("saved data/params.json, data/scores.npz")