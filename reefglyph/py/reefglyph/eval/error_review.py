"""Run from project root after calibrate.py:  uv run python -m reefglyph.eval.error_review
Prints bootstrap CIs for the held-out detector, checks near-duplicates between the calibration and held-out
halves, and writes contact sheets of the worst errors: data/errors_fp.jpg (healthy flagged as bleached)
and data/errors_fn.jpg (bleached missed)."""
import glob, json, os, numpy as np, cv2
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support as prf
from sklearn.model_selection import train_test_split


def ahash(im):
    g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (8, 8), interpolation=cv2.INTER_AREA)
    return (g > g.mean()).ravel()


def sheet(files, scores, idx, path, cols=5, T=200):
    tiles = []
    for i in idx:
        im = cv2.resize(cv2.imread(files[i]), (T, T))
        cv2.putText(im, f"{scores[i]:.2f} {os.path.basename(files[i])[:16]}", (3, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    while len(tiles) % cols: tiles.append(np.zeros((T, T, 3), np.uint8))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack(rows))


if __name__ == "__main__":
    files, y, H, size = [], [], [], []
    for lab, d in ((0, "healthy"), (1, "bleached")):          # same order and filtering as calibrate.load()
        for f in sorted(glob.glob(f"data/crops/{d}/*")):
            im = cv2.imread(f)
            if im is None: continue
            files.append(f); y.append(lab); H.append(ahash(im)); size.append(im.shape[:2])
    y, H, size = np.array(y), np.array(H), np.array(size)
    ic, ih = train_test_split(np.arange(len(y)), test_size=.5, stratify=y, random_state=0)
    d = np.load("data/scores.npz"); s, yh = d["s"], d["y"]
    assert (yh == y[ih]).all(), "scores.npz does not match this data/crops set: rerun calibrate"
    tau = json.load(open("data/params.json"))["tau"]; pred = s > tau

    # 1) bootstrap CIs on the held-out half
    rng = np.random.default_rng(0); A, P, R = [], [], []
    for _ in range(2000):
        b = rng.integers(0, len(yh), len(yh))
        if yh[b].min() == yh[b].max(): continue
        A.append(roc_auc_score(yh[b], s[b]))
        p, r, _, _ = prf(yh[b], pred[b], average="binary", zero_division=0); P.append(p); R.append(r)
    ci = lambda v: f"{np.mean(v):.3f} [{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]"
    print(f"held-out n={len(yh)}  AUC {ci(A)}  precision {ci(P)}  recall {ci(R)}  (95% bootstrap CI, 2000 reps)")

    # 2) near-duplicates between halves (average hash, Hamming distance <= 5 of 64)
    D = (H[ih][:, None, :] != H[ic][None, :, :]).sum(-1); nn = D.argmin(1); dmin = D.min(1)
    dup = dmin <= 5
    print(f"held-out images with a near-duplicate in the calibration half: {dup.sum()}/{len(ih)}; "
          f"of those, label differs from the duplicate: {(y[ih][dup] != y[ic][nn[dup]]).sum()}")

    # 3) image sizes per class (tight crops vs whole scenes)
    for lab, nm in ((0, "healthy"), (1, "bleached")):
        sz = size[y == lab]; print(f"{nm}: median size HxW {int(np.median(sz[:, 0]))}x{int(np.median(sz[:, 1]))}, n={len(sz)}")

    # 4) contact sheets of the worst errors (20 each)
    fp = np.where((yh == 0) & pred)[0]; fp = fp[np.argsort(-s[fp])][:20]
    fn = np.where((yh == 1) & ~pred)[0]; fn = fn[np.argsort(s[fn])][:20]
    hf = [files[i] for i in ih]
    sheet(hf, s, fp, "data/errors_fp.jpg"); sheet(hf, s, fn, "data/errors_fn.jpg")
    print(f"false positives {int(((yh == 0) & pred).sum())}, false negatives {int(((yh == 1) & ~pred).sum())}")
    print("wrote data/errors_fp.jpg and data/errors_fn.jpg: open them and note what you see")
