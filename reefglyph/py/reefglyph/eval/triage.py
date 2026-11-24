"""A4: triage policies under airtime budgets, bootstrap CIs.
Run from project root:  uv run python -m reefglyph.eval.triage [snr_db]
Needs data/scores.npz (from calibrate.py) and data/params.json; uses data/link_curves.json if present."""
import json, os, sys, numpy as np
from ..spec import airtime

SH, LG = airtime("short"), airtime("long")
B, N, PREVS, BUDGETS = 500, 100, (0.1, 0.3), (300, 600, 900, 1500)   # bootstrap reps, corals per dive, airtime budgets (s)
POLICIES = ("all_short", "all_long", "triage", "scheduler")


def link(snr):
    if snr is not None and os.path.exists("data/link_curves.json"):
        d = {float(k): v for k, v in json.load(open("data/link_curves.json")).items()}
        k = min(d, key=lambda x: abs(x - snr)); return d[k][0], d[k][1]
    return 0.95, 0.85                      # placeholder until link_curves has been run


def run(pol, s, y, tau, budget, p0, p2, rng):
    """Returns (evidence coverage, wasted-airtime fraction, alert coverage)."""
    t = 0.0; n = len(y); g0 = np.zeros(n, bool); g2 = np.zeros(n, bool); waste = 0.0
    flag = s > tau

    def short(i):
        nonlocal t; t += SH; g0[i] = rng.random() < p0

    def long_(i):
        nonlocal t, waste; t += LG; g0[i] = rng.random() < p0; g2[i] = rng.random() < p2
        if y[i] == 0: waste += LG

    if pol == "all_short":
        for i in range(n):
            if t + SH > budget: break
            short(i)
    elif pol == "all_long":
        for i in range(n):
            if t + LG > budget: break
            long_(i)
    elif pol == "triage":                  # capture order: LONG if flagged, else SHORT
        for i in range(n):
            c = LG if flag[i] else SH
            if t + c > budget: break
            long_(i) if flag[i] else short(i)
    elif pol == "scheduler":               # SHORT for all first, then LONG for flagged by descending score
        for i in range(n):
            if t + SH > budget: break
            short(i)
        for i in np.argsort(-s):
            if flag[i] and t + LG <= budget: long_(i)
    pos = y == 1
    cov = g2[pos].mean() if pos.any() else np.nan
    return cov, waste / max(t, 1e-9), (g0 & pos).sum() / max(pos.sum(), 1)


if __name__ == "__main__":
    d = np.load("data/scores.npz"); S, Y = d["s"], d["y"]; tau = json.load(open("data/params.json"))["tau"]
    p0, p2 = link(float(sys.argv[1]) if len(sys.argv) > 1 else None); rng = np.random.default_rng(0)
    P, Ng = np.where(Y == 1)[0], np.where(Y == 0)[0]
    print(f"held-out n={len(Y)} ({len(P)} pos)  p0={p0} p2={p2}  SHORT {SH:.2f}s LONG {LG:.2f}s  (mean [95% CI])")
    for prev in PREVS:
        for bud in BUDGETS:
            for pol in POLICIES:
                r = []
                for _ in range(B):
                    npos = int(round(N * prev))
                    idx = rng.permutation(np.r_[rng.choice(P, npos), rng.choice(Ng, N - npos)])
                    r.append(run(pol, S[idx], Y[idx], tau, bud, p0, p2, rng))
                r = np.array(r)
                f = lambda c: f"{np.nanmean(r[:, c]):.2f} [{np.nanpercentile(r[:, c], 2.5):.2f},{np.nanpercentile(r[:, c], 97.5):.2f}]"
                print(f"prev {prev:.1f} budget {bud:4d}s {pol:9s} evidence_cov {f(0)}  wasted {f(1)}  alert_cov {f(2)}")
