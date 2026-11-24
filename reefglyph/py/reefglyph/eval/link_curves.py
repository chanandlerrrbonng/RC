"""A2: delivery vs SNR in the simulator.
Run from project root:  uv run python -m reefglyph.eval.link_curves   -> data/link_curves.json
Each entry: snr_db -> [P(SHORT delivered), P(LONG fully delivered), mean snr_est from the modem]."""
import json, numpy as np
from ..modem import phy
from ..modem.framing import packetize
from ..sim.channel import channel

T = 20                        # trials per SNR point
SNRS = range(-14, 8, 2)       # broadband SNR in dB (in-band SNR is ~9 dB higher)
rng = np.random.default_rng(0)
mk = lambda n: rng.integers(0, 2, n).astype(np.uint8)


def trial(layers, seed):
    rx = channel(phy.modulate(packetize(layers, 1)), snr_db=cur, ppm=int(rng.integers(-100, 101)),
                 delay_s=float(rng.uniform(0, .5)), taps=[(0.004, .4), (0.011, .2)], seed=seed)
    return phy.demodulate(rx)


if __name__ == "__main__":
    res = {}
    for cur in SNRS:
        ok0 = ok2 = 0; est = []
        for t in range(T):
            L = {0: mk(76), 1: mk(175), 2: mk(576)}
            fr, info = trial({0: L[0]}, t)
            if info: est.append(info["snr_db"])
            ok0 += bool(fr and fr[0].ok)
            fr, info = trial(L, 1000 + t)
            ok2 += len(fr) == 9 and all(f.ok for f in fr)
        res[cur] = [ok0 / T, ok2 / T, float(np.mean(est)) if est else None]
        print(f"SNR {cur:+3d} dB  SHORT {ok0}/{T}  LONG {ok2}/{T}  snr_est {res[cur][2]}")
    json.dump(res, open("data/link_curves.json", "w"))
    print("saved data/link_curves.json")
