import numpy as np, cv2, pyefd
from reefglyph.codec.layers import *
from reefglyph.codec.glyph import render_glyph
from reefglyph.modem import phy
from reefglyph.modem.framing import *
from reefglyph.sim.channel import channel
from reefglyph.spec import airtime

print("airtime SHORT/LONG/ACK:", [round(airtime(k), 2) for k in ("short", "long", "ack")], "(expect 2.13 14.08 0.77)")
t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
cont = np.c_[0.5 + 0.35 * np.cos(t) * (1 + .2 * np.cos(3 * t)), 0.5 + 0.3 * np.sin(t)]
rec = CoralRecord(transect=5, coral=42, depth_m=7.5, time_s=300, bleach=.62, health=4, morph=0,
                  colors=[(80, 5, 10), (60, -4, 12)], evidence=1,
                  efd=np.clip(pyefd.elliptic_fourier_descriptors(cont, order=8, normalize=False), -.5, .5),
                  locus=tuple(pyefd.calculate_dc_coefficients(cont)), aspect=1.3,
                  thumb=np.random.default_rng(1).integers(0, 4, (16, 16)),
                  palette=np.array([[230, 225, 210], [200, 180, 150], [120, 90, 70], [60, 40, 30]], np.uint8))
layers = {0: pack_L0(rec), 1: pack_L1(rec), 2: pack_L2(rec)}
assert unpack_L0(layers[0]).coral == 42
frames = packetize(layers, burst_id=9)
audio = phy.modulate(frames); print("audio s:", len(audio) / 48000)

# 0) sync sign check: known delay + known CFO at baseband
sf, N = 7, 128
bb0 = phy.to_baseband(audio.astype(np.float64))
for d, c in ((1000, 3.3), (1037, -5.6), (1000, 0.0)):
    bb = np.concatenate([np.zeros(d, complex), bb0]) * np.exp(2j * np.pi * c * np.arange(len(bb0) + d) / N)
    s = phy.sync(bb, sf)
    print(f"sync d={d} cfo={c}: start err {s['start'] - (d + 10 * N)}, cfo est {s['cfo']:.2f}")
    assert abs(s['start'] - (d + 10 * N)) <= 2 and abs(s['cfo'] - c) < 0.6, "sync signs/estimates wrong"

# 1) noiseless loopback must be exact
fr, info = phy.demodulate(audio); assert all(f.ok for f in fr) and len(fr) == 9, "noiseless loopback failed"
assert (np.concatenate([f.payload for f in fr[:1]])[:76] == layers[0]).all()
print("noiseless OK, sync:", info)

# 2) impaired: noise + delay + multipath + +-100 ppm
for ppm in (-100, 0, 100):
    rx = channel(audio, snr_db=8, ppm=ppm, delay_s=0.37, taps=[(0.004, .4), (0.011, .2)], seed=3)
    fr, info = phy.demodulate(rx)
    print(f"ppm {ppm:+4d}: frames ok {sum(f.ok for f in fr)}/9, snr_est {info['snr_db'] if info else None}")

# 3) ARQ: drop frames 3..5 on attempt 1, resend only the missing ones
R = Reassembler(); fr, _ = phy.demodulate(audio); R.add([f for f in fr if f.idx not in (3, 4, 5)])
print("missing:", R.missing())
ack = pack_ack(9, R.bitmap(), 7)
bits, _ = phy.demodulate(channel(phy.modulate([ack], kind="ack"), 10), kind="ack")
print("ACK:", unpack_ack(bits))
fr2, _ = phy.demodulate(phy.modulate(packetize(layers, 9, only=set(R.missing()))))
R.add(fr2); L = R.layers(); print("layers complete:", sorted(L)); assert sorted(L) == [0, 1, 2]
out = unpack_L2(L[2], unpack_L1(L[1], unpack_L0(L[0])))
cv2.imwrite("glyph.png", render_glyph(out)); print("wrote glyph.png")
