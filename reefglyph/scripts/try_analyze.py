"""uv run python scripts/try_analyze.py [image.jpg]   -- times analyze() and writes try_glyph.png"""
import sys, time, numpy as np, cv2
from reefglyph.analysis.coral import analyze
from reefglyph.codec.layers import *
from reefglyph.codec.glyph import render_glyph
if len(sys.argv) > 1:
    im = cv2.imread(sys.argv[1])
else:   # synthetic pale blob on a dark-blue background
    im = np.full((480, 640, 3), (120, 60, 20), np.uint8)
    cv2.ellipse(im, (320, 240), (180, 110), 20, 0, 360, (210, 215, 220), -1)
    im = cv2.add(im, np.random.default_rng(0).integers(0, 15, im.shape, dtype=np.uint8))
h, w = im.shape[:2]
t = time.time(); r = analyze(im, (w // 10, h // 10, w * 8 // 10, h * 8 // 10)); dt = time.time() - t
r.coral = 1
print(f"analyze: {dt:.2f} s (target < 1 s)  bleach={r.bleach:.2f} health={r.health} evidence={r.evidence}")
L = {0: pack_L0(r), 1: pack_L1(r), 2: pack_L2(r)}
out = unpack_L2(L[2], unpack_L1(L[1], unpack_L0(L[0])))
cv2.imwrite("try_glyph.png", render_glyph(out)); print("wrote try_glyph.png")
