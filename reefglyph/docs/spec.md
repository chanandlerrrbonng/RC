# ReefGlyph: interface specification (v1, frozen)

Status: **frozen**. Python (`reefglyph/spec.py`) and Kotlin must both follow this file. Any change requires a new version tag and a matching change in both implementations.

All multi-bit fields are **MSB first**. All bit arrays are 0/1 values.

## 1. Scope

A diver's phone analyses a coral image on-device (classical CV, no ML) and sends the result to a buddy or surface phone over underwater acoustics in the 1-4 kHz band.

| Burst | Contents | Purpose |
|---|---|---|
| SHORT | L0 (76 bits, one frame) | Alert, sent for every coral |
| LONG | L0 + L1 + L2 (827 bits, nine frames) | Evidence for a human, sent for flagged corals |
| ACK | burst ID, 9-bit frame bitmap, SNR | Selective repeat |

Out of scope: L3+ refinement layers, ML codecs, depth colour correction, security, streaming video.

## 2. Physical layer

| Item | Value |
|---|---|
| Audio sample rate | 48 kHz (native rate of each phone to be verified) |
| Band / carrier | 1-4 kHz, carrier fc = 2.5 kHz, BW = 3 kHz |
| Baseband | complex, fs = BW = 3 kHz, oversampling OSR = 16 |
| Spreading factor | SF7 default (N = 128 samples/symbol, 42.67 ms); SF9 optional |
| Modulation | chirp spread spectrum; symbol k = base up-chirp cyclically shifted by k bins |
| Base chirp | `exp(j(pi n^2 / N - pi n))`, n = 0..N-1; down-chirp is its conjugate |
| Preamble | 8 up-chirps (k = 0) + 2 down-chirps, once per burst (10 symbols, 0.43 s) |
| Trailing guard | 2N zero samples appended after the last symbol (not counted in airtime) |
| Output level | peak-normalised to 0.8 |

### 2.1 Bit chain (per frame)

1. **Whitening**: 8-bit LFSR, seed `0xFF`, output bit = LSB, feedback = bits 7 ^ 5 ^ 4 ^ 3, shift left; XOR with data. Same operation de-whitens.
2. **FEC**: split into nibbles (MSB first). Data bits d0..d3, parity p1 = d0^d1^d3, p2 = d0^d2^d3, p3 = d1^d2^d3. Codeword = `d0 d1 d2 d3 p1 p2 p3`. Hamming(8,4) adds an overall parity bit (XOR of the 7 bits). Decoding is nearest-codeword (minimum Hamming distance).
3. **Interleaver**: group codewords in blocks of SF. A block of SF codewords of n bits becomes n symbols of SF bits; symbol j takes bit j of each codeword in the block. A final partial block is zero-padded.
4. **Gray mapping**: transmit `gray_dec(value)`; receiver applies `gray_enc(bin)`.

Interleaving limits one wrong symbol to one bit error per codeword.

### 2.2 Synchronisation and tracking

- **Detection**: sliding dechirp over windows of N samples. Lock requires 4 consecutive up-chirp peaks with peak/median ratio above 7 and peak bins within 2.5 bins of each other.
- **Down-chirp**: the first window whose down-chirp dechirp beats the up-chirp one marks the preamble tail.
- **Estimates** (verified by simulation): with b_u and b_d the up- and down-chirp peak bins, `CFO = (b_u + b_d)/2` bins and `timing error = (b_u - b_d)/2` samples. Data start = preamble start + 10N, corrected by the timing error. The signal is derotated by the CFO before demodulation.
- **Clock drift**: after every symbol, take the fractional part of the dechirp peak, fit a line to the last 48 values, and use it to place the next window (independent of decoded symbols). Verified at +/-100 ppm. No pilot chirp.
- **SNR estimate**: `20 log10(mean peak/median ratio over the last up-chirps)`. This is **uncalibrated**. The ACK field is `snr4 = clip(round(snr_db / 2), 0, 15)` until a calibration against the simulator replaces it.

## 3. Frame format (140 bits)

| Field | Bits |
|---|---|
| Layer | 2 |
| Frame index | 4 |
| Frames in burst | 4 |
| Burst ID | 6 |
| Payload (zero-padded) | 108 |
| CRC-16 (CCITT: poly 0x1021, init 0xFFFF) over the 124 header+payload bits | 16 |

Layer values: 0 = L0, 1 = L1, 2 = L2.

### 3.1 Coding and airtime

- 35 nibbles -> Hamming(7,4) -> 5 blocks of 7 codewords -> 7 symbols per block -> **35 symbols (1.49 s)**.
- **Frame 0 of every burst** (always L0) uses Hamming(8,4) **by position**: 40 symbols (1.71 s). No header is needed to pick the FEC.
- Frame length is fixed, so the receiver always knows where a frame ends.

| Burst | Frames | Symbols | Airtime (SF7) |
|---|---|---|---|
| SHORT | F0 | 10 + 40 = 50 | 2.13 s |
| LONG | F0, F1-F2 (L1), F3-F8 (L2) | 10 + 40 + 8 x 35 = 330 | 14.08 s |
| ACK | one block | 10 + 8 = 18 | 0.77 s |

Airtime = symbols x 2^SF / 3000.

### 3.2 Layer-to-frame mapping

A layer's bits are cut into 108-bit chunks, one per frame, the last zero-padded. The receiver truncates to the layer length.

| Layer | Bits | Frames | Frame indices (full LONG burst) |
|---|---|---|---|
| L0 | 76 | 1 | 0 |
| L1 | 175 | 2 | 1-2 |
| L2 | 576 | 6 | 3-8 |

The L0 *evidence follows* flag tells the receiver whether to expect F1-F8. A layer is usable only when all its frames passed CRC.

## 4. Payload layouts

### 4.1 L0 telemetry (76 bits of 108)

Fields in order:

| Field | Bits | Encoding |
|---|---|---|
| transect ID | 6 | integer |
| coral ID | 8 | integer |
| depth | 7 | 0-63.5 m, 0.5 m steps |
| time offset | 10 | 0-5115 s, 5 s steps |
| bleach index | 6 | 0-1, step 1/63 |
| health score | 3 | stores score-1 (score 1-6) |
| morphology tag | 2 | 0 branching, 1 massive, 2 plate, 3 other |
| dominant colour 1 | 16 | Lab: L 6 bits (0-100), a 5 bits, b 5 bits |
| dominant colour 2 | 16 | same |
| reference-patch-found flag | 1 | |
| evidence-follows flag | 1 | |

Lab a and b are stored as `round(v/4) + 16`, clamped to 0-31 (step 4, range -64..+60). The remaining 32 payload bits are reserved.

### 4.2 L1 outline (175 bits of 216)

Contour in the oriented-box frame (unit square), unnormalised elliptic Fourier descriptors (`pyefd`, `normalize=False`).

- 8 harmonics x 4 coefficients = 32 values, 5 bits each (range -0.5..+0.5), row-major (160 bits)
- locus (x, y): 2 x 5 bits, range 0..1 (10 bits)
- aspect ratio: 5 bits, `log2(aspect)` over -2..+2 (aspect clamped to 0.25-4)

### 4.3 L2 thumbnail (576 bits of 648)

- 16 x 16 pixels in the same box frame, raster order, 2-bit index into a 4-colour palette (512 bits)
- 4 palette colours, RGB565, 16 bits each (64 bits)
- No background index: the receiver clips with the L1 outline
- No rotation is sent, so the glyph is shown upright in the box frame

### 4.4 ACK frame (28 bits)

| Field | Bits |
|---|---|
| Burst ID | 6 |
| Frame bitmap (bit i = frame i received) | 9 |
| SNR estimate | 4 |
| CRC-8 (poly 0x07, init 0) over the 19 body bits zero-padded to 24 | 8 |
| Pad | 1 |

Coded as Hamming(8,4): 7 codewords, one interleaver block, 8 symbols.

## 5. Reliability (ARQ)

1. After a burst, the sender opens an ACK window that starts 0.3 s after the last symbol and lasts 3 s. It listens only then.
2. The receiver replies with an ACK bitmap.
3. The sender resends only the missing frames as a new burst with the **same burst ID**. Maximum 3 attempts.
4. The receiver keeps partial layers across attempts.

Known quirk: the first frame of a resend burst is coded with Hamming(8,4) by position, and its header still states 9 frames, so the receiver stops when the audio ends.

## 6. Analysis (M2)

`analyze(image, rect) -> CoralRecord`, target under 1 s per image.

1. Downscale to a longest side of 320 px.
2. Mask: user rectangle, or GrabCut from it (selected by calibration).
3. Optional gray-world colour normalisation (selected by calibration).
4. Erode the mask by about 3 px, take Lab pixels, drop the brightest 1 % of L.
5. **Bleach index** = fraction of pixels with L > L_t and chroma < C_t. The coral is flagged when the index exceeds tau.
6. **Health score** 1-6 from six equal-count bins of `mean(L) - mean(chroma)`; the five bin edges come from the calibration split.
7. Two dominant colours by median cut, converted to Lab.
8. Oriented box via `cv2.minAreaRect`; image and mask are warped into a 64 x 64 box frame; thumbnail 16 x 16 by area resampling; palette by median cut over mask pixels.
9. Morphology is fixed to "other" (no labels to beat the majority-class baseline).

Tuned parameters (`data/params.json`: L_t, C_t, tau, health_edges, norm, seg) are **not** part of the frozen interface. They are re-tuned on a calibration split and reported on a held-out split.

## 7. Glyph rendering

`render_glyph(CoralRecord) -> BGR image`, upright in the box frame.

- L0 only: fill with the two L0 colours.
- With L1: draw the outline from the EFD, fill with the thumbnail clipped by the outline, restore the aspect ratio.
- Header badge: red if bleach > 0.3, otherwise green, with coral ID, bleach index and health score.

## 8. Required functions

| Function | Notes |
|---|---|
| `analyze(image, rect)` | returns `CoralRecord` |
| `pack_L0/L1/L2`, `unpack_L0/L1/L2` | layouts in section 4 |
| `render_glyph(record)` | section 7 |
| `packetize(layers, burst_id, only=None)` / `Reassembler` | frames, selective repeat |
| `pack_ack`, `unpack_ack` | section 4.4 |
| `modulate(frames, sf, kind)` -> audio | float32 at 48 kHz |
| `demodulate(audio, sf, kind)` -> frames, info | info holds start, cfo, snr_db |
| `airtime(burst, sf)` -> seconds | section 3.1 |

## 9. Conformance

- Noiseless loopback is bit-exact.
- Golden WAVs (noise, +/-100 ppm, delay, multipath, clipping) with expected bytes live in `data/golden/`. The Kotlin decoder must reproduce the same bytes.
- Kotlin decode target: under 200 ms per frame.

## 10. Differences from the original README

- Thumbnail stays 16 x 16 (an 18 x 18 variant was considered and not adopted).
- No pilot chirp; a drift tracker replaces it.
- Calibrated detector uses a threshold rule on L and chroma; coverage is dropped; morphology is fixed to "other".
- SNR to `snr4` mapping is provisional pending calibration.

## 11. Limitations

The bleach index is a colour proxy, not a diagnosis. Outlines lose fine branches. Capture is simulated from stored images. Simulator results are not pool results, and a pool is not a reef. Energy results depend on the phone model.