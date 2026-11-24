# ReefGlyph: Project Progress and Implementation Status

**Date:** November 24, 2026  
**Project:** Underwater Acoustic Coral Telemetry & Verification (ReefGlyph)  
**Spec Version:** v1.1 (incorporating 4-bit Texture Index in L0)

---

## 1. Project Overview & Architecture

ReefGlyph enables divers to photograph coral on-device, analyze health/bleaching indicators using lightweight classical computer vision, and transmit compact telemetry records to surface or buddy diver devices across underwater acoustic channels using Chirp Spread Spectrum (CSS) modulation.

```
[Coral Image (Rect)] 
       │
       ▼ (Classical CV: Lab thresholding, Scharr Texture, EFD, 16x16 Median-Cut)
 [CoralRecord] ───► L0 Telemetry (80 bits) | L1 Outline (175 bits) | L2 Thumbnail (576 bits)
       │
       ▼ (Framing: 16b Header + 108b Payload + 16b CRC-16 CCITT)
 [140b Frames] ──► LFSR Whitening ──► Hamming(7,4)/(8,4) ──► Block Interleaver ──► Gray Mapping
       │
       ▼ (CSS Modulation: fc = 2.5 kHz, BW = 3 kHz, SF7, fs = 48 kHz Audio)
 [Acoustic Audio] ──► Preamble (8 Up + 2 Down Chirps) + Guard
       │
       ▼ (Acoustic Channel: Multipath, SNR, Clock Drift, Delay)
 [Receiver Loop] ──► Continuous RingBuffer ──► Sliding Dechirp Sync ──► CFO Derot ──► Demod
       │
       ▼ (CRC-16 Pass)
 [Decoded Glyph UI & Telemetry Logs]
```

---

## 2. Key Milestones & Completion Summary

| Component / Milestone | Target Specification | Status | Evidence / Implementation Details |
|---|---|---|---|
| **Specification (v1.1)** | Frozen layout, L0 expanded to 80 bits, $f_c=2.5\text{ kHz}$, $\text{BW}=3\text{ kHz}$, $\text{SF}=7$ | **Completed** | `docs/spec.md`, `reefglyph/spec.py` |
| **Python Codec & Glyph** | Pack/unpack L0, L1, L2; pyefd contour reconstruction; SVG/PNG glyph renderer | **Completed** | `py/reefglyph/codec/layers.py`, `glyph.py` |
| **Texture Index Feature** | Scharr gradient magnitude on 2D chroma $\to 16$ quantile bins (0..15) packed into L0 spare bits | **Completed** | Added to `coral.py`, `layers.py`, `spec.py`, `calibrate.py`, `glyph.py` |
| **PHY & Modem (Python)** | CSS modulate/demodulate, sliding-dechirp sync, fractional timing & CFO estimation | **Completed** | Tested bit-exact noiseless loopback; $\pm 100\text{ ppm}$ clock drift recovery in `demo.py` |
| **Channel Simulator** | Multipath taps, broadband noise, frequency offset, IR convolution | **Completed** | `py/reefglyph/sim/channel.py` |
| **Classical CV Detector** | Downsampling, Lab color space thresholding, bleach index, health scoring, texture index | **Completed** | Calibration script `calibrate.py`, held-out split AUC 0.781 (95% CI 0.733–0.825) |
| **Evaluation Suite** | Airtime budget triage, link curves vs. SNR, compression baselines (JPEG, Haar, Palette) | **Completed** | `eval/triage.py`, `link_curves.py`, `baselines.py`, `error_review.py` |
| **Android Kotlin App** | Single-file architecture (`MainActivity.kt`) housing full PHY, Modem, AudioTrack TX, AudioRecord RX | **Completed** | `android/app/src/main/java/org/reefglyph/MainActivity.kt` |
| **Android Permissions & Build** | AndroidManifest and `build.gradle.kts` configuration with JTransforms FFT & OpenCV | **Completed** | Permissions placed at manifest root; Gradle syntax validated |

---

## 3. Detailed Implementation Breakdown

### 3.1 Python Core (`py/reefglyph/`)
1. **`spec.py`**:
   - Audio rate: $48000\text{ Hz}$, Bandwidth: $3000\text{ Hz}$ ($1\text{--}4\text{ kHz}$), Carrier: $2500\text{ Hz}$, $\text{SF}=7$.
   - Bit allocation: L0 updated from 76 to 80 bits; Frame total = 140 bits (16b Header + 108b Payload + 16b CRC-16).
   - Airtime: SHORT burst = $2.13\text{ s}$ (50 symbols), LONG burst = $14.08\text{ s}$ (330 symbols), ACK = $0.77\text{ s}$ (18 symbols).
2. **`codec/layers.py` & `glyph.py`**:
   - `CoralRecord` holds transect, coral ID, depth, time, bleach, health, morphology, dominant colors, and texture.
   - `pack_L0` and `unpack_L0` bit-level serialisation with banker's rounding (`Math.rint()` equivalence).
   - Glyph rendering upright in box frame with real-time green/red bleach badge and texture indicator.
3. **`analysis/coral.py` & `calibrate.py`**:
   - Texture index implementation using 2D Scharr filters on Lab chroma:
     $$\text{Texture} = \text{mean}\left(\sqrt{(\nabla_x C)^2 + (\nabla_y C)^2}\right)_{eroded\_mask}$$
   - Maps continuous texture gradient to 16 equal-frequency calibration bins (`texture_edges`).
4. **`modem/` (`fec.py`, `framing.py`, `phy.py`)**:
   - 8-bit LFSR de/whitening (seed `0xFF`, taps 7, 5, 4, 3).
   - Hamming(7,4) and Hamming(8,4) coding.
   - Block interleaving over $SF$ codewords $\times n$ symbols.
   - Preamble sliding dechirp: peak detection, fractional bin interpolation, down-chirp marker lock, CFO & timing alignment.

---

### 3.2 Android Kotlin Implementation (`android/`)
Consolidated into **`android/app/src/main/java/org/reefglyph/MainActivity.kt`**:
1. **Modem Engine**:
   - Pure Kotlin port of Spec constants, bit whitening, Hamming tables, and CRC-16/CRC-8.
   - Fast 128-point complex FFT using JTransforms (`DoubleFFT_1D`).
   - Time-domain chirp synthesizer with peak normalization to 0.8.
2. **Audio I/O & DSP Pipeline**:
   - **Audio Transmission**: Low-latency `AudioTrack` playing float PCM at $48\text{ kHz}$.
   - **Audio Reception**: Continuous `AudioRecord` streaming from `MediaRecorder.AudioSource.UNPROCESSED`.
   - **Zero-Allocation Ring Buffer**: 6-second primitive circular `FloatArray` eliminating Android GC stalls.
   - **Continuous Receiver Loop**: Non-blocking audio capture allowing in-flight symbol accumulation without thread-blocking.
   - **Baseband Conversion**: OSR=16 integrate-and-dump decimation matched to transmitter carrier phase.
   - **Sync & Preamble Recovery**: Exact sample-index tracking without modulo clipping, CFO derotation, and sub-sample alignment.
3. **User Interface (Jetpack Compose)**:
   - Live coral evidence glyph badge with dynamic health/bleach/texture readout.
   - Sliders for simulation of Bleach and Texture values.
   - Hardware volume button trigger (`KEYCODE_VOLUME_DOWN`) for underwater transmission.
   - Independently scrollable terminal console with justified output cards for sync, CFO, SNR, and CRC status.

---

## 4. Key Bug Fixes & Engineering Resolutions

1. **Cross-Language Rounding**:
   - Replaced standard truncation/half-up with round-to-even (`Math.rint()`) to ensure identical encoding between Python and Kotlin.
2. **SHORT Burst Timing In-Flight Fix**:
   - Eliminated thread sleep calls inside the receiver loop that starved `AudioRecord`, allowing continuous streaming while waiting for trailing burst symbols.
3. **Sync Window Offset Alignment**:
   - Resolved a critical bug where `Math.floorMod(..., 128)` wrapped the global buffer start index back to zero. Restored absolute sample indexing $a = \lfloor g_0 N - \text{peak\_bin} \rceil$, resolving the 100% frame CRC failure.
4. **Manifest and Gradle Configurations**:
   - Fixed misplaced `<uses-permission>` tags inside `<intent-filter>` in `AndroidManifest.xml`.
   - Added missing parenthesis in `build.gradle.kts`.

---

## 5. Next Steps

1. **Acoustic Air & Water Characterization**:
   - In-air loopback verification (phone-to-phone at $1\text{--}3\text{ m}$).
   - Pool transmission testing using waterproof phone pouches.
2. **Hardware Energy Measurement**:
   - Run calibration suites (`android/calib`) to profile battery draw across states (`E0a` idle to `E0g` full analyze).
3. **Final Report & Documentation**:
   - Consolidate real-world error contact sheets and literature comparisons into final documentation.
