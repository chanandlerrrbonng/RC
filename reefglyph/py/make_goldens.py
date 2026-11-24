"""Generate Kotlin modem golden WAV fixtures from the Python reference."""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from reefglyph.codec.layers import CoralRecord, pack_L0, pack_L1, pack_L2
from reefglyph.modem.framing import Frame, pack_ack, pack_frame, packetize, unpack_ack
from reefglyph.modem.phy import demodulate, modulate
from reefglyph.sim.channel import channel, write_wav


OUTPUT_DIR = PROJECT_ROOT / "android" / "app" / "src" / "test" / "resources" / "golden"
BURST_ID = 23


def coral_record():
    record = CoralRecord(
        transect=17,
        coral=42,
        depth_m=12.75,
        time_s=1432.0,
        bleach=0.27,
        health=4,
        morph=1,
        colors=[(62.0, 18.0, -12.0), (38.0, -8.0, 22.0)],
        patch=1,
        evidence=1,
        texture=11,
    )
    record.efd = np.linspace(-0.45, 0.45, 32, dtype=np.float64).reshape(8, 4)
    record.locus = (0.37, 0.62)
    record.aspect = 1.4
    yy, xx = np.indices((16, 16))
    record.thumb = ((xx + 2 * yy) % 4).astype(np.uint8)
    record.palette = np.array(
        [[24, 92, 44], [192, 144, 72], [32, 164, 188], [216, 204, 176]],
        dtype=np.uint8,
    )
    return record


def bits_hex(bits):
    """Pack an MSB-first bit array, zero-padding its final byte."""
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes().hex()


def generate_case(name, kind, frames, audio):
    if kind == "ack":
        decoded_bits, info = demodulate(audio, kind="ack")
        if info is None:
            raise RuntimeError(f"{name}: ACK preamble was not detected")
        if not unpack_ack(decoded_bits)["ok"]:
            raise RuntimeError(f"{name}: decoded ACK CRC failure")
        decoded_hex = [bits_hex(decoded_bits)]
    else:
        decoded_frames, info = demodulate(audio, max_frames=len(frames))
        if info is None or len(decoded_frames) != len(frames):
            raise RuntimeError(f"{name}: decoded {len(decoded_frames)} of {len(frames)} frames")
        if not all(frame.ok for frame in decoded_frames):
            bad = [index for index, frame in enumerate(decoded_frames) if not frame.ok]
            raise RuntimeError(f"{name}: decoded frame CRC failure(s) at {bad}")
        decoded_hex = [
            bits_hex(
                pack_frame(
                    Frame(
                        frame.layer,
                        frame.idx,
                        frame.n,
                        frame.burst,
                        frame.payload,
                    )
                )
            )
            for frame in decoded_frames
        ]
        if decoded_hex != [bits_hex(frame) for frame in frames]:
            raise RuntimeError(f"{name}: decoded frame bytes differ from transmitted frames")

    filename = f"{name}.wav"
    write_wav(OUTPUT_DIR / filename, audio)
    return {
        "filename": filename,
        "kind": kind,
        "expected_decoded_frames_hex": decoded_hex,
        "expected_snr4": int(info["snr4"]),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    record = coral_record()
    layers = {0: pack_L0(record), 1: pack_L1(record), 2: pack_L2(record)}
    short_frames = packetize({0: layers[0]}, BURST_ID)
    long_frames = packetize(layers, BURST_ID)

    cases = {}
    cases["short_clean"] = generate_case(
        "short_clean", "short", short_frames, modulate(short_frames)
    )
    cases["long_clean"] = generate_case(
        "long_clean", "long", long_frames, modulate(long_frames)
    )

    channels = (
        ("long_noise", {"snr_db": 8, "seed": 101}),
        ("long_delay", {"delay_s": 0.37, "seed": 102}),
        ("long_multipath", {"taps": [(0.003, 0.35)], "seed": 103}),
        ("long_cfo_plus100ppm", {"ppm": 100, "seed": 104}),
        ("long_cfo_minus100ppm", {"ppm": -100, "seed": 105}),
        ("long_clipped", {"clip": 0.2, "seed": 106}),
    )
    for name, options in channels:
        cases[name] = generate_case(
            name, "long", long_frames, channel(modulate(long_frames), **options)
        )

    ack_bits = pack_ack(BURST_ID, [1, 1, 1, 1, 1, 1, 1, 1, 1], 9)
    cases["ack_clean"] = generate_case(
        "ack_clean", "ack", [], modulate([ack_bits], kind="ack")
    )

    manifest = {
        "cases": cases,
        "sample_layers": {f"L{layer}": bits_hex(bits) for layer, bits in layers.items()},
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
