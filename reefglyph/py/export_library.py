"""Export a balanced crop library and packed Python glyph references."""

import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "reefglyph" / "data"
ASSETS_DIR = PROJECT_ROOT / "android" / "app" / "src" / "main" / "assets"
LIBRARY_DIR = ASSETS_DIR / "library"
GOLDEN_DIR = DATA_DIR / "golden_glyphs"
SAMPLES_PER_CLASS = 15
MAX_PREVIEW_WIDTH = 256
SEED = 2026

sys.path.insert(0, str(PROJECT_ROOT / "reefglyph" / "py"))

from reefglyph.analysis import coral
from reefglyph.codec.glyph import render_glyph
from reefglyph.codec.layers import (
    pack_L0,
    pack_L1,
    pack_L2,
    unpack_L0,
    unpack_L1,
    unpack_L2,
)


def bits_hex(bits):
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes().hex()


def analyze_crop(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read crop: {path}")
    height, width = image.shape[:2]
    rect = (width // 10, height // 10, width * 8 // 10, height * 8 // 10)
    record = coral.analyze(image, rect)
    if record.efd is None or record.thumb is None or record.palette is None:
        raise ValueError(f"Analyzer did not produce all glyph layers for {path}")
    layers = (pack_L0(record), pack_L1(record), pack_L2(record))
    return image, record, layers


def write_preview(image, destination):
    height, width = image.shape[:2]
    if width > MAX_PREVIEW_WIDTH:
        image = cv2.resize(
            image,
            (MAX_PREVIEW_WIDTH, max(1, round(height * MAX_PREVIEW_WIDTH / width))),
            interpolation=cv2.INTER_AREA,
        )
    if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise OSError(f"Cannot write preview: {destination}")


def main():
    coral.load_params(str(DATA_DIR / "params.json"))
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    entries = []
    records = []
    extensions = {".jpg", ".jpeg", ".png"}

    for label in ("healthy", "bleached"):
        candidates = sorted(
            path
            for path in (DATA_DIR / "crops" / label).iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )
        rng.shuffle(candidates)
        accepted = 0
        for path in candidates:
            try:
                image, record, layers = analyze_crop(path)
            except ValueError as exc:
                print(f"Skipping {path.name}: {exc}", file=sys.stderr)
                continue

            sample_id = f"{label}-{accepted + 1:02d}"
            preview_name = f"{sample_id}.jpg"
            write_preview(image, LIBRARY_DIR / preview_name)
            entries.append(
                {
                    "id": sample_id,
                    "filename": path.name,
                    "bleach_index": record.bleach,
                    "flagged": bool(record.evidence),
                    "health_score": record.health,
                    "texture_idx": record.texture,
                    "l0_hex": bits_hex(layers[0]),
                    "l1_hex": bits_hex(layers[1]),
                    "l2_hex": bits_hex(layers[2]),
                }
            )
            records.append((label, sample_id, record, layers))
            accepted += 1
            if accepted == SAMPLES_PER_CLASS:
                break

        if accepted != SAMPLES_PER_CLASS:
            raise RuntimeError(
                f"Expected {SAMPLES_PER_CLASS} usable {label} crops; found {accepted}"
            )

    (LIBRARY_DIR / "library.json").write_text(
        json.dumps(entries, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copyfile(DATA_DIR / "params.json", ASSETS_DIR / "params.json")

    healthy = min((item for item in records if item[0] == "healthy"), key=lambda item: item[2].bleach)
    bleached = max((item for item in records if item[0] == "bleached"), key=lambda item: item[2].bleach)
    ordered = sorted(records, key=lambda item: item[2].bleach)
    middle = ordered[len(ordered) // 2]
    golden_records = (("healthy", healthy), ("bleached", bleached), ("representative", middle))
    for filename, (_, _, _, layers) in golden_records:
        decoded = unpack_L2(layers[2], unpack_L1(layers[1], unpack_L0(layers[0])))
        image = render_glyph(decoded)
        if not cv2.imwrite(str(GOLDEN_DIR / f"{filename}.png"), image):
            raise OSError(f"Cannot write golden glyph: {GOLDEN_DIR / filename}.png")

    print(f"Exported {len(entries)} crops (15 healthy, 15 bleached).")


if __name__ == "__main__":
    main()
