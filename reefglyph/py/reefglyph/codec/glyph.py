import numpy as np, cv2, pyefd


def render_glyph(r, S=256):
    """r: CoralRecord with whatever layers were decoded. Returns BGR image (upright, in box frame)."""
    img = np.full((S, S, 3), 255, np.uint8)
    if r.efd is None:   # L0 only: fill with the two L0 colours
        for i, lab in enumerate(r.colors):
            lab8 = np.array([[[np.clip(lab[0] * 2.55, 0, 255), np.clip(lab[1] + 128, 0, 255),
                               np.clip(lab[2] + 128, 0, 255)]]], np.uint8)
            img[:, i * S // 2:(i + 1) * S // 2] = cv2.cvtColor(lab8, cv2.COLOR_LAB2BGR)[0, 0]
    else:
        cnt = pyefd.reconstruct_contour(np.asarray(r.efd), r.locus, 200)
        pts = np.clip(cnt * (S - 1), 0, S - 1).astype(np.int32)
        mask = np.zeros((S, S), np.uint8); cv2.fillPoly(mask, [pts], 255)
        if r.thumb is not None:
            pal_bgr = np.asarray(r.palette)[:, ::-1].astype(np.uint8)
            fill = cv2.resize(pal_bgr[np.asarray(r.thumb)], (S, S), interpolation=cv2.INTER_NEAREST)
        else:
            fill = np.full((S, S, 3), 128, np.uint8)
        box = np.where(mask[..., None] > 0, fill, 255).astype(np.uint8)
        cv2.polylines(box, [pts], True, (0, 0, 0), 1)
        # restore the oriented box's aspect ratio (box frame was warped to a square)
        a = float(np.clip(r.aspect, .25, 4))
        w, h = (S, max(1, int(S / a))) if a >= 1 else (max(1, int(S * a)), S)
        box = cv2.resize(box, (w, h), interpolation=cv2.INTER_AREA)
        y0, x0 = (S - h) // 2, (S - w) // 2
        img[y0:y0 + h, x0:x0 + w] = box
    col = (0, 0, 255) if r.bleach > 0.3 else (0, 160, 0)    # bleach badge
    cv2.rectangle(img, (0, 0), (S, 22), col, -1)
    cv2.putText(img, f"coral {r.coral} bleach {r.bleach:.2f} H{r.health} T{r.texture}", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    return img
