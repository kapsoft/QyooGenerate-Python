#!/usr/bin/env python3
"""
validate_all.py
Visualise *every* sample in a YOLO‑Seg split.

It walks   <root>/<split>/labels/*.txt
and writes <root>/<split>/vis/<same_name>_vis.jpg

Usage
-----
    python validate_all.py                 # defaults: --root dataset --split train
    python validate_all.py --root dataset_v2 --split train
    python validate_all.py --split val
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ───────────────────────── draw helper ────────────────────────────
def draw_vis(img_path: Path, lbl_path: Path, out_path: Path):
    """Draw bbox (red) + polygon (green) and save to out_path."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)

    h, w = img.shape[:2]

    with open(lbl_path) as f:
        line = f.readline().strip().split()
    if len(line) < 6:
        raise ValueError(f"Label file {lbl_path} has no polygon points")

    # YOLO‑Seg format: cls cx cy bw bh x1 y1 x2 y2 …
    _, cx, cy, bw, bh, *poly = map(float, line)

    # bbox (denorm)
    x0 = int((cx - bw / 2) * w)
    y0 = int((cy - bh / 2) * h)
    x1 = int((cx + bw / 2) * w)
    y1 = int((cy + bh / 2) * h)
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)

    # polygon (denorm)
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    pts = pts.astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


# ─────────────────────────── main ─────────────────────────────────
def main(root: Path, split: str):
    lbl_dir = root / split / "labels"
    img_dir = root / split / "images"
    out_dir = root / split / "vis"

    lbl_files = sorted(lbl_dir.glob("*.txt"))
    if not lbl_files:
        raise FileNotFoundError(f"No label files found in {lbl_dir}")

    for lbl_path in tqdm(lbl_files, desc=f"visualising {split}"):
        stem = lbl_path.stem  # e.g. "000123"
        img_path = img_dir / f"{stem}.jpg"
        out_path = out_dir / f"{stem}_vis.jpg"
        try:
            draw_vis(img_path, lbl_path, out_path)
        except Exception as e:
            print(f"⚠️  {stem}: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset", help="dataset root folder")
    ap.add_argument("--split", default="train", help="train / val / test …")
    args = ap.parse_args()

    main(Path(args.root), args.split)

