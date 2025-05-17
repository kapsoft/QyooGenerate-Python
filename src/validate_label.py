#!/usr/bin/env python3
"""
validate_label.py  <index> [--split train] [--root dataset]

Draw the YOLO-seg polygon (green) and bounding box (red) on a
generated sample and save it to  <root>/<split>/test/.

Example:
    python validate_label.py 42
"""

import cv2, argparse, os
import numpy as np
from pathlib import Path

def draw_vis(img_path, lbl_path, out_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)

    h, w = img.shape[:2]

    with open(lbl_path) as f:
        line = f.readline().strip().split()
    if len(line) < 6:
        raise ValueError("Label file has no polygon points")

    # parse bbox
    _, cx, cy, bw, bh, *poly = map(float, line)
    x0 = int((cx - bw / 2) * w)
    y0 = int((cy - bh / 2) * h)
    x1 = int((cx + bw / 2) * w)
    y1 = int((cy + bh / 2) * h)

    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)   # red bbox

    # parse polygon (x1 y1 x2 y2 …)
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    pts = pts.astype(np.int32)

    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print("✔︎ wrote", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=int, help="sample index, e.g. 1234")
    ap.add_argument("--split", default="train", help="train / val / ...")
    ap.add_argument("--root",  default="dataset", help="dataset root folder")
    args = ap.parse_args()

    img_name = f"{args.index:06d}.jpg"
    lbl_name = f"{args.index:06d}.txt"

    split_path = Path(args.root) / args.split
    img_path   = split_path / "images" / img_name
    lbl_path   = split_path / "labels" / lbl_name
    out_path   = split_path / "test"   / f"{args.index:06d}_vis.jpg"

    draw_vis(img_path, lbl_path, out_path)

