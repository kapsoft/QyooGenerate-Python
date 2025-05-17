
"""generate_qyoo_synthetic.py
Generate a synthetic dataset of Qyoo symbols in YOLOv8 format.

* Creates `dataset/train/images` and `dataset/train/labels`
  (val split optional).
* Each sample is built by:
    1. rendering a flat Qyoo teardrop shape with a random 6×6 dot pattern
    2. applying a random perspective / scale / rotation
    3. compositing over a random background (solid or image)
    4. writing YOLO label (class 0) with bbox of the symbol

Requires: Pillow, numpy, opencv-python
Usage:
    python generate_qyoo_synthetic.py --count 50000 --bg-dir ./backgrounds
"""

import argparse, os, random, math, glob
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

# ------------------ Qyoo rendering helpers ------------------------ #
DOT_GRID = 6
DOT_R_PX = 12         # radius of each dot in the template
SHAPE_SIZE = 256      # template square canvas

# ── YOLO-seg needs a polygon with ≤ 32 points ──────────────────────────────
def normalize_polygon(pts, img_w, img_h, max_pts=32):
    """
    pts : Nx2 float32 array in pixel coords (clockwise or CCW)
    returns flat list [x1,y1,x2,y2,…] normalised 0-1, ≤ max_pts*2 length
    """
    # drop alpha channel pts from the same corner to reduce count
    if len(pts) > max_pts:
        step = math.ceil(len(pts) / max_pts)
        pts = pts[::step]

    norm = lambda v, m: max(0.0, min(1.0, v / m))
    flat = [norm(x, img_w) if i % 2 == 0 else norm(x, img_h)
            for i, x in enumerate(pts.flatten())]
    return flat

def render_qyoo(dot_bits=None):
    """
    Return an RGBA PIL image of a single Qyoo with one squared corner.
    dot_bits : 36‑char string of 0/1, row‑major.
    """
    if dot_bits is None:
        dot_bits = ''.join(random.choice('01') for _ in range(DOT_GRID * DOT_GRID))

    # decide light‑on‑dark or dark‑on‑light
    light = tuple(random.randint(200, 255) for _ in range(3))
    dark  = tuple(random.randint(0, 55)    for _ in range(3))
    if random.random() < 0.5:
        fg, bg_col = dark, light    # dark shape on light bg
        dot_col = light
    else:
        fg, bg_col = light, dark    # light shape on dark bg
        dot_col = dark

    img = Image.new("RGBA", (SHAPE_SIZE, SHAPE_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    r = SHAPE_SIZE // 2
    # circle
    draw.ellipse((0, 0, 2 * r, 2 * r), fill=fg)
    # single square corner (bottom‑right)
    draw.rectangle((r, r, SHAPE_SIZE, SHAPE_SIZE), fill=fg)

    # dots
    cell = SHAPE_SIZE // (DOT_GRID + 2)
    off = cell
    for row in range(DOT_GRID):
        for col in range(DOT_GRID):
            if dot_bits[row * DOT_GRID + col] == '1':
                cx = off + col * cell + cell // 2
                cy = off + row * cell + cell // 2
                draw.ellipse((cx - DOT_R_PX, cy - DOT_R_PX,
                              cx + DOT_R_PX, cy + DOT_R_PX), fill=dot_col)

    # optional full inversion
    if random.random() < 0.5:
        arr = np.array(img)
        arr[..., :3] = 255 - arr[..., :3]
        img = Image.fromarray(arr, 'RGBA')

    return img, dot_bits


# ------------------ augment & composite --------------------------- #
def random_perspective(w, h):
    """Return src and dst points for cv2.getPerspectiveTransform"""
    margin = 0.1
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([[random.uniform(0, w*margin), random.uniform(0, h*margin)],
                      [random.uniform(w*(1-margin), w), random.uniform(0, h*margin)],
                      [random.uniform(w*(1-margin), w), random.uniform(h*(1-margin), h)],
                      [random.uniform(0, w*margin), random.uniform(h*(1-margin), h)]])
    return src, dst

def augment_symbol(sym_rgba, canvas_size=640):
    """
    Warp + rotate + scale + translate the RGBA symbol into a square canvas.
    Returns warped RGBA ndarray and the 4 destination points (for bbox).
    """
    # 1. random in‑plane rotation
    angle = random.uniform(0, 360)
    sym_rgba = sym_rgba.rotate(angle, expand=True)

    sym = np.array(sym_rgba)
    h, w = sym.shape[:2]

    # 2. random perspective (margin up to 0.35)
    margin = random.uniform(0.15, 0.35)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[random.uniform(0, w * margin),      random.uniform(0, h * margin)],
                      [random.uniform(w * (1 - margin), w), random.uniform(0, h * margin)],
                      [random.uniform(w * (1 - margin), w), random.uniform(h * (1 - margin), h)],
                      [random.uniform(0, w * margin),      random.uniform(h * (1 - margin), h)]])

    # ⛑️ Reject bad transforms
    if not np.isfinite(dst).all():
        return None, None

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(sym, M, (w, h), borderValue=(0, 0, 0, 0))

    # 3. random scale 15‑70 % of frame, random position
    tgt_size = random.uniform(0.15, 0.7) * canvas_size
    scale = tgt_size / max(w, h)
    warped = cv2.resize(warped, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    wh, ww = warped.shape[:2]
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    max_x = canvas_size - ww
    max_y = canvas_size - wh
    x0 = random.randint(0, max_x)
    y0 = random.randint(0, max_y)
    canvas[y0:y0 + wh, x0:x0 + ww] = warped

    # updated quad in canvas coordinates
    quad = dst * scale
    quad[:, 0] += x0
    quad[:, 1] += y0
    return canvas, quad

def random_background(canvas_size, bg_dir=None):
    if bg_dir and random.random() < 0.7:
        imgs = glob.glob(os.path.join(bg_dir, '*'))
        if imgs:
            bg = Image.open(random.choice(imgs)).convert('RGB').resize((canvas_size, canvas_size))
            return np.array(bg)
    # fallback solid / noise
    if random.random() < 0.5:
        col = tuple(random.randint(0,255) for _ in range(3))
        return np.full((canvas_size, canvas_size, 3), col, dtype=np.uint8)
    else:
        noise = np.random.randint(0,255,(canvas_size, canvas_size, 3), dtype=np.uint8)
        return noise

# ------------------ main loop ------------------------------------ #
def main(args):
    out_root = Path(args.out)
    img_dir = out_root/'train/images'
    lbl_dir = out_root/'train/labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(args.count):
        # 1. render symbol
        sym_img, bits = render_qyoo()

        # 2. augment
        canvas_size = args.imgsz
        warped_rgba, quad = augment_symbol(sym_img, canvas_size)
        if warped_rgba is None:
            continue  # skip malformed sample

        # 3. compose on background
        bg = random_background(canvas_size, args.bg_dir)
        alpha = warped_rgba[:,:,3:]/255.0
        comp = (warped_rgba[:,:,:3]*alpha + bg*(1-alpha)).astype(np.uint8)

        # 4. save
        img_path = img_dir/f"{idx:06d}.jpg"
        cv2.imwrite(str(img_path), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))

        # 5. YOLO-SEG label (class 0)
        # ---------------------------------------------------------------▶▶
        # tight bbox from the 4 warped quad points
        xs, ys = quad[:, 0], quad[:, 1]
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()

        cx = (x0 + x1) / 2 / canvas_size
        cy = (y0 + y1) / 2 / canvas_size
        bw = (x1 - x0)     / canvas_size
        bh = (y1 - y0)     / canvas_size

        # --- binary mask of the pasted symbol (whole canvas) -------------
        mask = (warped_rgba[:, :, 3] > 0).astype(np.uint8) * 255  # 0/255, H×W

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue                      # should not happen but be safe
        cnt = max(cnts, key=cv2.contourArea).squeeze()  # Nx2

        if cnt.ndim != 2 or cnt.shape[0] < 3:
            continue

        # subsample to ≤32 points for YOLO-Seg
        if cnt.shape[0] > 32:
            keep = np.linspace(0, cnt.shape[0] - 1, 32, dtype=int)  # use another name
            cnt  = cnt[keep]


        poly_norm = normalize_polygon(cnt, canvas_size, canvas_size)

        # ----------------------------------------------------------------▶▶

        lbl_path = lbl_dir/f"{idx:06d}.txt"
        with open(lbl_path, "w") as f:
            f.write(
                "0 " +
                f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " +
                " ".join(f"{p:.6f}" for p in poly_norm) + "\n"
            )


        if idx % 500 == 0:
            print(f"Generated {idx} / {args.count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=50000)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--out', type=str, default='dataset')
    parser.add_argument('--bg-dir', type=str, default=None, help='folder of background images')
    args = parser.parse_args()
    main(args)
