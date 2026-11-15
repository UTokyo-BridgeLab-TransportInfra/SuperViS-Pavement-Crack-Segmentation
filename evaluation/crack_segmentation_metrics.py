# -*- coding: utf-8 -*-
"""
Grid-level precision, recall and Dice for crack segmentation masks.

Crack masks are only a few pixels wide, so a pixel-wise comparison is dominated
by small offsets between the annotated line and the predicted line.
Both masks are therefore reduced to their skeleton (centre line) and compared on
a coarse grid of square cells. With 12 x 12 px cells, a 960 x 960 px
bird's-eye-view image is split into 80 x 80 = 6,400 cells of about 5 cm each.

Procedure for one image pair
  1. Binarize both masks: grey value > 20 is crack.
  2. Prediction only: remove 8-connected components smaller than 100 px
     (isolated speckles). The ground truth is not filtered.
  3. Skeletonize both masks (skimage.morphology.skeletonize).
  4. A cell is positive if it contains at least one skeleton pixel.
  5. Compare the two cell maps cell by cell -> TP, FP, FN, TN
     (TP + FP + FN + TN = number of cells).
  6. Per-image scores
       ground truth contains crack (TP + FN > 0):
           precision = TP / (TP + FP)
           recall    = TP / (TP + FN)
       ground truth is crack-free (TP + FN = 0):
           precision = (TP + TN) / number of cells   (fraction of cells classified correctly)
           recall    = 1
       Dice = 2 * precision * recall / (precision + recall)
     A small epsilon (1e-8) keeps the denominators of precision and recall non-zero,
     so an image whose ground truth has crack but whose prediction is empty scores 0.
  7. Dataset scores are the mean of the per-image scores.

Usage
  One image pair (prints the scores):
      python crack_seg_metrics.py --gt gt/0001.png --pred pred/0001.png
  Two folders (files are paired by name; the extension may differ):
      python crack_seg_metrics.py --gt GT_DIR --pred PRED_DIR --out per_image.csv
  Options:
      --thr 20  --min-area 100  --cell 12  --workers 4

Input masks are 8-bit greyscale (or RGB, converted to greyscale) with crack
pixels bright (e.g. 255) and background 0. Ground truth and prediction must
have the same size.

Requirements: numpy, opencv-python, scikit-image
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# One thread per process: the folder mode already runs several processes in parallel.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import cv2
import numpy as np
from skimage.morphology import skeletonize

cv2.setNumThreads(1)

# ---------------------------------------------------------------- parameters
BIN_THR = 20     # grey value above which a pixel counts as crack
MIN_AREA = 100   # predicted connected components smaller than this (in pixels) are discarded
CELL = 12        # cell side length in pixels (960 / 12 = 80 cells per side, ~5 cm each)
EPS = 1e-8       # keeps the denominators of precision and recall non-zero

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ---------------------------------------------------------------- mask processing
def read_mask(path):
    """Read an image file as an 8-bit greyscale array (also works with non-ASCII paths on Windows)."""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"cannot read image: {path}")
    return img


def binarize(img, thr=BIN_THR):
    """Crack where the grey value is above thr. A boolean mask is returned unchanged."""
    if img.dtype == bool:
        return img
    return img > thr


def remove_small_components(mask, min_area=MIN_AREA):
    """Remove 8-connected components with fewer than min_area pixels from a boolean mask."""
    if min_area <= 1:
        return mask
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False                      # label 0 is the background
    return keep[labels]


def cell_map(skel, cell=CELL):
    """Split the image into cell x cell blocks; a block is positive if it holds any skeleton pixel.

    Rows and columns at the bottom/right edge that do not fill a whole block are ignored
    (none for 960 x 960 images with 12 px cells)."""
    h, w = skel.shape
    hb, wb = h // cell, w // cell
    s = skel[:hb * cell, :wb * cell]
    return s.reshape(hb, cell, wb, cell).any(axis=(1, 3))


def gt_cells(gt, thr=BIN_THR, cell=CELL):
    """Ground truth: binarize -> skeletonize -> cell map (no component filtering)."""
    return cell_map(skeletonize(binarize(gt, thr)), cell)


def pred_cells(pred, thr=BIN_THR, min_area=MIN_AREA, cell=CELL):
    """Prediction: binarize -> remove small components -> skeletonize -> cell map."""
    return cell_map(skeletonize(remove_small_components(binarize(pred, thr), min_area)), cell)


# ---------------------------------------------------------------- metrics
def confusion(g, p):
    """Cell-level confusion counts (TP, FP, FN, TN) of two boolean cell maps."""
    tp = int(np.sum(g & p))
    fp = int(np.sum(~g & p))
    fn = int(np.sum(g & ~p))
    tn = int(np.sum(~g & ~p))
    return tp, fp, fn, tn


def scores(tp, fp, fn, tn, eps=EPS):
    """Precision, recall and Dice of one image from its cell counts."""
    if tp + fn == 0:                     # crack-free ground truth
        precision = (tp + tn) / (tp + fp + fn + tn)
        recall = 1.0
    else:
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
    s = precision + recall
    dice = 2 * precision * recall / s if s > 0 else 0.0
    return precision, recall, dice


def evaluate_pair(gt, pred, thr=BIN_THR, min_area=MIN_AREA, cell=CELL):
    """Evaluate one ground-truth / prediction pair (2-D arrays of the same size).

    Returns a dict with the cell counts, whether the ground truth is crack-free,
    and the precision, recall and Dice of the image."""
    if gt.shape != pred.shape:
        raise ValueError(f"size mismatch: ground truth {gt.shape} vs prediction {pred.shape}")
    tp, fp, fn, tn = confusion(gt_cells(gt, thr, cell), pred_cells(pred, thr, min_area, cell))
    precision, recall, dice = scores(tp, fp, fn, tn)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "gt_empty": int(tp + fn == 0),
            "precision": precision, "recall": recall, "dice": dice}


def evaluate_files(paths, thr=BIN_THR, min_area=MIN_AREA, cell=CELL):
    """Worker for the folder mode: paths = (ground-truth file, prediction file)."""
    gt_path, pred_path = paths
    return evaluate_pair(read_mask(gt_path), read_mask(pred_path), thr, min_area, cell)


# ---------------------------------------------------------------- folder mode
def list_images(folder):
    """Map file stem -> file path for all images in a folder."""
    out = {}
    for f in os.listdir(folder):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMG_EXT:
            if stem in out:
                raise ValueError(f"two images named '{stem}' in {folder}")
            out[stem] = os.path.join(folder, f)
    return out


def pair_folders(gt_dir, pred_dir):
    """Pair ground-truth and prediction files by name, ordered by the ground-truth file name."""
    gt, pred = list_images(gt_dir), list_images(pred_dir)
    missing, extra = sorted(set(gt) - set(pred)), sorted(set(pred) - set(gt))
    if missing or extra:
        raise ValueError(f"file names differ: {len(missing)} ground-truth images have no prediction "
                         f"(e.g. {missing[:3]}), {len(extra)} predictions have no ground truth (e.g. {extra[:3]})")
    stems = sorted(gt, key=lambda k: os.path.basename(gt[k]))
    return [(os.path.basename(gt[k]), gt[k], pred[k]) for k in stems]


def summarize(rows):
    """Mean precision / recall / Dice over all images, crack images and crack-free images."""
    groups = [("all images", rows),
              ("ground truth with crack", [r for r in rows if not r["gt_empty"]]),
              ("crack-free ground truth", [r for r in rows if r["gt_empty"]])]
    out = []
    for name, rs in groups:
        if rs:
            out.append((name, len(rs), *(float(np.mean([r[k] for r in rs])) for k in ("precision", "recall", "dice"))))
    return out


def main():
    ap = argparse.ArgumentParser(description="Grid-level precision / recall / Dice for crack masks.")
    ap.add_argument("--gt", required=True, help="ground-truth mask file or folder")
    ap.add_argument("--pred", required=True, help="predicted mask file or folder")
    ap.add_argument("--out", help="folder mode: per-image CSV (default: per_image_metrics.csv in the current folder)")
    ap.add_argument("--thr", type=int, default=BIN_THR, help=f"binarization threshold (default {BIN_THR})")
    ap.add_argument("--min-area", type=int, default=MIN_AREA,
                    help=f"remove predicted components smaller than this many pixels (default {MIN_AREA}; 0 = keep all)")
    ap.add_argument("--cell", type=int, default=CELL, help=f"cell side length in pixels (default {CELL})")
    ap.add_argument("--workers", type=int, default=4, help="parallel processes in folder mode (default 4)")
    a = ap.parse_args()

    # ---- one image pair
    if os.path.isfile(a.gt) and os.path.isfile(a.pred):
        r = evaluate_pair(read_mask(a.gt), read_mask(a.pred), a.thr, a.min_area, a.cell)
        print(f"cells {r['TP'] + r['FP'] + r['FN'] + r['TN']}: TP {r['TP']}  FP {r['FP']}  FN {r['FN']}  TN {r['TN']}"
              + ("  (crack-free ground truth)" if r["gt_empty"] else ""))
        print(f"precision {r['precision']:.4f}  recall {r['recall']:.4f}  Dice {r['dice']:.4f}")
        return
    if not (os.path.isdir(a.gt) and os.path.isdir(a.pred)):
        sys.exit("--gt and --pred must both be files or both be folders")

    # ---- two folders
    pairs = pair_folders(a.gt, a.pred)
    print(f"{len(pairs)} image pairs | threshold > {a.thr} | prediction components < {a.min_area} px removed "
          f"| {a.cell} px cells | {a.workers} processes", flush=True)
    work = partial(evaluate_files, thr=a.thr, min_area=a.min_area, cell=a.cell)
    jobs = [(g, p) for _, g, p in pairs]
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            results = list(ex.map(work, jobs, chunksize=4))
    else:
        results = [work(j) for j in jobs]

    rows = [{"index": i, "filename": name, **r} for i, ((name, _, _), r) in enumerate(zip(pairs, results), 1)]
    out = a.out or "per_image_metrics.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})   # full precision

    print(f"{'':26s}{'images':>8s}{'precision':>11s}{'recall':>9s}{'Dice':>9s}")
    for name, n, p, r, d in summarize(rows):
        print(f"{name:26s}{n:8d}{p:11.4f}{r:9.4f}{d:9.4f}")
    print("per-image results:", os.path.abspath(out))


if __name__ == "__main__":
    main()
