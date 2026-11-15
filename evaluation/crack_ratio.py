# -*- coding: utf-8 -*-
"""
Grid-based pavement crack ratio (Japanese pavement inspection convention).

The road surface is divided into 50 cm x 50 cm cells. A cell scores 60% if it
contains one crack and 100% if it contains two or more cracks. The crack ratio
is the mean score over all cells.

Input : binary crack mask of one 4 m x 4 m road-surface image
        (crack = white, background = black; typically 960 x 960 px)
Output: crack ratio in percent, rounded to one decimal place

Procedure
  1. Binarise: pixels with grey level > 20 are crack pixels.
  2. Dilate once with a 3x3 kernel (closes 1-px gaps), then thin to a
     1-px-wide skeleton with skimage.morphology.skeletonize.
  3. Split the image into 8 x 8 square cells
     (960 px / 8 = 120 px per cell = 50 cm on a 4 m image).
  4. Score each cell:
        L  = skeleton pixels in the cell / cell side length (px)
             -> crack length in units of the cell side
        Nb = branch points in the cell
             -> skeleton pixels with >= 3 skeleton neighbours (8-connectivity,
                counted inside the cell only)
        L <= 0.25                                  -> 0.0  (no crack)
        (L > 1.0 and Nb > 0) or (L > 0.5 and Nb > 2) -> 1.0  (two or more cracks)#To reduce false branch-point detections caused by skeletonization artifacts, we applied a length-dependent criterion. Longer cracks were classified as complex when at least one branch point was detected, whereas shorter cracks required stronger evidence from multiple branch-point pixels. This empirical criterion reduces the influence of isolated skeleton spurs on grid-level crack classification.
        otherwise                                  -> 0.6  (one crack)
  5. Crack ratio = mean cell score x 100.

Usage
  python crack_ratio.py mask.png                         # one mask, prints the ratio
  python crack_ratio.py mask_dir -o result.xlsx          # folder -> Excel (image, crack_ratio)
  python crack_ratio.py mask_dir -o result.xlsx --cells  # also save the 8x8 cell scores
  python crack_ratio.py mask_dir --check reference.xlsx  # compare with existing results

Requirements: numpy, opencv-python, scikit-image, pandas, openpyxl
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.morphology import skeletonize

THRESH = 20        # binarisation threshold: grey level > THRESH is crack
GRID_ROWS = 8      # cells per image height; cell side = image height / GRID_ROWS
MIN_LEN = 0.25     # minimum normalised crack length for a cell to count as cracked
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

_K3 = np.ones((3, 3), np.float32)


# ----------------------------------------------------------------------------
# Reading and pre-processing
# ----------------------------------------------------------------------------
def read_mask(path):
    """Read a mask as uint8 greyscale.

    skimage returns floats in [0, 1] for colour or float images; these are
    scaled by 255 and truncated to uint8.
    """
    img = imread(str(path), as_gray=True)
    if img.dtype != np.uint8:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def to_skeleton(gray, thresh=THRESH):
    """Binarise -> 3x3 dilation -> skeleton (boolean array, 1-px-wide cracks)."""
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)
    return skeletonize(binary > 0)


# ----------------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------------
def grid_edges(h, w, rows=GRID_ROWS):
    """Row edges `ys` and column edges `xs` of the cell grid, in pixels.

    Cells are square with side h / rows. Along the width, a remaining strip
    of at least 0.3 cell is kept as an extra narrow column; a narrower strip is
    discarded. A 960 x 960 image gives the standard 8 x 8 grid.
    """
    side = h / rows
    n_full = int(w // side)
    xs = [k * side for k in range(n_full + 1)]
    if w - n_full * side >= 0.3 * side:
        xs.append(w)
    xs = np.unique([int(round(v)) for v in np.clip(xs, 0, w)]).tolist()
    if len(xs) < 2:                       # image narrower than 0.3 cell
        xs = [0, min(w, max(1, int(round(min(side, w)))))]
    ys = [int(round(v)) for v in np.linspace(0, h, rows + 1)]
    ys[-1] = h
    return ys, xs


# ----------------------------------------------------------------------------
# Cell scoring
# ----------------------------------------------------------------------------
def branch_points(cell):
    """Number of skeleton pixels with >= 3 skeleton neighbours inside the cell."""
    f = cell.astype(np.float32)
    neighbours = cv2.filter2D(f, -1, _K3, borderType=cv2.BORDER_CONSTANT) - f
    return int(np.count_nonzero(cell & (np.rint(neighbours) >= 3)))


def cell_score(cell):
    """Score of one cell: 0.0 (no crack), 0.6 (one crack) or 1.0 (two or more)."""
    if cell.size == 0:
        return 0.0
    length = cell.sum() / min(cell.shape)          # crack length / cell side
    if length <= MIN_LEN:
        return 0.0
    nb = branch_points(cell)
    if (length > 1.0 and nb > 0) or (length > 0.5 and nb > 2):
        return 1.0
    return 0.6


# ----------------------------------------------------------------------------
# Crack ratio
# ----------------------------------------------------------------------------
def crack_ratio(gray, return_cells=False):
    """Crack ratio (%) of a uint8 greyscale mask (crack = white).

    With return_cells=True, also returns the matrix of cell scores.
    """
    skel = to_skeleton(gray)
    ys, xs = grid_edges(*skel.shape)
    cells = np.array([[cell_score(skel[y0:y1, x0:x1])
                       for x0, x1 in zip(xs[:-1], xs[1:])]
                      for y0, y1 in zip(ys[:-1], ys[1:])])
    ratio = round(float(cells.mean()) * 100, 1)
    return (ratio, cells) if return_cells else ratio


def crack_ratio_file(path, return_cells=False):
    """Crack ratio (%) of a mask image file."""
    return crack_ratio(read_mask(path), return_cells)


# ----------------------------------------------------------------------------
# Batch processing
# ----------------------------------------------------------------------------
def _work(path):
    ratio, cells = crack_ratio_file(path, return_cells=True)
    return Path(path).stem, ratio, cells.ravel()


def run_folder(folder, workers=8):
    """Crack ratio of every mask in a folder.

    Returns two DataFrames: (image, crack_ratio) and the flattened cell scores.
    """
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_work, files, chunksize=8))
    ratios = pd.DataFrame([(n, r) for n, r, _ in res], columns=['image', 'crack_ratio'])
    n_cells = max((len(c) for _, _, c in res), default=0)
    cells = pd.DataFrame([list(c) + [np.nan] * (n_cells - len(c)) for _, _, c in res],
                         columns=[f'cell_{k + 1}' for k in range(n_cells)])
    cells.insert(0, 'image', ratios['image'])
    return ratios, cells


def check(ratios, reference_xlsx):
    """Compare with an existing Excel file (column 1 = image name, column 2 = crack ratio)."""
    ref = pd.read_excel(reference_xlsx).iloc[:, :2]
    ref.columns = ['image', 'reference']
    ref['image'] = ref['image'].astype(str)
    m = ratios.merge(ref, on='image')
    d = m['crack_ratio'] - m['reference']
    same = np.isclose(d, 0)
    print(f'{reference_xlsx}: {len(m)} images in common, {same.sum()} identical '
          f'({same.mean() * 100:.2f}%), mean absolute difference {d.abs().mean():.3f}')


def main():
    ap = argparse.ArgumentParser(description='Grid-based crack ratio (50 cm cells, scores 0 / 0.6 / 1.0)')
    ap.add_argument('input', help='a mask image or a folder of masks')
    ap.add_argument('-o', '--out', help='output Excel file (folder input)')
    ap.add_argument('--cells', action='store_true', help='also save per-image cell scores')
    ap.add_argument('--check', help='reference Excel file to compare against')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    src = Path(a.input)
    if src.is_file():
        ratio, cells = crack_ratio_file(src, return_cells=True)
        print(f'{src.name}: crack ratio {ratio}%')
        print(cells)
        return

    ratios, cells = run_folder(src, a.workers)
    print(f'{len(ratios)} masks, mean crack ratio {ratios["crack_ratio"].mean():.2f}%')
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(out) as w:
            ratios.to_excel(w, sheet_name='crack_ratio', index=False)
            if a.cells:
                cells.to_excel(w, sheet_name='cells', index=False)
        print('saved', out)
    if a.check:
        check(ratios, a.check)


if __name__ == '__main__':
    main()
