"""
Precision, recall and Dice of the image matching.The inference code uses LoFTR as an example. For other models, simply replace the model.

Matcher
    Official LoFTR (https://github.com/zju3dv/LoFTR, commit df7ca80) with the outdoor dual-softmax
    weights (outdoor_ds.ckpt). Configuration: the repository's default_cfg with
    coarse.temp_bug_fix = False (required by this checkpoint) and match_coarse.thr = 0.1
    (repository default 0.2). Each image pair is run once; the fine matches (mkpts0_f, mkpts1_f) are scored.

Input
    Both images are read as grayscale, resized to 480 x 640 (width x height) and scaled to [0, 1].
    Ground truth is a 3 x 3 homography H (text file, comma-separated) that maps image0 pixel
    coordinates to image1 pixel coordinates, both in the 480 x 640 frame.

Block-level scoring (32 x 32 blocks; image0 is a 15 x 20 grid of 300 blocks)
    matched block  an image0 block that contains at least one fine match              -> N
    predicted      the median image1 coordinate of the matches in that block
    true           the block centre mapped into image1 by H
    correct block  predicted and true points fall in the same 32 x 32 block of image1
                   (zero tolerance)                                                    -> M

    precision = M / N                                    (0 if there are no matches)
    recall    = min(M / (overlap_ratio * 234), 1)
                overlap_ratio = area of the warped image0 that lies inside image1 / image area;
                234 = 13 x 18, the blocks of the 15 x 20 grid without the outermost ring
    dice      = 2 * precision * recall / (precision + recall)   (0 if both are 0)

    Fig. 3a lists these per-pair values; the reported figure is their plain mean over the 14,900 pairs.

Usage
    git clone https://github.com/zju3dv/LoFTR.git        # code; outdoor_ds.ckpt is linked in its README
    python loftr_fig3a_metrics.py --loftr_root LoFTR --ckpt outdoor_ds.ckpt \
        --img0_dir img0_raw1 --img1_dir img1_raw1 --h_dir h_txt1 \
        --pairs fig3a_pairs.csv --out loftr_fig3a_per_pair.csv

    fig3a_pairs.csv has two columns: File_names (row number in Fig. 3a) and pair_name; the files
    are <img0_dir>/<pair_name>.jpg, <img1_dir>/<pair_name>.jpg and <h_dir>/<pair_name>.txt.

Requirements: torch, kornia, einops, yacs, opencv-python, shapely, numpy, pandas.
"""
import argparse
import sys
import types
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
import torch
from shapely.geometry import Polygon

W_IMG, H_IMG = 480, 640                 # network input size (width, height)
BLOCK = 32                              # block side in pixels
N_COLS, N_ROWS = W_IMG // BLOCK, H_IMG // BLOCK   # 15 x 20 blocks
N_GT_BLOCKS = 234                       # 13 x 18 inner blocks: recall denominator per unit overlap;Following the border-removal setting of the original LoFTR implementation, the outermost ring of the 15 × 20 patch grid was excluded for all methods, leaving 234 interior patches; precision, recall and Dice were computed for each image pair and averaged over all pairs
COARSE_THR = 0.1                        # LoFTR coarse-match confidence threshold


# --------------------------------------------------------------------------- matcher
def load_loftr(loftr_root, ckpt, device):
    """Build official LoFTR with the configuration described above and load the weights."""
    import kornia
    try:                                # LoFTR imports kornia.utils.grid, which newer kornia removed
        import kornia.utils.grid  # noqa: F401
    except ImportError:
        shim = types.ModuleType('kornia.utils.grid')
        shim.create_meshgrid = kornia.utils.create_meshgrid
        sys.modules['kornia.utils.grid'] = shim
    sys.path.insert(0, loftr_root)
    from src.loftr import LoFTR, default_cfg

    cfg = deepcopy(default_cfg)
    cfg['coarse']['temp_bug_fix'] = False
    cfg['match_coarse']['thr'] = COARSE_THR
    model = LoFTR(config=cfg)
    model.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
    return model.eval().to(device)


def load_gray(path, device):
    """Grayscale image -> 1 x 1 x 640 x 480 tensor in [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (W_IMG, H_IMG))
    return torch.from_numpy(img)[None, None].float().to(device) / 255.0


@torch.no_grad()
def match(model, path0, path1, device):
    """Run LoFTR on one pair; return fine matches (N x 2 each, pixels in the 480 x 640 frame)."""
    batch = {'image0': load_gray(path0, device), 'image1': load_gray(path1, device)}
    model(batch)
    return batch['mkpts0_f'].cpu().numpy(), batch['mkpts1_f'].cpu().numpy()


# --------------------------------------------------------------------------- geometry
def load_homography(path):
    return np.loadtxt(path, delimiter=',', dtype=np.float64).reshape(3, 3)


def apply_homography(pts, H):
    """Map N x 2 points with a 3 x 3 homography."""
    ph = np.hstack([pts.astype(np.float64), np.ones((len(pts), 1))])
    out = (H @ ph.T).T
    return out[:, :2] / out[:, 2:3]


def overlap_ratio(H):
    """Fraction of image1 covered by image0 warped with H (both 480 x 640)."""
    corners = np.array([[0, 0], [W_IMG, 0], [W_IMG, H_IMG], [0, H_IMG]], dtype=np.float64)
    try:
        warped = Polygon(apply_homography(corners, H))
        if not warped.is_valid:
            warped = warped.buffer(0)
        inter = warped.intersection(Polygon(corners)).area
    except Exception:
        return 0.0
    return float(inter) / float(W_IMG * H_IMG)


# --------------------------------------------------------------------------- metrics
def block_counts(pts0, pts1, H):
    """Return (M correct blocks, N matched blocks) under the same-block rule."""
    if len(pts0) == 0:
        return 0, 0
    pts0, pts1 = np.asarray(pts0, np.float64), np.asarray(pts1, np.float64)
    bx = np.floor(pts0[:, 0] / BLOCK).astype(int)
    by = np.floor(pts0[:, 1] / BLOCK).astype(int)
    keep = (bx >= 0) & (bx < N_COLS) & (by >= 0) & (by < N_ROWS)
    pts1, bx, by = pts1[keep], bx[keep], by[keep]
    if len(pts1) == 0:
        return 0, 0

    bid = by * N_COLS + bx
    blocks = np.unique(bid)                                   # matched image0 blocks
    centres = np.stack([(blocks % N_COLS + 0.5) * BLOCK,      # source point = block centre
                        (blocks // N_COLS + 0.5) * BLOCK], axis=1)
    pred = np.array([np.median(pts1[bid == b], axis=0) for b in blocks])  # predicted target
    true = apply_homography(centres, H)                       # ground-truth target

    same = (np.floor(pred / BLOCK).astype(int) == np.floor(true / BLOCK).astype(int)).all(axis=1)
    return int(same.sum()), len(blocks)


def pair_metrics(M, N, ov):
    """Precision, recall (GT = overlap x 234, clipped to 1) and Dice for one pair."""
    precision = M / N if N > 0 else 0.0
    gt = ov * N_GT_BLOCKS
    recall = min(M / gt, 1.0) if gt > 0 else 0.0
    dice = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, dice


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--loftr_root', required=True, help='clone of github.com/zju3dv/LoFTR')
    ap.add_argument('--ckpt', required=True, help='outdoor_ds.ckpt')
    ap.add_argument('--img0_dir', required=True)
    ap.add_argument('--img1_dir', required=True)
    ap.add_argument('--h_dir', required=True)
    ap.add_argument('--pairs', required=True, help='CSV with columns File_names, pair_name')
    ap.add_argument('--out', default='loftr_fig3a_per_pair.csv')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--limit', type=int, default=0, help='score only the first N pairs (0 = all)')
    a = ap.parse_args()

    device = torch.device(a.device)
    model = load_loftr(a.loftr_root, a.ckpt, device)
    pairs = pd.read_csv(a.pairs, dtype={'pair_name': str})
    if a.limit:
        pairs = pairs.head(a.limit)

    rows = []
    for k, (fid, name) in enumerate(zip(pairs.File_names, pairs.pair_name), 1):
        pts0, pts1 = match(model, f'{a.img0_dir}/{name}.jpg', f'{a.img1_dir}/{name}.jpg', device)
        H = load_homography(f'{a.h_dir}/{name}.txt')
        M, N = block_counts(pts0, pts1, H)
        ov = overlap_ratio(H)
        p, r, d = pair_metrics(M, N, ov)
        rows.append(dict(File_names=fid, pair_name=name, n_matches=len(pts0), matched_blocks=N,
                         correct_blocks=M, overlap_ratio=ov, precision=p, recall=r, dice=d))
        if k % 500 == 0:
            print(f'{k}/{len(pairs)}', flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(a.out, index=False)
    print(f'{len(res)} pairs -> {a.out}')
    print('mean precision %.4f  recall %.4f  dice %.4f' % tuple(res[['precision', 'recall', 'dice']].mean()))


if __name__ == '__main__':
    main()
