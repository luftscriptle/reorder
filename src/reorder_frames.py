#!/usr/bin/env python3
"""
Recover true frame order from a folder of shuffled frames
using Optical Flow Error + Optimal Transport + 2-opt refinement.

Output:
- recovered_permutation.txt (indices of original files)
- reordered_frames/frame_0001.png ... (frames in correct order)

Usage:
    python recover_order_ot_flow.py --frames path/to/frames

Dependencies:
    pip install opencv-python numpy scikit-learn POT tqdm
"""

import os
import sys
import glob
import math
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.manifold import SpectralEmbedding
import ot  # Python Optimal Transport (POT)

MAX_SIZE = 250  # Resize longest side to this for flow computation

def load_and_resize(img_path, max_side):
    im = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    if max_side is not None and max(im.shape[:2]) > max_side:
        h, w = im.shape[:2]
        scale = max_side / float(max(h, w))
        im = cv2.resize(im, (int(round(w*scale)), int(round(h*scale))), interpolation=cv2.INTER_AREA)
    return im


def optical_flow(img1_gray, img2_gray):
    """Dense optical flow (Farnebäck)."""
    return cv2.calcOpticalFlowFarneback(
        img1_gray, img2_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

def warp(img, flow):
    """Warp image using optical flow (forward flow)."""
    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped


def optical_flow_error(img1, img2):
    """Compute symmetric optical flow reconstruction error."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow12 = optical_flow(g1, g2)
    warped2 = warp(img2, flow12)
    err12 = np.mean(np.abs(img1.astype(np.float32) - warped2.astype(np.float32))) / 255.0

    flow21 = optical_flow(g2, g1)
    warped1 = warp(img1, flow21)
    err21 = np.mean(np.abs(img2.astype(np.float32) - warped1.astype(np.float32))) / 255.0

    return 0.5 * (err12 + err21)


def build_flow_cost_matrix(frames):
    """Compute pairwise symmetric optical flow error matrix."""
    n = len(frames)
    C = np.full((n, n), np.inf, dtype=np.float32)
    for i in tqdm(range(n), desc="Computing flow errors"):
        for j in range(i + 1, n):
            e = optical_flow_error(frames[i], frames[j])
            C[i, j] = C[j, i] = e
    return C


def spectral_embedding_1d(C):
    """Compute 1D spectral embedding from cost matrix."""
    Cf = C[np.isfinite(C)]
    sigma = np.median(Cf) if np.median(Cf) > 1e-9 else np.mean(Cf)
    W = np.exp(-(C ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(W, 1.0)
    emb = SpectralEmbedding(n_components=1, affinity='precomputed', random_state=0)
    z = emb.fit_transform(W).squeeze()
    z = (z - z.min()) / max(1e-9, z.max() - z.min())
    return z


def sinkhorn_time_alignment(z, reg=0.01):
    """Align 1D coordinates z to uniform timeline via OT (Sinkhorn)."""
    N = len(z)
    a = np.ones(N) / N
    b = np.ones(N) / N
    p = (np.arange(N) + 0.5) / N
    C_ot = (z[:, None] - p[None, :]) ** 2
    T = ot.sinkhorn(a, b, C_ot, reg)
    bary = (T * np.arange(N)[None, :]).sum(axis=1)
    return bary


def path_cost(order, C):
    return float(sum(C[order[i], order[i + 1]] for i in range(len(order) - 1)))


def two_opt(order, C):
    """Simple 2-opt local refinement."""
    best = order[:]
    best_cost = path_cost(best, C)
    improved = True
    n = len(best)
    while improved:
        improved = False
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                new_order = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                c = path_cost(new_order, C)
                if c + 1e-9 < best_cost:
                    best, best_cost = new_order, c
                    improved = True
                    break
            if improved:
                break
    return best, best_cost


def write_frames(order, files, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for t, idx in enumerate(order, start=1):
        img = cv2.imread(files[idx], cv2.IMREAD_COLOR)
        out_path = os.path.join(out_dir, f"frame_{t:04d}.png")
        cv2.imwrite(out_path, img)
    print(f"Saved reordered frames to: {out_dir}")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Recover frame order using Optical Flow + OT")
    ap.add_argument('--frames', type=str, required=True, help="Path to folder with frames")
    args = ap.parse_args()

    # Gather files
    exts = ("*.png", "*.jpg", "*.jpeg")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(args.frames, ext)))
    if not files:
        sys.exit("No image files found in folder.")



    print(f"Found {len(files)} frames in {args.frames}")

    # Load & downscale
    small_frames = [load_and_resize(f, MAX_SIZE) for f in tqdm(files, desc="Loading frames")]

    # Compute optical flow cost matrix
    C = build_flow_cost_matrix(small_frames)

    # Spectral embedding
    z = spectral_embedding_1d(C)

    # OT alignment to timeline
    bary = sinkhorn_time_alignment(z)

    # Sort and refine
    order_init = np.argsort(bary).tolist()
    order_rev = order_init[::-1]
    best, cost_best = two_opt(order_init, C)
    best_rev, cost_rev = two_opt(order_rev, C)
    if cost_rev < cost_best:
        best, cost_best = best_rev, cost_rev

    perm_path = os.path.join(args.frames, "recovered_permutation.txt")
    np.savetxt(perm_path, best, fmt="%d")
    print(f"Saved permutation to {perm_path}")

    out_dir = os.path.join(args.frames, "reordered_frames")
    write_frames(best, files, out_dir)



if __name__ == "__main__":
    main()
