#!/usr/bin/env python3
"""
FC matrices -> model input (one row of 10878 edge features per scan).

Pipeline (the one the released models were trained with):
  1. 150 x 150 matrix: drop the two empty ROIs (indices 41, 116) -> 148 ROIs.
     A 148 x 148 matrix is used as is.
  2. Upper triangle, k = 1 -> C(148, 2) = 10878 edges (row-major order).
  3. Fisher z: arctanh(r), r clipped to +-0.999.
  4. z-score every edge across the scans of the cohort (mean 0, std 1).
     Pass --stats to reuse the mean/std of another cohort instead
     (written by a previous run with --save-stats).

Input : a folder of matrices, one file per scan (.npy, .txt or .csv; the
        scan id is the file name without extension).
Output: CSV with a `scan_id` column followed by the 10878 edge columns.

    python preprocess.py --input fc_matrices/ --output features.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

N_ROI_RAW = 150
DEAD_ROIS = (41, 116)
FISHER_CLIP = 0.999


def load_matrix(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path)
    return np.loadtxt(path, delimiter=',' if ext == '.csv' else None)


def edge_vector(M):
    if M.shape == (N_ROI_RAW, N_ROI_RAW):
        keep = np.setdiff1d(np.arange(N_ROI_RAW), DEAD_ROIS)
        M = M[np.ix_(keep, keep)]
    if M.shape != (148, 148):
        raise ValueError(f'expected a 150x150 or 148x148 matrix, got {M.shape}')
    iu = np.triu_indices(148, k=1)
    return np.arctanh(np.clip(M[iu], -FISHER_CLIP, FISHER_CLIP))


def column_names():
    keep = np.setdiff1d(np.arange(N_ROI_RAW), DEAD_ROIS)
    iu = np.triu_indices(148, k=1)
    # same names as in training: indices refer to the original 150-ROI numbering
    return [f'fmri_e{a}_{b}' for a, b in zip(keep[iu[0]], keep[iu[1]])]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, help='folder of FC matrices')
    ap.add_argument('--output', required=True, help='output CSV')
    ap.add_argument('--stats', help='.npz with mean/std to z-score with')
    ap.add_argument('--save-stats', help='write this cohort\'s mean/std to an .npz')
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.input)
                   if f.lower().endswith(('.npy', '.txt', '.csv')))
    if not files:
        raise SystemExit(f'no .npy/.txt/.csv matrices in {args.input}')
    E = np.stack([edge_vector(load_matrix(os.path.join(args.input, f))) for f in files])

    if args.stats:
        st = np.load(args.stats)
        mu, sd = st['mean'], st['std']
    else:
        if len(files) < 2:
            raise SystemExit('need >= 2 scans to z-score; pass --stats otherwise')
        mu, sd = E.mean(axis=0), E.std(axis=0)
    if args.save_stats:
        np.savez(args.save_stats, mean=mu, std=sd)
    dead = sd < 1e-12
    Z = (E - mu) / np.where(dead, 1.0, sd)
    Z[:, dead] = 0.0

    df = pd.DataFrame(Z.astype(np.float32), columns=column_names())
    df.insert(0, 'scan_id', [os.path.splitext(f)[0] for f in files])
    df.to_csv(args.output, index=False, float_format='%.5g')
    print(f'wrote {args.output}: {len(df)} scans x {Z.shape[1]} edges')


if __name__ == '__main__':
    main()
