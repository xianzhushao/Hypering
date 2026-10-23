#!/usr/bin/env python3
"""
Draw embeddings on the disk together with the learned stage rings.

Each scan is placed at
    radius = rho (its hyperbolic distance from the origin)      --view equidistant
             tanh(rho / 2), its true Poincare-disk radius        --view poincare
    angle  = direction of the first two PCA components of log_0(embedding)
and every stage ring k_c is drawn as a circle. Points are coloured by
--label-col if given, otherwise by the predicted stage.

    python visualize.py --pred pred.csv --model checkpoints/seed_042 \
        --output disk.png [--label-col stage] [--view poincare]
"""
import argparse
import os


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

COLORS = ['#1f77b4', '#17becf', '#2ca02c', '#9467bd', '#d62728']


def logmap0_np(x, c=1.0):
    n = np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, (1 - 1e-5) / np.sqrt(c))
    return np.arctanh(np.sqrt(c) * n) * x / (np.sqrt(c) * n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred', required=True, help='CSV written by predict.py')
    ap.add_argument('--model', required=True, help='seed folder (for the ring radii)')
    ap.add_argument('--output', default='disk.png')
    ap.add_argument('--label-col', default=None)
    ap.add_argument('--view', choices=['equidistant', 'poincare'], default='equidistant')
    args = ap.parse_args()

    df = pd.read_csv(args.pred)
    info = torch.load(os.path.join(args.model, 'model.pt'), map_location='cpu')
    radii, names = np.asarray(info['radii']), info['class_names']
    emb = df[[c for c in df.columns if c.startswith('emb_')]].to_numpy()
    rho = df['rho'].to_numpy()
    lab = (df[args.label_col] if args.label_col else df['pred']).to_numpy().astype(int)

    xy = PCA(n_components=2).fit_transform(logmap0_np(emb))
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    to_r = (lambda r: np.tanh(r / 2.0)) if args.view == 'poincare' else (lambda r: r)
    r = to_r(rho)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    t = np.linspace(0, 2 * np.pi, 400)
    if args.view == 'poincare':
        ax.plot(np.cos(t), np.sin(t), color='0.3', lw=1.0)       # disk boundary
    for k, (rk, n) in enumerate(zip(radii, names)):
        rr = to_r(rk)
        ax.plot(rr * np.cos(t), rr * np.sin(t), color=COLORS[k % 5], lw=2.0)
        ax.text(0, rr, f' {n}', color=COLORS[k % 5], fontsize=9,
                fontweight='bold', va='bottom')
    for k, n in enumerate(names):
        s = lab == k
        ax.scatter(r[s] * np.cos(theta[s]), r[s] * np.sin(theta[s]), s=16,
                   color=COLORS[k % 5], edgecolor='white', linewidth=0.4,
                   label=n, zorder=3)
    lim = (1.02 if args.view == 'poincare' else max(radii.max(), r.max()) * 1.08)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=len(names),
              frameon=False, fontsize=9, handletextpad=0.2)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f'written: {args.output}')


if __name__ == '__main__':
    main()
