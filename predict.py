#!/usr/bin/env python3
"""
Stage prediction with the pretrained ring model.

Input : CSV from preprocess.py (10878 `fmri_e*` edge columns; any other
        columns, e.g. scan_id or a label, are carried over to the output).
Output: CSV with, per scan, rho = hyperbolic distance from the origin, the
        affinity to each stage ring, the predicted stage and the embedding.

    # one model (seed 42)
    python predict.py --input features.csv --output pred.csv \
        --model checkpoints/seed_042

    # ensemble of all seeds in a folder (affinities averaged)
    python predict.py --input features.csv --output pred.csv \
        --model checkpoints
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from hypering import load_pretrained


def run_dirs(path):
    if os.path.exists(os.path.join(path, 'model.pt')):
        return [path]
    dirs = sorted(os.path.join(path, d) for d in os.listdir(path)
                  if os.path.exists(os.path.join(path, d, 'model.pt')))
    if not dirs:
        raise SystemExit(f'no model.pt under {path}')
    return dirs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--model', default='checkpoints/seed_042',
                    help='a seed folder, or the checkpoints folder to ensemble all seeds')
    ap.add_argument('--label-col', default=None,
                    help='optional column with true stages (0-4) to report accuracy')
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    feat_cols = [c for c in df.columns if c.startswith('fmri_e')]
    meta = df.drop(columns=feat_cols)
    x = torch.from_numpy(df[feat_cols].to_numpy(np.float32))

    dirs = run_dirs(args.model)
    aff, rho, emb = [], [], None
    for d in dirs:
        m = load_pretrained(d)
        if x.shape[1] != m.encoder.feat_norm.normalized_shape[0]:
            raise SystemExit(f'{x.shape[1]} feature columns, model expects '
                             f'{m.encoder.feat_norm.normalized_shape[0]}')
        out = m(x)
        aff.append(out['affinity'].numpy())
        rho.append(out['rho'].numpy())
        if emb is None:
            emb = out['embedding'].numpy()           # embedding of the first model only
    names = m.class_names
    aff, rho = np.mean(aff, axis=0), np.mean(rho, axis=0)
    pred = aff.argmax(axis=1)

    res = meta.copy()
    res['rho'] = rho
    for k, n in enumerate(names):
        res[f'affinity_{n}'] = aff[:, k]
    res['pred'] = pred
    res['pred_stage'] = [names[k] for k in pred]
    for j in range(emb.shape[1]):
        res[f'emb_{j}'] = emb[:, j]
    res.to_csv(args.output, index=False, float_format='%.6g')
    print(f'{len(dirs)} model(s); wrote {args.output} ({len(res)} rows)')
    print('predicted stages:', pd.Series(res['pred_stage']).value_counts().to_dict())
    if args.label_col:
        print(f"accuracy vs {args.label_col}: {(pred == df[args.label_col].to_numpy()).mean():.4f}")


if __name__ == '__main__':
    main()
