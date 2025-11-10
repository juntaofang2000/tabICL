#!/usr/bin/env python3
"""
Benchmark TabICLClassifier on OpenML-style datasets stored in TabZilla's datasets folder.

Expected dataset folder layout (per-dataset directory):
  - X.npy.gz
  - y.npy.gz
  - split_indeces.npy.gz   (optional but recommended)
  - metadata.json          (optional)

split_indeces handling (robust/flexible):
  - If boolean mask (0/1 or True/False) -> treat 1/True as TRAIN mask
  - If 1D integer array with values < n_samples -> treat as TRAIN indices (test = complement)
  - If 2D array with shape (2, n) -> treat first row as train_idx, second as test_idx
  - Otherwise fall back to an 80/20 stratified split when possible

This script mirrors the style of the TALENT bench script and writes results to outdir.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path
import sys
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def list_datasets(root: Path):
    return [d for d in sorted(root.iterdir()) if d.is_dir()]


def load_npy_gz(p: Path):
    # try direct np.load (some numpy versions accept .npy.gz), else use gzip
    try:
        return np.load(p, allow_pickle=False)
    except Exception:
        with gzip.open(p, 'rb') as fh:
            try:
                return np.load(fh, allow_pickle=False)
            except Exception:
                # last resort: allow pickle
                with gzip.open(p, 'rb') as fh2:
                    return np.load(fh2, allow_pickle=True)


def load_dataset(dirpath: Path) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Load X, y and optional (train_idx, test_idx) from dataset folder."""
    x_p = dirpath / 'X.npy.gz'
    y_p = dirpath / 'y.npy.gz'
    si_p = dirpath / 'split_indeces.npy.gz'

    if not x_p.exists() or not y_p.exists():
        raise FileNotFoundError(f"Missing X.npy.gz or y.npy.gz in {dirpath}")

    X = load_npy_gz(x_p)
    y = load_npy_gz(y_p)

    # ensure X shape
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    y = np.asarray(y)
    # ensure y is 1D
    if y.ndim > 1:
        if y.shape[1] == 1:
            y = y.squeeze(1)
        elif y.shape[0] == 1:
            y = y.squeeze(0)

    train_test_idx = None
    if si_p.exists():
        try:
            si = load_npy_gz(si_p)
            # handle npz
            if isinstance(si, np.lib.npyio.NpzFile):
                # try common keys
                if 'train' in si and 'test' in si:
                    train_idx = np.asarray(si['train'])
                    test_idx = np.asarray(si['test'])
                    train_test_idx = (train_idx, test_idx)
                else:
                    # pick first two arrays
                    files = list(si.files)
                    if len(files) >= 2:
                        train_test_idx = (np.asarray(si[files[0]]), np.asarray(si[files[1]]))
            else:
                si = np.asarray(si)
                if si.dtype == bool or set(np.unique(si)).issubset({0, 1}):
                    mask = si.astype(bool)
                    train_idx = np.where(mask)[0]
                    test_idx = np.where(~mask)[0]
                    train_test_idx = (train_idx, test_idx)
                elif si.ndim == 2 and si.shape[0] == 2:
                    train_test_idx = (np.asarray(si[0]), np.asarray(si[1]))
                elif si.ndim == 1 and np.issubdtype(si.dtype, np.integer) and np.max(si) < X.shape[0]:
                    # treat as train idx
                    train_idx = si.astype(int)
                    all_idx = np.arange(len(X))
                    test_idx = np.setdiff1d(all_idx, train_idx, assume_unique=True)
                    train_test_idx = (train_idx, test_idx)
                else:
                    train_test_idx = None
        except Exception:
            train_test_idx = None

    return X, y, train_test_idx


def make_train_test(X: np.ndarray, y: np.ndarray, idxs: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                    test_size: float = 0.2, random_state: int = 42):
    from sklearn.model_selection import train_test_split

    if idxs is not None:
        ti, te = idxs
        X_train = X[ti]
        y_train = y[ti]
        X_test = X[te]
        y_test = y[te]
        return X_train, X_test, y_train, y_test

    # fallback: stratify when applicable
    stratify = y if (len(np.unique(y)) > 1 and len(y) >= 2 * len(np.unique(y))) else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def evaluate_openml(data_root: str, model_path: str, outdir: str, max_datasets: Optional[int] = None,
                    skip_regression: bool = True, bins: int = 0, verbose: bool = False):
    # lazy import classifier
    from tabicl.sklearn.classifier import TabICLClassifier

    root = Path(data_root)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # instantiate classifier once
    clf = TabICLClassifier(verbose=verbose, model_path=model_path)

    ds = list_datasets(root)
    if max_datasets:
        ds = ds[:max_datasets]

    results = []

    from sklearn.utils.multiclass import type_of_target
    from sklearn.preprocessing import KBinsDiscretizer

    total_start = time.time()
    for d in ds:
        try:
            ds_start = time.time()
            try:
                X, y, idxs = load_dataset(d)
            except FileNotFoundError:
                logging.info(f"跳过 {d.name}: 数据文件缺失")
                continue

            X_train, X_test, y_train, y_test = make_train_test(X, y, idxs)

            # handle continuous labels
            tgt_type = None
            try:
                tgt_type = type_of_target(y_train)
            except Exception:
                tgt_type = None

            if tgt_type is not None and tgt_type.startswith('continuous'):
                if bins and bins > 1:
                    est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
                    y_train = est.fit_transform(y_train.reshape(-1, 1)).astype(int).ravel()
                    y_test = est.transform(y_test.reshape(-1, 1)).astype(int).ravel()
                    logging.info(f"{d.name}: continuous target -> {bins} bins")
                elif skip_regression:
                    logging.info(f"跳过 {d.name}: 连续标签 (regression) 检测到")
                    continue
                else:
                    logging.info(f"{d.name}: continuous target but trying to fit (skip_regression=False)")

            # fit & eval
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = float(np.mean(y_pred == y_test))
            duration = time.time() - ds_start
            logging.info(f"{d.name}: acc={acc:.4f} time={duration:.2f}s")
            results.append((d.name, acc, duration))

        except Exception as e:
            logging.exception(f"评测失败 {d.name}: {e}")

    if results:
        with open(outdir / 'openml_detailed.txt', 'w') as f:
            f.write('dataset\taccuracy\ttime_s\n')
            for name, acc, duration in results:
                f.write(f"{name}\t{acc:.6f}\t{duration:.3f}\n")

        total_time = sum(r[2] for r in results)
        avg_time = total_time / len(results)
        avg_acc = sum(r[1] for r in results) / len(results)
        with open(outdir / 'openml_summary.txt', 'w') as f:
            f.write(f"Total datasets: {len(results)}\n")
            f.write(f"Average accuracy: {avg_acc:.6f}\n")
            f.write(f"Total time s: {total_time:.3f}\n")
            f.write(f"Average time s: {avg_time:.3f}\n")

        logging.info(f"完成 {len(results)} 个数据集, 平均准确率 {avg_acc:.4f}, 总耗时 {total_time:.2f}s")
    else:
        logging.info("没有成功的评测结果。")


def main(argv=None):
    p = argparse.ArgumentParser(description='Benchmark TabICL on OpenML-format datasets (TabZilla)')
    p.add_argument('--data-root', default='/data0/fangjuntao2025/tabzilla-main/TabZilla/tabzill_benchmark', help='Root of OpenML-format datasets')
    p.add_argument('--model-path', default='/data0/fangjuntao2025/tabicl-main/tabICLOrignCheckpoint/tabicl-classifier-v1.1-0506.ckpt', help='Path to TabICL checkpoint')
    p.add_argument('--outdir', default='evaluation_results', help='Output directory for results')
    p.add_argument('--max-datasets', type=int, default=None, help='Limit number of datasets to evaluate')
    p.add_argument('--bins', type=int, default=0, help='If >1, discretize continuous targets into this many bins')
    p.add_argument('--no-skip-regression', dest='skip_regression', action='store_false', help='Do not skip regression targets (try to discretize or fit)')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='[%(levelname)s] %(message)s')
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(outdir / 'bench_openml.log')
    fh.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    fh.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(fh)

    evaluate_openml(args.data_root, args.model_path, args.outdir, max_datasets=args.max_datasets,
                    skip_regression=args.skip_regression, bins=args.bins, verbose=args.verbose)


if __name__ == '__main__':
    main()
