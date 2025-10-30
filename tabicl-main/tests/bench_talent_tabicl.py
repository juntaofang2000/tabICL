#!/usr/bin/env python3
"""
批量在 TALENT 数据目录上评测 TabICLClassifier 的脚本。

用法示例：
  python scripts/bench_talent_tabicl.py --model-path /path/to/checkpoint --data-root /path/to/TALENT/data --outdir evaluation_results

说明：
 - 脚本会尝试在每个子目录下寻找单个数据文件或 TRAIN/TEST 文件，若只找到单文件则按 80/20 做分割。
 - 需要可导入 `tabicl.sklearn.classifier.TabICLClassifier`，脚本本身的导入不会自动实例化模型（除非运行 main）。
 - 运行完整评测会实例化并加载 checkpoint，请在有显卡/足够内存时执行。
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import os
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import time


def find_data_files(dataset_dir: Path):
    """Return (train_path, test_path) if TRAIN/TEST files exist, else a single data file path or None."""
    # Only support TALENT-style datasets: expect N_train.npy, y_train.npy, N_test.npy, y_test.npy
    files = [p for p in dataset_dir.iterdir() if p.is_file()]
    lower_names = {p.name.lower(): p for p in files}

    def find_by_suffix(key: str):
        for name, p in lower_names.items():
            if name.endswith(key):
                return p
        return None

    n_train = find_by_suffix('n_train.npy')
    y_train = find_by_suffix('y_train.npy')
    n_test = find_by_suffix('n_test.npy')
    y_test = find_by_suffix('y_test.npy')
    if n_train and y_train and n_test and y_test:
        return (n_train, y_train), (n_test, y_test)

    # If the exact TALENT pattern is not present, return (None, None) and skip this dataset
    return None, None


def load_table(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single file and return (X, y). Assume label is first or last column.
    Supported: csv/tsv/parquet/npy/npz
    """
    # If file_path is a tuple (X_path, y_path) load pair
    if isinstance(file_path, tuple) or isinstance(file_path, list):
        Xp, yp = Path(file_path[0]), Path(file_path[1])
        return load_pair(Xp, yp)

    suffix = file_path.suffix.lower()
    if suffix in {'.npy', '.npz'}:
        try:
            arr = np.load(file_path, allow_pickle=False)
        except ValueError:
            # fallback to allow pickle if object arrays were saved
            arr = np.load(file_path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            # pick first array
            arr = arr[list(arr.files)[0]]
        data = np.asarray(arr)
    elif suffix == '.parquet':
        df = pd.read_parquet(file_path)
        data = df.values
    else:
        # try tsv if .tsv, else csv with auto sep
        sep = '\t' if file_path.suffix.lower() == '.tsv' else None
        df = pd.read_csv(file_path, sep=sep, header=None)
        data = df.values

    if data.ndim == 1:
        # likely a label-only file (unexpected here)
        raise ValueError(f"Unsupported 1D data in {file_path}")

    # assume label is first column if discrete, else last
    # Heuristic: if first column has few unique values (< number of rows/2), treat as label
    col0 = data[:, 0]
    try:
        uniques0 = np.unique(col0)
    except Exception:
        uniques0 = np.array([])

    if 0 < uniques0.size < max(2, data.shape[0] // 2):
        y = col0
        X = data[:, 1:]
    else:
        y = data[:, -1]
        X = data[:, :-1]

    # convert X to float, coerce non-numeric to nan then fill
    X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    # convert y to 1-d numpy
    y = pd.Series(y).values
    return X, y


def load_pair(X_path: Path, y_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load X and y from separate files. Supports numpy and csv/parquet."""
    def load_any(p: Path):
        s = p.suffix.lower()
        if s in {'.npy', '.npz'}:
            try:
                v = np.load(p, allow_pickle=False)
            except ValueError:
                v = np.load(p, allow_pickle=True)
            if isinstance(v, np.lib.npyio.NpzFile):
                v = v[list(v.files)[0]]
            return np.asarray(v)
        elif s == '.parquet':
            return pd.read_parquet(p).values
        else:
            sep = '\t' if s == '.tsv' else None
            return pd.read_csv(p, sep=sep, header=None).values

    X = load_any(X_path)
    y = load_any(y_path)

    # ensure y is 1d
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.squeeze(1)
    elif y.ndim > 1 and y.shape[0] == 1:
        y = y.squeeze(0)

    # ensure X is 2d
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # convert X to numeric
    X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    y = pd.Series(y).values
    return X, y


def split_train_test(X: np.ndarray, y: np.ndarray, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split

    stratify = y if (len(np.unique(y)) > 1 and len(y) >= 2 * len(np.unique(y))) else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def evaluate_datasets(model_path: str, data_root: str, outdir: str, max_datasets: Optional[int] = None, verbose: bool = False, skip_regression: bool = True, bins: int = 0):
    # lazy import classifier to avoid import-time heavy work when module imported
    from tabicl.sklearn.classifier import TabICLClassifier

    data_root = Path(data_root)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # instantiate classifier once
    clf = TabICLClassifier(verbose=verbose, model_path=model_path)

    results = []
    dirs = [d for d in sorted(data_root.iterdir()) if d.is_dir()]
    if max_datasets:
        dirs = dirs[:max_datasets]

    from sklearn.utils.multiclass import type_of_target
    from sklearn.preprocessing import KBinsDiscretizer

    total_start = time.time()
    for d in dirs:
        try:
            ds_start = time.time()
            train_path, test_path = find_data_files(d)
            if train_path is None and test_path is None:
                logging.info(f"跳过：{d} (没有可识别的数据文件)")
                continue

            if train_path and test_path:
                X_train, y_train = load_table(train_path)
                X_test, y_test = load_table(test_path)
            else:
                single = train_path
                X, y = load_table(single)
                X_train, X_test, y_train, y_test = split_train_test(X, y)

            # ensure shapes
            if X_train.ndim == 3 and X_train.shape[1] == 1:
                X_train = X_train.squeeze(1)
            if X_test.ndim == 3 and X_test.shape[1] == 1:
                X_test = X_test.squeeze(1)

            # check label type: skip/regress handling
            tgt_type = None
            try:
                tgt_type = type_of_target(y_train)
            except Exception:
                tgt_type = None

            if tgt_type is not None and tgt_type.startswith('continuous'):
                if bins and bins > 1:
                    # discretize continuous labels into bins (fit on train)
                    est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
                    y_train = est.fit_transform(y_train.reshape(-1, 1)).astype(int).ravel()
                    # apply same transform to test
                    y_test = est.transform(y_test.reshape(-1, 1)).astype(int).ravel()
                    logging.info(f"{d.name}: converted continuous target to {bins} bins for classification")
                elif skip_regression:
                    logging.info(f"跳过数据集 {d.name}: 连续标签 (regression target) 检测到，跳过")
                    continue
                else:
                    logging.info(f"{d.name}: 连续标签检测到，但继续尝试拟合 (skip_regression=False)")

            # fit and predict
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = float(np.mean(y_pred == y_test))
            ds_end = time.time()
            duration = ds_end - ds_start
            logging.info(f"{d.name}: accuracy={acc:.4f}  time={duration:.2f}s")
            results.append((d.name, acc, duration))

        except Exception as e:
            logging.exception(f"评测失败 {d.name}: {e}")

    if results:
        # save results with timing
        with open(outdir / 'talent_detailed.txt', 'w') as f:
            f.write('dataset\taccuracy\ttime_s\n')
            for name, acc, duration in results:
                f.write(f"{name}\t{acc:.6f}\t{duration:.3f}\n")
        total_time = sum(duration for _, _, duration in results)
        avg_time = total_time / len(results)
        avg_acc = sum(acc for _, acc, _ in results) / len(results)
        with open(outdir / 'talent_summary.txt', 'w') as f:
            f.write(f"Total datasets: {len(results)}\n")
            f.write(f"Average accuracy: {avg_acc:.6f}\n")
            f.write(f"Total time s: {total_time:.3f}\n")
            f.write(f"Average time s: {avg_time:.3f}\n")
        logging.info(f"评测完成，共 {len(results)} 个数据集，平均准确率 {avg_acc:.4f}, 总耗时 {total_time:.2f}s, 平均每数据集 {avg_time:.2f}s")
    else:
        logging.info("没有成功的评测结果。")


def main(argv=None):
    p = argparse.ArgumentParser(description='Benchmark TabICLClassifier on TALENT datasets')
    p.add_argument('--model-path', default='/data0/fangjuntao2025/tabicl-main/tabICLOrignCheckpoint/tabicl-classifier-v1.1-0506.ckpt', help='Path to TabICL checkpoint (model_path)')
    p.add_argument('--data-root', default='/data0/fangjuntao2025/tabicl-main/TALENT/data', help='Root path to TALENT data folder')
    p.add_argument('--outdir', default='evaluation_results', help='Directory to save results')
    p.add_argument('--max-datasets', type=int, default=None, help='Limit number of datasets to evaluate')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG, format='[%(levelname)s] %(message)s')

    # Also write logs to a file inside outdir (keep console output)
    outdir_path = Path(args.outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(outdir_path / 'bench_talent.log')
    file_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(file_handler)

    evaluate_datasets(args.model_path, args.data_root, args.outdir, max_datasets=args.max_datasets, verbose=args.verbose)


if __name__ == '__main__':
    main()
