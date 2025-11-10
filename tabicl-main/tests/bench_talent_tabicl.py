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
from typing import Optional, Tuple, Union

import json
import numpy as np
import pandas as pd
import time


def convert_features(X: np.ndarray, enabled: bool) -> np.ndarray:
    """Optionally coerce feature matrix to numeric values.

    When enabled, columns that cannot be parsed as numeric are ordinal-encoded
    so each distinct string receives a stable integer id (0, 1, 2, ...).
    """

    X = np.asarray(X)
    if not enabled:
        return X

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    df = pd.DataFrame(X)
    encoded = pd.DataFrame(index=df.index)

    for col in df.columns:
        series = df.iloc[:, col]
        numeric_series = pd.to_numeric(series, errors='coerce')

        if series.isna().equals(numeric_series.isna()):
            encoded[col] = numeric_series
        else:
            string_series = series.astype("string")
            codes, uniques = pd.factorize(string_series, sort=True)
            codes = codes.astype(np.int32)
            if (codes == -1).any():
                codes[codes == -1] = len(uniques)
            encoded[col] = codes

    return encoded.fillna(0).values.astype(np.float32)


def handle_missing_entries(X: np.ndarray, y: np.ndarray, *, context: str) -> tuple[np.ndarray, np.ndarray]:
    """Apply missing-value policy and keep X/y aligned."""

    X = np.asarray(X)
    y = np.asarray(y)
    context = context or "dataset"

    df = pd.DataFrame(X)
    # Ensure y alignment with dataframe rows
    y_series = pd.Series(y, index=df.index)

    drop_mask = pd.Series(False, index=df.index)

    for col in df.columns:
        series = df.iloc[:, col]
        numeric_series = pd.to_numeric(series, errors='coerce')

        # Numeric column (or successfully coerced)
        if series.isna().equals(numeric_series.isna()):
            nan_mask = numeric_series.isna()
            if nan_mask.any():
                mean_value = float(numeric_series.mean(skipna=True))
                if np.isnan(mean_value):
                    mean_value = 0.0
                df.iloc[:, col] = numeric_series.fillna(mean_value)
                logging.info(
                    "%s: 数值列 %s 使用均值 %.6f 填充 %d 个 NaN",
                    context,
                    col,
                    mean_value,
                    int(nan_mask.sum()),
                )
        else:
            # String column with missing: drop entire row later
            nan_mask = series.isna()
            if nan_mask.any():
                drop_mask |= nan_mask

    if drop_mask.any():
        drop_count = int(drop_mask.sum())
        df = df.loc[~drop_mask].copy()
        y_series = y_series.loc[df.index]
        logging.info("%s: 删除 %d 行包含字符串缺失值", context, drop_count)

    return df.values, y_series.values


def count_missing(values: np.ndarray) -> int:
    """Count NaN/None entries in the given array-like object."""

    if values is None:
        return 0

    arr = np.asarray(values)
    if arr.dtype.kind in {"f", "c"}:
        return int(np.isnan(arr).sum())

    mask = pd.isna(pd.DataFrame(arr))
    return int(mask.values.sum())


def log_nan_presence(context: str, values: np.ndarray, *, dataset_id: str | None = None, missing_registry: set[str] | None = None) -> None:
    """Log a warning when NaNs are present and record the dataset if requested."""

    missing = count_missing(values)
    if missing:
        logging.warning(f"{context}: 原始数据包含 {missing} 个 NaN/缺失值")
        if dataset_id and missing_registry is not None:
            missing_registry.add(dataset_id)


def find_data_files(dataset_dir: Path):
    """Return (train_path, val_path, test_path) if TALENT-style files exist."""
    # Only support TALENT-style datasets: expect N_train.npy, y_train.npy, N_test.npy, y_test.npy (val optional)
    files = [p for p in dataset_dir.iterdir() if p.is_file()]
    lower_names = {p.name.lower(): p for p in files}

    def find_by_suffix(key: str):
        for name, p in lower_names.items():
            if name.endswith(key):
                return p
        return None

    n_train = find_by_suffix('n_train.npy')
    c_train = find_by_suffix('c_train.npy')
    y_train = find_by_suffix('y_train.npy')
    n_val = find_by_suffix('n_val.npy')
    c_val = find_by_suffix('c_val.npy')
    y_val = find_by_suffix('y_val.npy')
    n_test = find_by_suffix('n_test.npy')
    c_test = find_by_suffix('c_test.npy')
    y_test = find_by_suffix('y_test.npy')

    if y_train and y_test and (n_train or c_train) and (n_test or c_test):
        val_pair = None
        if y_val and (n_val or c_val):
            val_pair = (n_val, c_val, y_val)
        return (n_train, c_train, y_train), val_pair, (n_test, c_test, y_test)

    # Try fallback: a single data table that needs heuristic splitting later
    table_candidates = [p for p in files if p.suffix.lower() in {'.npy', '.npz', '.csv', '.tsv', '.parquet'}]
    if len(table_candidates) == 1:
        return table_candidates[0], None, None

    # If nothing matched, return (None, None, None) and let caller skip
    return None, None, None


def load_array(file_path: Path) -> np.ndarray:
    """Load array-like data from numpy/csv/parquet files."""
    suffix = file_path.suffix.lower()
    if suffix in {'.npy', '.npz'}:
        try:
            arr = np.load(file_path, allow_pickle=False)
        except ValueError:
            arr = np.load(file_path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[list(arr.files)[0]]
        return np.asarray(arr)
    if suffix == '.parquet':
        return pd.read_parquet(file_path).values
    sep = '\t' if suffix == '.tsv' else None
    return pd.read_csv(file_path, sep=sep, header=None).values


def load_table(file_path: Union[Path, Tuple], context: str = "", coerce_numeric: bool = False, dataset_id: str | None = None, missing_registry: set[str] | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load a dataset split and return (X, y).

    When provided with a tuple/list, it delegates to helpers that already know where labels live.
    For single files it falls back to a simple heuristic (first vs last column) and logs the usage.
    `coerce_numeric` controls whether non-numeric entries are converted to floats via pandas.
    """
    # If file_path is a tuple (X_path, y_path) load pair
    if isinstance(file_path, (tuple, list)):
        if len(file_path) == 2:
            Xp, yp = Path(file_path[0]), Path(file_path[1])
            return load_pair(Xp, yp, context=context, coerce_numeric=coerce_numeric, dataset_id=dataset_id, missing_registry=missing_registry)
        if len(file_path) == 3:
            num_path, cat_path, y_path = file_path
            return load_split(
                Path(num_path) if num_path else None,
                Path(cat_path) if cat_path else None,
                Path(y_path),
                context=context,
                coerce_numeric=coerce_numeric,
                dataset_id=dataset_id,
                missing_registry=missing_registry,
            )
        raise ValueError(f"Unsupported tuple format for load_table: {file_path}")

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

    log_target = context or str(file_path)
    log_nan_presence(f"{log_target}-raw", data, dataset_id=dataset_id, missing_registry=missing_registry)

    # assume label is first column if discrete, else last
    # Heuristic: if first column has few unique values (< number of rows/2), treat as label
    col0 = data[:, 0]
    try:
        uniques0 = np.unique(col0)
    except Exception:
        uniques0 = np.array([])

    heuristic_column = None
    if 0 < uniques0.size < max(2, data.shape[0] // 2):
        y = col0
        X = data[:, 1:]
        heuristic_column = 'first'
    else:
        y = data[:, -1]
        X = data[:, :-1]
        heuristic_column = 'last'

    log_nan_presence(f"{log_target}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    log_nan_presence(f"{log_target}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)

    y = pd.Series(y).values
    clean_context = log_target
    X, y = handle_missing_entries(X, y, context=clean_context)
    X = convert_features(X, coerce_numeric)

    if heuristic_column:
        target = context or str(file_path)
        logging.info(f"{target}: 使用单文件启发式拆分标签 (取 {heuristic_column} 列)")
    return X, y


def load_pair(X_path: Path, y_path: Path, context: str = "", coerce_numeric: bool = False, dataset_id: str | None = None, missing_registry: set[str] | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load X and y from separate files. Supports numpy and csv/parquet."""
    X = load_array(X_path)
    y = load_array(y_path)

    ctx = context or X_path.stem
    log_nan_presence(f"{ctx}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    log_nan_presence(f"{ctx}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)

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

    y = pd.Series(y).values
    clean_context = ctx
    X, y = handle_missing_entries(X, y, context=clean_context)
    X = convert_features(X, coerce_numeric)
    return X, y


def load_split(num_path: Optional[Path], cat_path: Optional[Path], y_path: Path, context: str = "", coerce_numeric: bool = False, dataset_id: str | None = None, missing_registry: set[str] | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load numeric and categorical feature arrays (if any) and concatenate."""
    features = []
    ctx_base = context or (num_path.stem if num_path else (cat_path.stem if cat_path else y_path.stem))
    if num_path:
        X_num = load_array(num_path)
        X_num = np.asarray(X_num)
        if X_num.ndim == 1:
            X_num = X_num.reshape(-1, 1)
        log_nan_presence(f"{ctx_base}-num_raw", X_num, dataset_id=dataset_id, missing_registry=missing_registry)
        features.append(X_num)
    if cat_path:
        X_cat = load_array(cat_path)
        X_cat = np.asarray(X_cat)
        if X_cat.ndim == 1:
            X_cat = X_cat.reshape(-1, 1)
        log_nan_presence(f"{ctx_base}-cat_raw", X_cat, dataset_id=dataset_id, missing_registry=missing_registry)
        features.append(X_cat)

    if not features:
        raise ValueError("No numeric or categorical feature files found for split")

    # ensure feature arrays have matching sample counts
    n_samples = features[0].shape[0]
    for idx, feat in enumerate(features):
        if feat.shape[0] != n_samples:
            raise ValueError(f"Feature array #{idx} has mismatched sample count: {feat.shape[0]} vs {n_samples}")

    X = features[0] if len(features) == 1 else np.concatenate(features, axis=1)
    log_nan_presence(f"{ctx_base}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)

    y = load_array(y_path)
    y = np.asarray(y)
    log_nan_presence(f"{ctx_base}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.squeeze(1)
    elif y.ndim > 1 and y.shape[0] == 1:
        y = y.squeeze(0)
    y = pd.Series(y).values
    clean_context = ctx_base
    X, y = handle_missing_entries(X, y, context=clean_context)
    X = convert_features(X, coerce_numeric)
    return X, y


CLASSIFICATION_TASKS = {'binclass', 'multiclass'}


def load_dataset_info(dataset_dir: Path) -> Optional[dict]:
    """Load dataset metadata from info.json if present."""
    info_path = dataset_dir / 'info.json'
    if not info_path.exists():
        return None
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as exc:
        logging.warning(f"读取 {info_path} 失败: {exc}")
        return None


def summarize_task_types(dirs: list[Path]) -> dict[str, int]:
    """Return a count of task types across TALENT dataset directories."""

    counts = {'regression': 0, 'binclass': 0, 'multiclass': 0, 'unknown': 0}
    for dataset_dir in dirs:
        info = load_dataset_info(dataset_dir)
        task_type = None
        if info:
            task_type = str(info.get('task_type', '')).lower()

        if not task_type:
            counts['unknown'] += 1
        elif task_type in counts:
            counts[task_type] += 1
        else:
            counts['unknown'] += 1

    logging.info(
        "TALENT 数据集任务统计: regression=%d, binclass=%d, multiclass=%d, unknown=%d (总计 %d)",
        counts['regression'],
        counts['binclass'],
        counts['multiclass'],
        counts['unknown'],
        len(dirs),
    )
    return counts


def split_train_test(X: np.ndarray, y: np.ndarray, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split

    stratify = y if (len(np.unique(y)) > 1 and len(y) >= 2 * len(np.unique(y))) else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def evaluate_datasets(model_path: str, data_root: str, outdir: str, max_datasets: Optional[int] = None, verbose: bool = False, skip_regression: bool = True, bins: int = 0, merge_val: bool = False, coerce_numeric: bool = True):
    # lazy import classifier to avoid import-time heavy work when module imported
    from tabicl.sklearn.classifier import TabICLClassifier

    data_root = Path(data_root)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # instantiate classifier once
    clf = TabICLClassifier(verbose=verbose, model_path=model_path)

    results = []
    datasets_with_missing: set[str] = set()
    dirs = [d for d in sorted(data_root.iterdir()) if d.is_dir()]
    if max_datasets:
        dirs = dirs[:max_datasets]

    summarize_task_types(dirs)

    from sklearn.utils.multiclass import type_of_target
    from sklearn.preprocessing import KBinsDiscretizer

    for d in dirs:
        try:
            # ds_start = time.time()
            info = load_dataset_info(d)
            task_type = None
            if info:
                task_type = str(info.get('task_type', '')).lower()
                if task_type == 'regression':
                    logging.info(f"跳过数据集 {d.name}: task_type=regression 已过滤，仅评测分类任务")
                    continue
                if task_type and task_type not in CLASSIFICATION_TASKS:
                    logging.info(f"跳过数据集 {d.name}: 未知 task_type={task_type} (仅支持 {CLASSIFICATION_TASKS})")
                    continue

            train_path, val_path, test_path = find_data_files(d)
            if train_path is None and test_path is None:
                logging.info(f"跳过：{d} (没有可识别的数据文件)")
                continue

            if train_path and test_path:
                X_train, y_train = load_table(train_path, context=f"{d.name}-train", coerce_numeric=coerce_numeric, dataset_id=d.name, missing_registry=datasets_with_missing)
                X_test, y_test = load_table(test_path, context=f"{d.name}-test", coerce_numeric=coerce_numeric, dataset_id=d.name, missing_registry=datasets_with_missing)
            else:
                # single = train_path
                # X, y = load_table(single, context=f"{d.name}-single")
                # X_train, X_test, y_train, y_test = split_train_test(X, y)
                # logging.info(f"数据集：{d} (没有 TRAIN/TEST 文件，自动按 80/20 划分)")
                # val_path = None
                logging.info(f"数据集：{d} (没有 TRAIN/TEST 文件, 跳过)")
                val_path = None
                continue

            if merge_val and val_path:
                X_val, y_val = load_table(val_path, context=f"{d.name}-val", coerce_numeric=coerce_numeric, dataset_id=d.name, missing_registry=datasets_with_missing)
                if X_val.ndim == 3 and X_val.shape[1] == 1:
                    X_val = X_val.squeeze(1)
                if X_val.ndim == 1:
                    X_val = X_val.reshape(-1, 1)
                y_val = np.asarray(y_val)
                if y_val.ndim > 1 and y_val.shape[-1] == 1:
                    y_val = y_val.reshape(-1)
                X_train = np.concatenate([X_train, X_val], axis=0)
                y_train = np.concatenate([y_train, y_val], axis=0)
                logging.info(f"{d.name}: 已将 validation split 合并进训练，总计 {X_train.shape[0]} 条训练样本")


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

            if task_type is None:
                if tgt_type is not None and tgt_type.startswith('continuous'):
                    if bins and bins > 1:
                        # discretize continuous labels into bins (fit on train)
                        est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
                        y_train = est.fit_transform(y_train.reshape(-1, 1)).astype(int).ravel()
                        # apply same transform to test
                        y_test = est.transform(y_test.reshape(-1, 1)).astype(int).ravel()
                        logging.info(f"{d.name}: converted continuous target to {bins} bins for classification")
                    elif skip_regression:
                        logging.info(f"跳过数据集 {d.name}: 连续标签 (可能为回归任务) 检测到，跳过")
                        continue
                    else:
                        logging.info(f"{d.name}: 连续标签检测到，但继续尝试拟合 (skip_regression=False)")
            else:
                if tgt_type and tgt_type.startswith('continuous'):
                    logging.debug(f"{d.name}: info.json 标注为分类任务，但 labels 检测为连续值，继续按分类任务处理")

            # fit and predict
            ds_start = time.time()
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
        missing_results = [(name, acc) for name, acc, _ in results if name in datasets_with_missing]
        missing_names = sorted(name for name, _ in missing_results)
        avg_missing_acc = sum(acc for _, acc in missing_results) / len(missing_results) if missing_results else None
        with open(outdir / 'talent_summary.txt', 'w') as f:
            f.write(f"Total datasets: {len(results)}\n")
            f.write(f"Average accuracy: {avg_acc:.6f}\n")
            f.write(f"Total time s: {total_time:.3f}\n")
            f.write(f"Average time s: {avg_time:.3f}\n")
            if missing_results:
                f.write(f"Datasets with NaN values: {len(missing_names)}\n")
                f.write(f"Average accuracy (NaN datasets): {avg_missing_acc:.6f}\n")
                f.write(f"List (NaN datasets): {', '.join(missing_names)}\n")
            else:
                f.write("Datasets with NaN values: 0\n")
        logging.info(f"评测完成，共 {len(results)} 个数据集，平均准确率 {avg_acc:.4f}, 总耗时 {total_time:.2f}s, 平均每数据集 {avg_time:.2f}s")
        if missing_results:
            logging.info("含 NaN/缺失值的数据集(%d): %s", len(missing_names), ', '.join(missing_names))
            logging.info("含 NaN/缺失值数据集平均准确率 %.4f", avg_missing_acc)
        else:
            logging.info("没有包含 NaN/缺失值的数据集。")
    else:
        logging.info("没有成功的评测结果。")


def main(argv=None):
    p = argparse.ArgumentParser(description='Benchmark TabICLClassifier on TALENT datasets')
    p.add_argument('--model-path', default='/data0/fangjuntao2025/tabicl-main/tabICLOrignCheckpoint/tabicl-classifier-v1.1-0506.ckpt', help='Path to TabICL checkpoint (model_path)')
    p.add_argument('--data-root', default='/data0/fangjuntao2025/TalentDatasetLast/data', help='Root path to TALENT data folder')
    p.add_argument('--outdir', default='evaluation_results', help='Directory to save results')
    p.add_argument('--max-datasets', type=int, default=None, help='Limit number of datasets to evaluate')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--merge-val', action='store_true', help='Include validation split into training when available')
    p.add_argument('--no-coerce-numeric', dest='coerce_numeric', action='store_false', help='Disable auto conversion of non-numeric features to numeric encodings')
    p.set_defaults(coerce_numeric=True)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG, format='[%(levelname)s] %(message)s')

    # Also write logs to a file inside outdir (keep console output)
    outdir_path = Path(args.outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(outdir_path / 'bench_talent.log')
    file_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(file_handler)

    evaluate_datasets(
        args.model_path,
        args.data_root,
        args.outdir,
        max_datasets=args.max_datasets,
        verbose=args.verbose,
        merge_val=args.merge_val,
        coerce_numeric=args.coerce_numeric,
    )


if __name__ == '__main__':
    main()
