from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

# Make sure local imports work even without editable install.
# This repo uses a src-layout (src/tabicl), so we need to add `.../src`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(_SRC_ROOT))

from tabicl import InferenceConfig, MantisICL  # noqa: E402
from tabicl.prior.data_reader import DataReader  # noqa: E402
from tabicl.sklearn.preprocessing import EnsembleGenerator, TransformToNumerical  # noqa: E402


def _softmax_np(x: np.ndarray, axis: int = -1, temperature: float = 0.9) -> np.ndarray:
    x = x / temperature
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _load_mantisicl_from_ckpt(ckpt_path: Path) -> Tuple[MantisICL, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

    # Some checkpoints in this repo have odd keys like 'config\n'
    key_map = {str(k).strip(): k for k in ckpt.keys()}
    if "config" not in key_map or "state_dict" not in key_map:
        raise ValueError(
            f"Checkpoint missing required keys. Found: {list(ckpt.keys())}. "
            "Expected 'config' and 'state_dict' (possibly with whitespace)."
        )

    config = ckpt[key_map["config"]]
    state_dict = ckpt[key_map["state_dict"]]

    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise ValueError("Checkpoint 'config' or 'state_dict' has unexpected type.")

    model = MantisICL(**config)
    model.load_state_dict(state_dict)
    model.eval()

    return model, config


@dataclass
class EvalConfig:
    uea_data_path: str
    ucr_data_path: str
    transform_ts_size: int
    model_path: Path
    device: str
    use_amp: bool
    verbose: bool

    n_estimators: int
    norm_methods: Optional[List[str]]
    feat_shuffle_method: str
    class_shift: bool
    outlier_threshold: float
    softmax_temperature: float
    average_logits: bool
    batch_size: int
    random_state: int

    save_dir: str


def _iter_datasets(reader: DataReader, only: Optional[List[str]]) -> Iterable[str]:
    if only:
        for name in only:
            yield name
    else:
        for name in reader.dataset_list_ucr:
            yield name


def _evaluate_one_dataset(
    dataset_name: str,
    reader: DataReader,
    model: MantisICL,
    device: torch.device,
    cfg: EvalConfig,
) -> float:
    X_train, y_train = reader.read_dataset(dataset_name, which_set="train")
    X_test, y_test = reader.read_dataset(dataset_name, which_set="test")

    if len(X_train.shape) == 3:
        X_train = X_train.squeeze(1)
    if len(X_test.shape) == 3:
        X_test = X_test.squeeze(1)

    # Encode labels to 0..C-1
    y_enc = LabelEncoder()
    y_train_enc = y_enc.fit_transform(y_train)
    y_test_enc = y_enc.transform(y_test)

    # Feature transform (no-op for numpy arrays; kept for parity with sklearn pipeline)
    X_encoder = TransformToNumerical(verbose=cfg.verbose)
    X_train_num = X_encoder.fit_transform(X_train)
    X_test_num = X_encoder.transform(X_test)

    # Build ensemble prompts: concat(train, test) and generate feature shuffles + class shifts
    ensemble = EnsembleGenerator(
        n_estimators=cfg.n_estimators,
        norm_methods=cfg.norm_methods,
        feat_shuffle_method=cfg.feat_shuffle_method,
        class_shift=cfg.class_shift,
        outlier_threshold=cfg.outlier_threshold,
        random_state=cfg.random_state,
    )
    ensemble.fit(X_train_num, y_train_enc)

    data = ensemble.transform(X_test_num)

    # Inference config
    inference_config = InferenceConfig()
    inference_config.update_from_dict(
        {
            "COL_CONFIG": {"device": device, "use_amp": cfg.use_amp, "verbose": cfg.verbose},
            "ROW_CONFIG": {"device": device, "use_amp": cfg.use_amp, "verbose": cfg.verbose},
            "ICL_CONFIG": {"device": device, "use_amp": cfg.use_amp, "verbose": cfg.verbose},
        }
    )

    # Forward all ensemble members (possibly grouped by norm method)
    outputs_all = []
    class_shift_offsets: List[int] = []

    for norm_method, (Xs, ys) in data.items():
        shuffle_patterns = ensemble.feature_shuffle_patterns_[norm_method]
        offsets = ensemble.class_shift_offsets_[norm_method]
        class_shift_offsets.extend(offsets)

        # Batched forward over ensemble members
        bs = max(1, int(cfg.batch_size))
        n_members = Xs.shape[0]
        for start in range(0, n_members, bs):
            end = min(start + bs, n_members)
            X_batch = torch.from_numpy(Xs[start:end]).float().to(device)
            y_batch = torch.from_numpy(ys[start:end]).float().to(device)
            pattern_batch = shuffle_patterns[start:end]

            with torch.no_grad():
                out = model(
                    X_batch,
                    y_batch,
                    feature_shuffles=pattern_batch,
                    return_logits=True if cfg.average_logits else False,
                    softmax_temperature=cfg.softmax_temperature,
                    inference_config=inference_config,
                )
            outputs_all.append(out.float().cpu().numpy())

    outputs = np.concatenate(outputs_all, axis=0)

    # Aggregate predictions and undo class shifts (same logic as MantisICLClassifier)
    n_estimators_actual = len(class_shift_offsets)
    avg = None
    for i, offset in enumerate(class_shift_offsets):
        out = outputs[i]
        out = np.concatenate([out[..., offset:], out[..., :offset]], axis=-1)
        avg = out if avg is None else (avg + out)

    avg = avg / max(1, n_estimators_actual)

    if cfg.average_logits:
        probs = _softmax_np(avg, axis=-1, temperature=cfg.softmax_temperature)
    else:
        probs = avg

    y_pred = np.argmax(probs, axis=1)
    acc = float(np.mean(y_pred == y_test_enc))
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate UCR datasets using MantisICL (no MantisICLClassifier).")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data0/fangjuntao2025/tabicl-main/checkpoints/mantisICL_mixup_run_v1/step-18800.ckpt",
        help="Path to a checkpoint containing config + state_dict for MantisICL.",
    )
    parser.add_argument(
        "--uea-data-path",
        type=str,
        default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/",
    )
    parser.add_argument(
        "--ucr-data-path",
        type=str,
        default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/",
    )
    parser.add_argument("--transform-ts-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names (default: all UCR).")

    # Ensemble / preprocessing knobs (kept consistent with sklearn wrapper defaults)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument(
        "--norm-methods",
        type=str,
        default=None,
        help="Comma-separated norm methods, e.g. 'none,power'. Default: same as wrapper (None -> ['none','power']).",
    )
    parser.add_argument("--feat-shuffle-method", type=str, default="latin")
    parser.add_argument("--class-shift", action="store_true")
    parser.add_argument("--no-class-shift", dest="class_shift", action="store_false")
    parser.set_defaults(class_shift=True)
    parser.add_argument("--outlier-threshold", type=float, default=4.0)
    parser.add_argument("--softmax-temperature", type=float, default=0.9)
    parser.add_argument("--average-logits", action="store_true")
    parser.add_argument("--average-probs", dest="average_logits", action="store_false")
    parser.set_defaults(average_logits=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--save-dir", type=str, default="evaluation_results")

    args = parser.parse_args()

    dataset_list = None
    if args.datasets:
        dataset_list = [x.strip() for x in args.datasets.split(",") if x.strip()]

    norm_methods = None
    if args.norm_methods is not None:
        norm_methods = [x.strip() for x in args.norm_methods.split(",") if x.strip()]

    cfg = EvalConfig(
        uea_data_path=args.uea_data_path,
        ucr_data_path=args.ucr_data_path,
        transform_ts_size=args.transform_ts_size,
        model_path=Path(args.model_path),
        device=args.device,
        use_amp=bool(args.use_amp),
        verbose=bool(args.verbose),
        n_estimators=int(args.n_estimators),
        norm_methods=norm_methods,
        feat_shuffle_method=str(args.feat_shuffle_method),
        class_shift=bool(args.class_shift),
        outlier_threshold=float(args.outlier_threshold),
        softmax_temperature=float(args.softmax_temperature),
        average_logits=bool(args.average_logits),
        batch_size=int(args.batch_size),
        random_state=int(args.random_state),
        save_dir=str(args.save_dir),
    )

    reader = DataReader(
        UEA_data_path=cfg.uea_data_path,
        UCR_data_path=cfg.ucr_data_path,
        transform_ts_size=cfg.transform_ts_size,
    )

    model, model_config = _load_mantisicl_from_ckpt(cfg.model_path)
    device = torch.device(cfg.device)
    model.to(device)

    results: List[Tuple[str, float]] = []
    total = 0.0
    count = 0

    for dataset_name in _iter_datasets(reader, dataset_list):
        try:
            acc = _evaluate_one_dataset(dataset_name, reader, model, device, cfg)
            results.append((dataset_name, acc))
            total += acc
            count += 1
            print(f"{dataset_name}: {acc:.4f}")
        except Exception as e:
            print(f"{dataset_name}: FAILED ({e})")

    avg = total / count if count else 0.0
    print("\nSummary")
    print("-----------------------------")
    print(f"Total datasets evaluated: {count}")
    print(f"Average accuracy: {avg:.4f}")
    print("-----------------------------")

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    detailed_path = save_dir / f"mantisicl_ucr_detailed_{stamp}.txt"
    summary_path = save_dir / f"mantisicl_ucr_summary_{stamp}.txt"
    json_path = save_dir / f"mantisicl_ucr_results_{stamp}.json"

    with open(detailed_path, "w") as f:
        for name, acc in results:
            f.write(f"{name}: {acc:.6f}\n")

    with open(summary_path, "w") as f:
        f.write(f"Total datasets: {count}\n")
        f.write(f"Average accuracy: {avg:.6f}\n")

    with open(json_path, "w") as f:
        json.dump(
            {
                "avg_accuracy": avg,
                "count": count,
                "results": [{"dataset": n, "accuracy": a} for n, a in results],
                "model_path": str(cfg.model_path),
                "model_config": model_config,
                "eval_config": {
                    "device": cfg.device,
                    "use_amp": cfg.use_amp,
                    "n_estimators": cfg.n_estimators,
                    "norm_methods": cfg.norm_methods,
                    "feat_shuffle_method": cfg.feat_shuffle_method,
                    "class_shift": cfg.class_shift,
                    "outlier_threshold": cfg.outlier_threshold,
                    "softmax_temperature": cfg.softmax_temperature,
                    "average_logits": cfg.average_logits,
                    "batch_size": cfg.batch_size,
                    "random_state": cfg.random_state,
                    "transform_ts_size": cfg.transform_ts_size,
                },
            },
            f,
            indent=2,
        )

    print(f"\nSaved:\n- {detailed_path}\n- {summary_path}\n- {json_path}")


if __name__ == "__main__":
    main()
