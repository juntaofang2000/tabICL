import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from tabicl.prior.data_reader import DataReader
from tabicl.sklearn.classifier import TabICLClassifier

DEFAULT_UEA_PATH = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/"
DEFAULT_UCR_PATH = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/"
DEFAULT_RESULTS_DIR = "evaluation_results"
DEFAULT_TABICL_CKPT = "/data0/fangjuntao2025/tabicl-main/tabICLOrignCheckpoint/tabicl-classifier-v1.1-0506.ckpt"
DEFAULT_MANTIS_CKPT = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/Models/Mantis/Mantis_cheickpoint/"
DEFAULT_NORM_METHODS = ["none", "robust"]


def prepare_feature_matrix(array: np.ndarray) -> np.ndarray:
    """Ensure each sample is represented as a 2D matrix (n_samples, n_features)."""
    array = np.asarray(array)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    return array.reshape(array.shape[0], -1)


def majority_vote(channel_predictions: Sequence[np.ndarray]) -> np.ndarray:
    """Aggregate per-channel predictions via majority vote."""
    if not channel_predictions:
        raise ValueError("channel_predictions cannot be empty")
    stacked = np.stack(channel_predictions, axis=0)  # (num_channels, num_samples)
    stacked = stacked.T  # (num_samples, num_channels)
    votes = []
    for sample_votes in stacked:
        labels, counts = np.unique(sample_votes, return_counts=True)
        votes.append(labels[np.argmax(counts)])
    return np.asarray(votes)


class UEAChannelVotingEvaluator:
    def __init__(
        self,
        uea_path: str,
        ucr_path: str,
        tabicl_ckpt: str,
        use_mantis: bool,
        mantis_ckpt: Optional[str],
        mantis_batch_size: int,
        norm_methods: Optional[Sequence[str]],
        seq_len: int,
        log_processing: bool,
        results_dir: str | Path,
    ) -> None:
        self.reader = DataReader(
            UEA_data_path=uea_path,
            UCR_data_path=ucr_path,
            transform_ts_size=seq_len,
            log_processing=log_processing,
        )
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        clf_kwargs = dict(
            verbose=False,
            n_estimators=32,
            checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
            model_path=tabicl_ckpt,
        )
        if norm_methods is not None:
            clf_kwargs["norm_methods"] = list(norm_methods)
        if use_mantis:
            clf_kwargs.update(mantis_checkpoint=mantis_ckpt, mantis_batch_size=mantis_batch_size)
        self.classifier = TabICLClassifier(**clf_kwargs)

    def evaluate_dataset(self, dataset_name: str) -> Tuple[float, int]:
        X_train, y_train = self.reader.read_dataset(dataset_name, which_set="train")
        X_test, y_test = self.reader.read_dataset(dataset_name, which_set="test")
        if X_train.ndim < 3:
            raise ValueError(f"Dataset {dataset_name} is not multi-channel (shape={X_train.shape}).")
        num_channels = X_train.shape[1]
        predictions: List[np.ndarray] = []
        for ch in range(num_channels):
            train_ch = prepare_feature_matrix(X_train[:, ch, :])
            test_ch = prepare_feature_matrix(X_test[:, ch, :])
            train_ch =  torch.tensor(train_ch, dtype=torch.float).unsqueeze(1)
            test_ch=  torch.tensor(test_ch, dtype=torch.float).unsqueeze(1)
            clf = self.classifier
            clf.fit(train_ch, y_train)
            preds = clf.predict(test_ch)
            predictions.append(preds)
        final_preds = majority_vote(predictions)
        accuracy = float(np.mean(final_preds == y_test))
        return accuracy, num_channels

    def run(self, datasets: Sequence[str]) -> List[Tuple[str, float, int]]:
        outcomes: List[Tuple[str, float, int]] = []
        for name in datasets:
            print(f"\n===== Evaluating {name} ({len(outcomes)+1}/{len(datasets)}) =====")
            try:
                acc, num_channels = self.evaluate_dataset(name)
                print(f"{name}: accuracy={acc:.4f} (channels={num_channels})")
                outcomes.append((name, acc, num_channels))
            except Exception as exc:
                print(f"[FAIL] {name}: {exc}")
        return outcomes

    def save(self, results: Sequence[Tuple[str, float, int]]) -> None:
        if not results:
            return
        detail_path = self.results_dir / "uea_channel_vote_detailed.txt"
        summary_path = self.results_dir / "uea_channel_vote_summary.txt"
        with open(detail_path, "w", encoding="utf-8") as handle:
            for name, acc, chans in results:
                handle.write(f"{name}: {acc:.6f} (channels={chans})\n")
        avg_acc = sum(acc for _, acc, _ in results) / len(results)
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write(f"Total datasets: {len(results)}\n")
            handle.write(f"Average accuracy: {avg_acc:.6f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UEA benchmark evaluation with per-channel TabICL classification and majority voting.",
    )
    parser.add_argument("--uea-path", default=DEFAULT_UEA_PATH, help="UEA dataset root directory")
    parser.add_argument("--ucr-path", default=DEFAULT_UCR_PATH, help="UCR dataset root directory (unused)")
    parser.add_argument("--tabicl-ckpt", default=DEFAULT_TABICL_CKPT, help="Path to TabICL checkpoint")
    parser.add_argument(
        "--mantis-ckpt",
        default=DEFAULT_MANTIS_CKPT,
        help="Path to Mantis checkpoint (used when --use-mantis)",
    )
    parser.add_argument(
        "--use-mantis",
        dest="use_mantis",
        action="store_true",
        help="Enable Mantis encoder inside TabICL (default)",
    )
    parser.add_argument(
        "--no-use-mantis",
        dest="use_mantis",
        action="store_false",
        help="Disable the Mantis encoder stage",
    )
    parser.add_argument("--mantis-batch-size", type=int, default=64, help="Batch size for Mantis encoder")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length fed to DataReader")
    parser.add_argument("--log-processing", action="store_true", help="Print DataReader preprocessing info")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Directory to store outputs")
    parser.add_argument("--datasets", nargs="*", help="Optional list of dataset names to evaluate")
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of datasets evaluated (useful for quick smoke tests)",
    )
    parser.add_argument(
        "--default-norm",
        dest="use_default_norm",
        action="store_true",
        help="Use TabICL's built-in normalization (power transform, etc.)",
    )
    parser.add_argument(
        "--robust-norm",
        dest="use_default_norm",
        action="store_false",
        help='Use norm_methods=["none","robust"] (default)',
    )
    parser.set_defaults(use_default_norm=False)
    parser.set_defaults(use_mantis=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    norm_methods = None if args.use_default_norm else DEFAULT_NORM_METHODS.copy()
    evaluator = UEAChannelVotingEvaluator(
        uea_path=args.uea_path,
        ucr_path=args.ucr_path,
        tabicl_ckpt=args.tabicl_ckpt,
        use_mantis=args.use_mantis,
        mantis_ckpt=args.mantis_ckpt,
        mantis_batch_size=args.mantis_batch_size,
        norm_methods=norm_methods,
        seq_len=args.seq_len,
        log_processing=args.log_processing,
        results_dir=args.results_dir,
    )
    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = sorted(evaluator.reader.dataset_list_uea)
    if args.limit:
        dataset_names = dataset_names[: args.limit]
    results = evaluator.run(dataset_names)
    evaluator.save(results)


if __name__ == "__main__":
    main()
