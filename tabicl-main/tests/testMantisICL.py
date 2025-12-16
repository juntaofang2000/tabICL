import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from tabicl.model.mantisICL import MantisICL
from tabicl.prior.data_reader import DataReader


DEFAULT_CHECKPOINT = (
    "/data0/fangjuntao2025/tabicl-main/checkpoints/mantisICL_mixup_run_v1/step-8000.ckpt"
)
DEFAULT_UEA_PATH = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/"
DEFAULT_UCR_PATH = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/"


def _prepare_timeseries(array: np.ndarray) -> np.ndarray:
    """Flatten multi-channel series so MantisICL receives (N, seq_len)."""
    if array.ndim == 3:
        if array.shape[1] == 1:
            return array[:, 0, :]
        return array.reshape(array.shape[0], -1)
    return array


class BenchmarkEvaluator:
    def __init__(self, model: MantisICL, reader: DataReader, device: torch.device) -> None:
        self.model = model
        self.reader = reader
        self.device = device

    def _load_dataset(self, name: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        X_train, y_train = self.reader.read_dataset(name, which_set="train")
        X_test, y_test = self.reader.read_dataset(name, which_set="test")

        X_train = torch.from_numpy(_prepare_timeseries(X_train)).float().to(self.device)
        y_train = torch.from_numpy(y_train).long().to(self.device)
        X_test = torch.from_numpy(_prepare_timeseries(X_test)).float().to(self.device)
        y_test = torch.from_numpy(y_test).long().to(self.device)
        return X_train, y_train, X_test, y_test

    @torch.no_grad()
    def evaluate(self, dataset_names: Sequence[str]) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        self.model.eval()
        for name in dataset_names:
            try:
                X_train, y_train, X_test, y_test = self._load_dataset(name)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[WARN] Failed to load {name}: {exc}")
                continue

            X_all = torch.cat([X_train, X_test], dim=0).unsqueeze(0)
            ctx_y = y_train.unsqueeze(0)
            try:
                logits = self.model(
                    X_all,
                    ctx_y,
                    embed_with_test=False,
                    return_logits=True,
                )
            except torch.cuda.OutOfMemoryError:
                print(f"[OOM] Skipping {name} during forward pass.")
                torch.cuda.empty_cache()
                continue

            preds = logits.argmax(dim=-1).squeeze(0)
            acc = (preds == y_test).float().mean().item()
            results.append((name, acc))
            print(f"{name}: accuracy={acc:.4f}")

        return results

    @staticmethod
    def summarize(collection_name: str, results: Sequence[Tuple[str, float]]) -> None:
        if not results:
            print(f"No successful evaluations for {collection_name}.")
            return
        avg_acc = sum(acc for _, acc in results) / len(results)
        print(
            f"\n{collection_name} summary: evaluated {len(results)} datasets, average accuracy = {avg_acc:.4f}"
        )


def load_mantisicl(checkpoint_path: Path, device: torch.device, args: argparse.Namespace) -> MantisICL:
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get(
        "config",
        dict(
            max_classes=args.max_classes,
            icl_num_blocks=args.icl_num_blocks,
            icl_nhead=args.icl_nhead,
            ff_factor=args.ff_factor,
            dropout=args.dropout,
            activation=args.activation,
            norm_first=args.norm_first,
            train_mantis=args.train_mantis,
        ),
    )

    model = MantisICL(**config)
    model.to(device)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing {len(missing)} keys when loading checkpoint: {missing[:5]}")
    if unexpected:
        print(f"[WARN] Unexpected keys in checkpoint: {unexpected[:5]}")
    return model


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a fine-tuned MantisICL checkpoint on UCR and UEA datasets."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to MantisICL checkpoint")
    parser.add_argument("--uea-path", default=DEFAULT_UEA_PATH, help="UEA dataset root directory")
    parser.add_argument("--ucr-path", default=DEFAULT_UCR_PATH, help="UCR dataset root directory")
    parser.add_argument("--device", default=None, help="Torch device to use (default: auto)")
    parser.add_argument("--max-uea", type=int, default=None, help="Limit number of UEA datasets")
    parser.add_argument("--max-ucr", type=int, default=None, help="Limit number of UCR datasets")
    parser.add_argument("--train-mantis", action="store_true", help="Instantiate model with trainable Mantis")
    parser.add_argument("--max-classes", type=int, default=60)
    parser.add_argument("--icl-num-blocks", type=int, default=12)
    parser.add_argument("--icl-nhead", type=int, default=4)
    parser.add_argument("--ff-factor", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", default="gelu")
    parser.add_argument("--norm-first", action="store_true", default=True)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model = load_mantisicl(checkpoint_path, device, args)
    reader = DataReader(
        UEA_data_path=args.uea_path,
        UCR_data_path=args.ucr_path,
        transform_ts_size=512,
    )
    evaluator = BenchmarkEvaluator(model, reader, device)

    uea_names: List[str] = sorted(reader.dataset_list_uea)
    ucr_names: List[str] = sorted(reader.dataset_list_ucr)
    if args.max_uea is not None:
        uea_names = uea_names[: args.max_uea]
    if args.max_ucr is not None:
        ucr_names = ucr_names[: args.max_ucr]

    print("\n===== Evaluating UCR datasets =====")
    ucr_results = evaluator.evaluate(ucr_names)
    evaluator.summarize("UCR", ucr_results)

    print("\n===== Evaluating UEA datasets =====")
    uea_results = evaluator.evaluate(uea_names)
    evaluator.summarize("UEA", uea_results)




if __name__ == "__main__":
    main()
