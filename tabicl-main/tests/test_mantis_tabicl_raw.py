import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from tabicl.model.mantis_tabicl import MantisTabICL, encode_with_mantis
from tabicl.prior.data_reader import DataReader

DEFAULT_TABICL_CHECKPOINT = "/data0/fangjuntao2025/tabicl-main/ft_checkpoints/mantis_tabicl_ft_epoch1.ckpt"
DEFAULT_MANTIS_CHECKPOINT = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/Models/Mantis/Mantis_cheickpoint/"
DEFAULT_UEA_PATH = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/"
DEFAULT_UCR_PATH = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/"
DEFAULT_REPR_CACHE = "representation_cache"


def _ensure_three_dim(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        return arr[None, None, :]
    if arr.ndim == 2:
        return arr[:, None, :]
    return arr


def _sanitize(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _maybe_load(cache_dir: Optional[Path], dataset: str) -> Optional[Tensor]:
    if cache_dir is None:
        return None
    cache_file = cache_dir / f"{_sanitize(dataset)}.pt"
    if not cache_file.is_file():
        return None
    return torch.load(cache_file, map_location="cpu")


def _maybe_save(cache_dir: Optional[Path], dataset: str, tensor: Tensor) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{_sanitize(dataset)}.pt"
    torch.save(tensor.cpu(), cache_file)


class MantisTabICLRawEvaluator:
    def __init__(
        self,
        model: MantisTabICL,
        reader: DataReader,
        device: torch.device,
        embed_with_test: bool,
        cache_dir: Optional[Path],
        skip_if_too_many_classes: bool,
        max_classes: Optional[int],
    ) -> None:
        self.model = model
        self.reader = reader
        self.device = device
        self.embed_with_test = embed_with_test
        self.cache_dir = cache_dir
        self.skip_if_too_many_classes = skip_if_too_many_classes
        self.class_limit = max_classes

        self.model.eval()
        if hasattr(self.model, "mantis_model"):
            self.model.mantis_model.eval()
        self.model.tabicl_model.eval()

    def _load_dataset(self, name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train, y_train = self.reader.read_dataset(name, which_set="train")
        X_test, y_test = self.reader.read_dataset(name, which_set="test")
        return X_train, y_train, X_test, y_test

    def _encode_table(self, dataset: str, full_series: np.ndarray) -> Tensor:
        cached = _maybe_load(self.cache_dir, dataset)
        if cached is not None:
            return cached.float().to(self.device)

        series = np.asarray(full_series, dtype=np.float32)
        if series.ndim != 3:
            raise ValueError(f"Expected (rows, channels, length) array, got shape {series.shape}")

        mantis_repr = encode_with_mantis(
            self.model.mantis_model,
            series,
            device=self.device,
            batch_size=self.model.mantis_batch_size,
        )

        tensor = torch.from_numpy(mantis_repr).float().to(self.device)
        _maybe_save(self.cache_dir, dataset, tensor)
        return tensor

    def _forward(self, table_repr: Tensor, ctx_labels: Tensor) -> Tensor:
        table = table_repr.unsqueeze(0)
        ctx = ctx_labels.unsqueeze(0)
        logits = self.model.tabicl_model(
            table,
            ctx,
            embed_with_test=self.embed_with_test,
            return_logits=True,
        )
        return logits

    @torch.no_grad()
    def evaluate_dataset(self, name: str) -> Optional[Tuple[str, float]]:
        try:
            X_train, y_train, X_test, y_test = self._load_dataset(name)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[FAIL] {name}: unable to load dataset ({exc})")
            return None

        if X_train.size == 0 or X_test.size == 0:
            print(f"[SKIP] {name}: empty split")
            return None

        train_arr = _ensure_three_dim(X_train)
        test_arr = _ensure_three_dim(X_test)
        try:
            full_series = np.concatenate([train_arr, test_arr], axis=0)
        except ValueError as exc:
            print(f"[FAIL] {name}: cannot concatenate train/test arrays ({exc})")
            return None

        ctx_labels = torch.from_numpy(y_train).long().to(self.device)
        test_labels = torch.from_numpy(y_test).long().to(self.device)
        total_rows = full_series.shape[0]
        if total_rows != ctx_labels.numel() + test_labels.numel():
            print(
                f"[WARN] {name}: row count mismatch (rows={total_rows}, labels={ctx_labels.numel() + test_labels.numel()})"
            )

        y_all = np.concatenate([y_train, y_test])
        class_count = int(np.unique(y_all).size)
        class_limit = self.class_limit or getattr(self.model.tabicl_model, "max_classes", None)
        if self.skip_if_too_many_classes and class_limit is not None and class_count > class_limit:
            print(f"[SKIP] {name}: {class_count} classes exceed limit {class_limit}")
            return None

        try:
            table_repr = self._encode_table(name, full_series)
            logits = self._forward(table_repr, ctx_labels)
            preds = logits.argmax(dim=-1).squeeze(0)
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] {name}: CUDA out of memory, skipping")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[FAIL] {name}: {exc}")
            return None

        expected = min(preds.numel(), test_labels.numel())
        if preds.numel() != test_labels.numel():
            print(
                f"[WARN] {name}: predictions ({preds.numel()}) != labels ({test_labels.numel()}); truncating to {expected}"
            )
        acc = (preds[:expected] == test_labels[:expected]).float().mean().item()
        print(f"{name}: accuracy={acc:.4f}")
        return name, acc

    def evaluate_collection(self, datasets: Sequence[str]) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        for name in datasets:
            outcome = self.evaluate_dataset(name)
            if outcome is not None:
                results.append(outcome)
        return results

    @staticmethod
    def summarize(label: str, results: Sequence[Tuple[str, float]]) -> None:
        if not results:
            print(f"No successful evaluations for {label}.")
            return
        avg = sum(acc for _, acc in results) / len(results)
        print(f"\n{label}: evaluated {len(results)} datasets, average accuracy={avg:.4f}")


def load_mantis_tabicl(args: argparse.Namespace, device: torch.device) -> MantisTabICL:
    model = MantisTabICL(
        tabicl_checkpoint=args.tabicl_ckpt,
        mantis_checkpoint=args.mantis_ckpt,
        mantis_batch_size=args.mantis_batch_size,
        device=device,
    )
    model.to(device)

    if args.composite_ckpt:
        ckpt_path = Path(args.composite_ckpt)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Composite checkpoint not found at {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("mantis_tabicl_state_dict")
        if state_dict is None:
            raise KeyError(
                f"Composite checkpoint {ckpt_path} is missing 'mantis_tabicl_state_dict'."
            )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys when loading composite checkpoint: {missing[:5]}")
        if unexpected:
            print(f"[WARN] Unexpected keys in composite checkpoint: {unexpected[:5]}")

    return model


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate MantisTabICL directly on UCR/UEA datasets using raw Mantis embeddings.",
    )
    parser.add_argument("--tabicl-ckpt", default=DEFAULT_TABICL_CHECKPOINT, help="Path to TabICL checkpoint")
    parser.add_argument("--mantis-ckpt", default=DEFAULT_MANTIS_CHECKPOINT, help="Path to Mantis checkpoint")
    parser.add_argument("--composite-ckpt", default=None, help="Optional checkpoint containing mantis_tabicl_state_dict")
    parser.add_argument("--uea-path", default=DEFAULT_UEA_PATH, help="UEA dataset root directory")
    parser.add_argument("--ucr-path", default=DEFAULT_UCR_PATH, help="UCR dataset root directory")
    parser.add_argument("--device", default=None, help="Torch device to use (default: auto)")
    parser.add_argument("--mantis-batch-size", type=int, default=64, help="Mini-batch size for Mantis encoder")
    parser.add_argument("--transform-ts-size", type=int, default=512, help="Sequence length supplied to DataReader")
    parser.add_argument("--embed-with-test", action="store_true", help="Allow CLS tokens to attend to test rows")
    parser.add_argument("--repr-cache", default=None, help="Directory to cache Mantis representations")
    parser.add_argument("--max-classes", type=int, default=None, help="Override TabICL max_classes for skipping")
    parser.add_argument("--skip-too-many-classes", action="store_true", help="Skip datasets exceeding max_classes")
    parser.add_argument("--collections", nargs="+", choices=["ucr", "uea"], default=["ucr", "uea"], help="Dataset groups to evaluate")
    parser.add_argument("--datasets", nargs="+", help="Explicit dataset names (overrides --collections)")
    parser.add_argument("--max-uea", type=int, default=None, help="Limit number of UEA datasets")
    parser.add_argument("--max-ucr", type=int, default=None, help="Limit number of UCR datasets")
    return parser


def build_dataset_lists(reader: DataReader, args: argparse.Namespace) -> Dict[str, List[str]]:
    if args.datasets:
        return {"custom": list(dict.fromkeys(args.datasets))}

    collections: Dict[str, List[str]] = {}
    if "ucr" in args.collections:
        names = sorted(reader.dataset_list_ucr)
        if args.max_ucr is not None:
            names = names[: args.max_ucr]
        collections["UCR"] = names
    if "uea" in args.collections:
        names = sorted(reader.dataset_list_uea)
        if args.max_uea is not None:
            names = names[: args.max_uea]
        collections["UEA"] = names
    return collections


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_mantis_tabicl(args, device)

    reader = DataReader(
        UEA_data_path=args.uea_path,
        UCR_data_path=args.ucr_path,
        transform_ts_size=args.transform_ts_size,
    )

    cache_dir = Path(args.repr_cache) if args.repr_cache else None

    evaluator = MantisTabICLRawEvaluator(
        model=model,
        reader=reader,
        device=device,
        embed_with_test=args.embed_with_test,
        cache_dir=cache_dir,
        skip_if_too_many_classes=args.skip_too_many_classes,
        max_classes=args.max_classes,
    )

    dataset_groups = build_dataset_lists(reader, args)

    for label, names in dataset_groups.items():
        print(f"\n===== Evaluating {label} datasets ({len(names)} total) =====")
        results = evaluator.evaluate_collection(names)
        evaluator.summarize(label, results)


if __name__ == "__main__":
    main()
