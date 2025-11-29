import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tabicl.prior.data_reader import DataReader
from tabicl.sklearn.classifier import TabICLClassifier
from tabicl.model.mantis_dev.architecture.architecture import Mantis8MWithFDDM
from tabicl.model.mantis_tabicl import _load_generic_state_dict

DEFAULT_UEA_PATH = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/"
DEFAULT_UCR_PATH = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/"
DEFAULT_RESULTS_DIR = "evaluation_results"
DEFAULT_TABICL_CKPT = "/data0/fangjuntao2025/tabicl-main/tabICLOrignCheckpoint/tabicl-classifier-v1.1-0506.ckpt"
DEFAULT_MANTIS_FDDM_CKPT = "/data0/fangjuntao2025/CauKer/CauKerOrign/checkpoint/mantis8m_fddm20251128_epoch040.pt"
DEFAULT_NORM_METHODS = ["none", "robust"]

# SKIP_UEA_DATASETS = {
#     "AtrialFibrillation",
#     "BasicMotions",
#     "CricSket",
#     "ArticularyWordRecognition",
#     "ERing",
#     "CharacterTrajectories",
#     "EigenWorms",
#     "Epilepsy",
#     "EthanolConcentration",
#     "FingerMovements",
#     "HandMovementDirection",
#     "Handwriting",
#     "Heartbeat",
#     "JapaneseVowels",
#     "LSST",
#     "Libras",
#     "NATOPS",
#     "MotorImagery",
#     "PenDigits",
#     "RacketSports",
#     "SelfRegulationSCP1",
#     "SelfRegulationSCP2",
#     "StandWalkJump",
#     "UWaveGestureLibrary",
#     "PhonemeSpectra",
# }



SKIP_UEA_DATASETS = {
    "AtrialFibrillation",
    "BasicMotions",
    "Cricket",
    "ArticularyWordRecognition",
    "ERing",
    "CharacterTrajectories",
    "EigenWorms",
    "Epilepsy",
    "EthanolConcentration",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "JapaneseVowels",
    "LSST",
    "Libras",
    "NATOPS",
    "MotorImagery",
    "PenDigits",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "StandWalkJump",
    "UWaveGestureLibrary",
    "PhonemeSpectra",
}
def load_dataset_names_from_file(filepath: str | Path) -> List[str]:
    names: List[str] = []
    with open(filepath, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                line = line.split(":", maxsplit=1)[0].strip()
            names.append(line)
    return names


def split_by_collection(names: Sequence[str], reader: DataReader) -> Dict[str, List[str]]:
    uea = []
    ucr = []
    unknown = []
    uea_set = set(reader.dataset_list_uea)
    ucr_set = set(reader.dataset_list_ucr)
    for name in names:
        if name in uea_set:
            uea.append(name)
        elif name in ucr_set:
            ucr.append(name)
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(f"Unknown dataset names: {unknown}")
    return {"uea": uea, "ucr": ucr}


def build_mantis_fddm_encoder(
    checkpoint: str | Path,
    device: torch.device,
    seq_len: int,
    hidden_dim: int,
    num_patches: int,
    num_channels: int,
    fddm_output_dim: int,
    transf_depth: int = 6,
    transf_num_heads: int = 8,
    transf_mlp_dim: int = 512,
    transf_dim_head: int = 128,
    transf_dropout: float = 0.1,
) -> Mantis8MWithFDDM:
    model = Mantis8MWithFDDM(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_patches=num_patches,
        num_channels=num_channels,
        transf_depth=transf_depth,
        transf_num_heads=transf_num_heads,
        transf_mlp_dim=transf_mlp_dim,
        transf_dim_head=transf_dim_head,
        transf_dropout=transf_dropout,
        fddm_output_dim=fddm_output_dim,
        device=str(device),
        pre_training=False,
    )

    ckpt_path = Path(checkpoint)
    if ckpt_path.exists():
        if ckpt_path.is_dir():
            model = model.from_pretrained(str(ckpt_path))
        else:
            state = torch.load(ckpt_path, map_location="cpu")
            _load_generic_state_dict(model, state, strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist")

    model.to(device)
    model.eval()
    return model


class SequenceEncoder:
    def __init__(
        self,
        model: Mantis8MWithFDDM,
        device: torch.device,
        seq_len: int,
        num_channels: int,
        batch_size: int,
    ) -> None:
        self.model = model
        self.device = device
        self.seq_len = seq_len
        self.num_channels = num_channels
        self.batch_size = max(1, batch_size)

    @torch.no_grad()
    def encode(self, array: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {tensor.shape}")
        if tensor.size(1) != self.num_channels:
            if tensor.size(1) == 1 and self.num_channels > 1:
                tensor = tensor.repeat(1, self.num_channels, 1)
            else:
                raise ValueError(
                    f"Channel mismatch: expected {self.num_channels}, received {tensor.size(1)}"
                )
        if tensor.size(-1) != self.seq_len:
            tensor = F.interpolate(
                tensor,
                size=self.seq_len,
                mode="linear",
                align_corners=False,
            )

        outputs: List[torch.Tensor] = []
        for start in range(0, tensor.size(0), self.batch_size):
            end = start + self.batch_size
            batch = tensor[start:end].to(self.device)
            reps = self.model(batch)
            outputs.append(reps.detach().cpu())
        return torch.cat(outputs, dim=0).numpy()


class MantisFDDMTabICLEvaluator:
    def __init__(
        self,
        uea_path: str,
        ucr_path: str,
        tabicl_ckpt: str,
        mantis_fddm_ckpt: str,
        seq_len: int = 512,
        num_channels: int = 1,
        hidden_dim: int = 256,
        num_patches: int = 32,
        fddm_output_dim: int = 256,
        mantis_batch_size: int = 64,
        tabicl_batch_size: int = 8,
        tabicl_n_estimators: int = 32,
        norm_methods: Optional[List[str]] = None,
        results_dir: str | Path = DEFAULT_RESULTS_DIR,
        device: Optional[str] = None,
        skip_uea: Optional[Iterable[str]] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.reader = DataReader(
            UEA_data_path=uea_path,
            UCR_data_path=ucr_path,
            transform_ts_size=seq_len,
            log_processing=True,
        )
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.skip_uea = set(skip_uea or [])

        mantis_model = build_mantis_fddm_encoder(
            mantis_fddm_ckpt,
            device=self.device,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_patches=num_patches,
            num_channels=num_channels,
            fddm_output_dim=fddm_output_dim,
        )

        self.encoder = SequenceEncoder(
            mantis_model,
            device=self.device,
            seq_len=seq_len,
            num_channels=num_channels,
            batch_size=mantis_batch_size,
        )

        clf_kwargs = dict(
            verbose=False,
            n_estimators=tabicl_n_estimators,
            checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
            model_path=tabicl_ckpt,
            batch_size=tabicl_batch_size,
            device=self.device,
        )
        if norm_methods is not None:
            clf_kwargs["norm_methods"] = norm_methods
        self.clf = TabICLClassifier(**clf_kwargs)

    def _evaluate_dataset(self, dataset_name: str) -> float:
        X_train, y_train = self.reader.read_dataset(dataset_name, which_set="train")
        X_test, y_test = self.reader.read_dataset(dataset_name, which_set="test")
        train_repr = self.encoder.encode(X_train)
        test_repr = self.encoder.encode(X_test)
        self.clf.fit(train_repr, y_train)
        preds = self.clf.predict(test_repr)
        return float(np.mean(preds == y_test))

    def evaluate_collection(self, collection: str, dataset_names: Sequence[str]) -> List[Tuple[str, float, Optional[str]]]:
        results: List[Tuple[str, float, Optional[str]]] = []
        for name in dataset_names:
            if collection == "uea" and name in self.skip_uea:
                print(f"[SKIP] {name} (UEA skip list)")
                continue
            print(f"\n===== Evaluating {collection.upper()} :: {name} =====")
            try:
                acc = self._evaluate_dataset(name)
                print(f"{name} accuracy: {acc:.4f}")
                results.append((name, acc, None))
            except torch.cuda.OutOfMemoryError as oom:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                msg = f"CUDA OOM: {oom}"
                print(f"[OOM] {name}: {msg}")
                results.append((name, 0.0, msg))
            except Exception as exc:
                msg = str(exc)
                print(f"[FAIL] {name}: {msg}")
                results.append((name, 0.0, msg))
        return results

    def save_results(self, entries: Sequence[Tuple[str, float, Optional[str]]], prefix: str) -> None:
        successes = [(n, acc) for n, acc, err in entries if err is None]
        failures = [(n, err) for n, _, err in entries if err is not None]
        if successes:
            detail_path = self.results_dir / f"{prefix}_detailed.txt"
            summary_path = self.results_dir / f"{prefix}_summary.txt"
            with open(detail_path, "w", encoding="utf-8") as handle:
                for name, acc in successes:
                    handle.write(f"{name}: {acc:.6f}\n")
            avg_acc = sum(acc for _, acc in successes) / len(successes)
            with open(summary_path, "w", encoding="utf-8") as handle:
                handle.write(f"Total datasets: {len(successes)}\n")
                handle.write(f"Average accuracy: {avg_acc:.6f}\n")
        if failures:
            fail_path = self.results_dir / f"{prefix}_failed.txt"
            with open(fail_path, "w", encoding="utf-8") as handle:
                for name, err in failures:
                    handle.write(f"{name}: {err}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TabICLClassifier on UEA/UCR using Mantis8MWithFDDM embeddings.",
    )
    parser.add_argument("--uea-path", default=DEFAULT_UEA_PATH, help="UEA dataset root path")
    parser.add_argument("--ucr-path", default=DEFAULT_UCR_PATH, help="UCR dataset root path")
    parser.add_argument("--tabicl-ckpt", default=DEFAULT_TABICL_CKPT, help="Path to TabICL checkpoint")
    parser.add_argument(
        "--mantis-fddm-ckpt",
        default=DEFAULT_MANTIS_FDDM_CKPT,
        help="Path to the pretrained Mantis8MWithFDDM checkpoint (folder or file)",
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        choices=["uea", "ucr"],
        default=["ucr"],
        help="Dataset collections to evaluate when explicit names are not provided",
    )
    parser.add_argument(
        "--dataset-file",
        help="Optional text file listing dataset names to evaluate (one per line)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Explicit dataset names to evaluate; overrides --collections",
    )
    parser.add_argument(
        "--limit-per-collection",
        type=int,
        help="Limit the number of datasets evaluated per collection (useful for smoke tests)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length expected by Mantis8MWithFDDM",
    )
    parser.add_argument("--num-channels", type=int, default=1, help="Expected channel count for the encoder")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension of Mantis tokens")
    parser.add_argument("--num-patches", type=int, default=32, help="Number of patches used by Mantis")
    parser.add_argument("--fddm-output-dim", type=int, default=256, help="Dimension of FDDM plugin features")
    parser.add_argument("--mantis-batch-size", type=int, default=64, help="Mini-batch size for encoder forward")
    parser.add_argument(
        "--classifier-batch-size",
        type=int,
        default=8,
        help="Batch size for TabICL ensemble inference",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=32,
        help="Number of ensemble members inside TabICLClassifier",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory to store detailed/summary evaluation logs",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Computation device (e.g., 'cuda:0'). Defaults to CUDA if available",
    )
    parser.add_argument(
        "--skip-default-uea",
        dest="skip_default_uea",
        action="store_true",
        help="Skip known problematic UEA datasets (default)",
    )
    parser.add_argument(
        "--no-skip-default-uea",
        dest="skip_default_uea",
        action="store_false",
        help="Evaluate every UEA dataset even if it is in the skip list",
    )
    parser.set_defaults(skip_default_uea=True)
    parser.add_argument(
        "--robust-norm",
        dest="robust_norm",
        action="store_true",
        help='Use norm_methods=["none","robust"] inside TabICLClassifier (default).',
    )
    parser.add_argument(
        "--default-norm",
        dest="robust_norm",
        action="store_false",
        help="Use TabICL default normalization pipeline",
    )
    parser.set_defaults(robust_norm=True)
    return parser.parse_args()


def build_dataset_plan(args: argparse.Namespace, reader: DataReader) -> Dict[str, List[str]]:
    if args.datasets or args.dataset_file:
        names: List[str] = []
        if args.datasets:
            names.extend(args.datasets)
        if args.dataset_file:
            names.extend(load_dataset_names_from_file(args.dataset_file))
        if not names:
            raise ValueError("No dataset names provided via --datasets or --dataset-file")
        split = split_by_collection(names, reader)
        plan = {k: sorted(v) for k, v in split.items() if v}
    else:
        plan: Dict[str, List[str]] = {}
        if "uea" in args.collections:
            plan["uea"] = sorted(reader.dataset_list_uea)
        if "ucr" in args.collections:
            plan["ucr"] = sorted(reader.dataset_list_ucr)
    if args.limit_per_collection:
        plan = {k: v[: args.limit_per_collection] for k, v in plan.items()}
    return plan


def main() -> None:
    args = parse_args()
    norm_methods = DEFAULT_NORM_METHODS.copy() if args.robust_norm else None

    evaluator = MantisFDDMTabICLEvaluator(
        uea_path=args.uea_path,
        ucr_path=args.ucr_path,
        tabicl_ckpt=args.tabicl_ckpt,
        mantis_fddm_ckpt=args.mantis_fddm_ckpt,
        seq_len=args.seq_len,
        num_channels=args.num_channels,
        hidden_dim=args.hidden_dim,
        num_patches=args.num_patches,
        fddm_output_dim=args.fddm_output_dim,
        mantis_batch_size=args.mantis_batch_size,
        tabicl_batch_size=args.classifier_batch_size,
        tabicl_n_estimators=args.n_estimators,
        norm_methods=norm_methods,
        results_dir=args.results_dir,
        device=args.device,
        skip_uea=SKIP_UEA_DATASETS if args.skip_default_uea else None,
    )

    plan = build_dataset_plan(args, evaluator.reader)
    if not plan:
        print("No datasets selected for evaluation.")
        return

    for collection, names in plan.items():
        results = evaluator.evaluate_collection(collection, names)
        evaluator.save_results(results, f"mantis_fddm_tabicl_{collection}20251128")


if __name__ == "__main__":
    main()
