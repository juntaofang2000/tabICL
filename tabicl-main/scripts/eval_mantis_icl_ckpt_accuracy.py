from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


# Ensure we import the local workspace package (repo_root/src/tabicl)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tabicl import TabICL  # noqa: E402
from tabicl.model.mantis_tabicl import build_mantis_encoder  # noqa: E402
from tabicl.prior.data_reader import DataReader  # noqa: E402
from tabicl.sklearn.classifier import MantisICLClassifier  # noqa: E402


def _load_tabicl_checkpoint(path: str) -> tuple[TabICL, dict]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "config" not in ckpt:
        raise ValueError("TabICL checkpoint must be a dict containing 'config'.")

    state_dict = ckpt.get("state_dict")
    if state_dict is None:
        for k in ("model_state_dict", "model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]
                break
    if state_dict is None or not isinstance(state_dict, dict):
        raise ValueError("TabICL checkpoint must contain a model state dict ('state_dict' or similar).")

    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model = TabICL(**ckpt["config"])
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model, ckpt["config"]


def _ensure_2d_timeseries(X: np.ndarray) -> np.ndarray:
    """Coerce X into (N, L). Supports UCR (N,L) and UEA (N,C,L)."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        return X
    if X.ndim == 3:
        # UCR sometimes comes as (N,1,L). UEA can be (N,C,L).
        if X.shape[1] == 1:
            return X[:, 0, :]
        # simplest: average channels
        return X.mean(axis=1)
    raise ValueError(f"Unexpected X shape: {X.shape}")


class _MantisPlusICL(nn.Module):
    """A drop-in replacement for MantisICL used inside MantisICLClassifier.

    It implements the same forward signature expected by MantisICLClassifier._batch_forward:
      forward(X, y_train, feature_shuffles=None, return_logits=True, softmax_temperature=0.9, inference_config=None)

    Internals:
      Mantis encoder (from mantis_ckpt) -> TabICL icl_predictor (from tabicl_ckpt)
    """

    def __init__(
        self,
        *,
        mantis_model: nn.Module,
        icl_predictor: nn.Module,
        mantis_seq_len: int = 512,
        mantis_batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.mantis_model = mantis_model
        self.icl_predictor = icl_predictor
        self.mantis_seq_len = int(mantis_seq_len)
        self.mantis_batch_size = int(mantis_batch_size)

        # Required by MantisICLClassifier.fit() for sanity checks
        self.max_classes = int(getattr(self.icl_predictor, "max_classes", 10))

    def train(self, mode: bool = True):
        super().train(mode)
        # keep frozen modules in eval mode
        self.mantis_model.eval()
        self.icl_predictor.eval()
        return self

    def _pad_or_truncate(self, X: torch.Tensor) -> torch.Tensor:
        target = self.mantis_seq_len
        if X.shape[-1] == target:
            return X
        if X.shape[-1] > target:
            return X[..., :target]
        pad = X.new_zeros((*X.shape[:-1], target - X.shape[-1]))
        return torch.cat([X, pad], dim=-1)

    def _encode(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, T, H)
        B, T, H = X.shape
        X = self._pad_or_truncate(X)
        H2 = X.shape[-1]
        x_flat = X.reshape(B * T, 1, H2)

        device = next(self.mantis_model.parameters()).device
        x_flat = x_flat.to(device)

        reps = []
        bs = max(1, int(self.mantis_batch_size))
        with torch.no_grad():
            for i in range(0, x_flat.shape[0], bs):
                reps.append(self.mantis_model(x_flat[i : i + bs]))
        reps = torch.cat(reps, dim=0)
        return reps.reshape(B, T, -1)

    def forward(
        self,
        X: torch.Tensor,
        y_train: torch.Tensor,
        d: torch.Tensor | None = None,
        feature_shuffles=None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config=None,
    ) -> torch.Tensor:
        # feature_shuffles/embed_with_test/d are ignored (MantisICL in this repo also doesn't use them here)
        reps = self._encode(X).to(X.device)

        mgr_config = None
        if inference_config is not None and hasattr(inference_config, "ICL_CONFIG"):
            mgr_config = inference_config.ICL_CONFIG

        return self.icl_predictor(
            reps,
            y_train=y_train,
            return_logits=return_logits,
            softmax_temperature=float(softmax_temperature),
            mgr_config=mgr_config,
        )


def _stratified_support_indices(y: np.ndarray, n_support: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = y.shape[0]
    if n_support >= n:
        return np.arange(n, dtype=np.int64)

    classes = np.unique(y)
    picked: list[int] = []
    # take one from each class (up to n_support)
    rng.shuffle(classes)
    for c in classes:
        idx = np.where(y == c)[0]
        if idx.size == 0:
            continue
        picked.append(int(rng.choice(idx)))
        if len(picked) >= n_support:
            break

    picked = sorted(set(picked))
    if len(picked) < n_support:
        remaining = np.setdiff1d(np.arange(n), np.array(picked, dtype=np.int64), assume_unique=False)
        extra = rng.choice(remaining, size=(n_support - len(picked)), replace=False)
        picked = np.concatenate([np.array(picked, dtype=np.int64), extra]).astype(np.int64)

    rng.shuffle(picked)
    return np.asarray(picked, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate accuracy using MantisICLClassifier, but with its internal model built from: "
            "(1) Mantis checkpoint (.pt) and (2) TabICL checkpoint's icl_predictor (.ckpt)."
        )
    )

    parser.add_argument(
        "--mantis_ckpt",
        type=str,
        default="/data0/fangjuntao2025/tabicl-main/mantis512checkpoints/CaukerImpro-data100k_emb512_100epochs.pt",
    )
    parser.add_argument(
        "--tabicl_ckpt",
        type=str,
        default="/data0/fangjuntao2025/tabicl-main/tabICLOrignCheckpoint/tabicl-classifier-v1.1-0506.ckpt",
    )

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n_estimators", type=int, default=1, help="Ensemble size in MantisICLClassifier.")
    parser.add_argument(
        "--feat_shuffle_method",
        type=str,
        default="latin",
        help="Feature shuffle method used by MantisICLClassifier (none/shift/random/latin).",
    )

    parser.add_argument("--ucr_path", type=str, default= "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/")
    parser.add_argument("--uea_path", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/")
    parser.add_argument("--use_uea", action="store_true")

    parser.add_argument("--dataset", type=str, default=None, help="Evaluate a single dataset name.")

    parser.add_argument("--mantis_hidden_dim", type=int, default=512)
    parser.add_argument("--mantis_seq_len", type=int, default=512)
    parser.add_argument("--mantis_batch_size", type=int, default=64)

    # Note: MantisICLClassifier already has AMP control via its own config.

    args = parser.parse_args()

    device = torch.device(args.device)

    # Load TabICL and take icl_predictor only
    tabicl_model, _tabicl_cfg = _load_tabicl_checkpoint(args.tabicl_ckpt)
    icl_predictor = tabicl_model.icl_predictor
    for p in icl_predictor.parameters():
        p.requires_grad_(False)
    icl_predictor.eval()

    # Load mantis encoder from the provided .pt
    mantis_model = build_mantis_encoder(
        mantis_checkpoint=Path(args.mantis_ckpt),
        device=device,
        hidden_dim=int(args.mantis_hidden_dim),
        seq_len=int(args.mantis_seq_len),
    )
    for p in mantis_model.parameters():
        p.requires_grad_(False)
    mantis_model.eval()

    # Build a MantisICL-compatible module: mantis -> icl_predictor
    custom_model = _MantisPlusICL(
        mantis_model=mantis_model,
        icl_predictor=icl_predictor,
        mantis_seq_len=int(args.mantis_seq_len),
        mantis_batch_size=int(args.mantis_batch_size),
    )
    for p in custom_model.parameters():
        p.requires_grad_(False)
    custom_model.eval()

    reader = DataReader(
        UEA_data_path="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/",
         UCR_data_path= "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/",
        transform_ts_size=int(args.mantis_seq_len),
    )

    if args.dataset is not None:
        dataset_names = [args.dataset]
    else:
        dataset_names = list(reader.dataset_list_ucr)
        if args.use_uea:
            dataset_names = list(reader.dataset_list_ucr) + list(reader.dataset_list_uea)

    # Reuse one classifier instance across datasets (model is heavy; fit() is lightweight setup).
    clf = MantisICLClassifier(
        n_estimators=int(args.n_estimators),
        feat_shuffle_method=str(args.feat_shuffle_method),
        device=device,
        verbose=False,
        model_path=None,
        allow_auto_download=False,
        checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
    )
    clf.model_ = custom_model

    accs: list[float] = []
    for name in dataset_names:
        try:
            X_tr, y_tr = reader.read_dataset(name, which_set="train")
            X_te, y_te = reader.read_dataset(name, which_set="test")
            X_tr_2d = _ensure_2d_timeseries(X_tr)
            X_te_2d = _ensure_2d_timeseries(X_te)

            clf.fit(X_tr_2d, y_tr)
            y_pred = clf.predict(X_te_2d)
            acc = float(np.mean(y_pred == y_te))
            print(f"{name}: {acc:.4f}")
            accs.append(acc)
        except Exception as e:
            print(f"{name}: failed: {e}")

    if accs:
        print(f"\nEvaluated {len(accs)} datasets | mean accuracy: {float(np.mean(accs)):.4f}")
    else:
        print("No datasets evaluated successfully.")


if __name__ == "__main__":
    main()
