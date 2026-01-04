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
from tabicl.model.mantis_adapter_icl import TokenMLPAdapter  # noqa: E402
from tabicl.model.mantis_tabicl import build_mantis_encoder  # noqa: E402
from tabicl.prior.data_reader import DataReader  # noqa: E402
from tabicl.sklearn.classifier import MantisICLClassifier  # noqa: E402


def _remap_labels(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map labels to contiguous ints [0..K-1] based on train set."""
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    classes = np.unique(y_train)
    cls_to_id = {c: i for i, c in enumerate(classes.tolist())}

    y_train_m = np.vectorize(cls_to_id.get)(y_train)
    y_test_m = np.vectorize(cls_to_id.get)(y_test)

    if np.any(y_test_m == None):  # noqa: E711
        missing = set(np.unique(y_test)) - set(classes)
        raise ValueError(f"Test labels contain unseen classes: {sorted(missing)}")

    return y_train_m.astype(np.int64), y_test_m.astype(np.int64), classes


def _select_support_indices(y: np.ndarray, support_size: int, seed: int) -> np.ndarray:
    """Pick support indices ensuring all classes appear at least once."""
    rng = np.random.RandomState(int(seed))
    y = np.asarray(y)
    classes = np.unique(y)
    n_classes = int(classes.shape[0])
    support_size = int(support_size)
    if support_size < n_classes:
        support_size = n_classes

    chosen: list[int] = []
    remaining: list[int] = []

    for c in classes:
        idx = np.where(y == c)[0]
        if idx.size == 0:
            continue
        pick = int(rng.choice(idx))
        chosen.append(pick)
        # store the rest as remaining pool
        remaining.extend([int(i) for i in idx if int(i) != pick])

    if len(chosen) > support_size:
        # Extremely rare: support_size < n_classes handled above; keep a stable subset anyway
        chosen = chosen[:support_size]

    need = support_size - len(chosen)
    if need > 0:
        remaining = np.array(remaining, dtype=np.int64)
        if remaining.size > 0:
            extra = rng.choice(remaining, size=min(need, remaining.size), replace=False)
            chosen.extend([int(i) for i in extra])

    return np.array(chosen, dtype=np.int64)


@torch.no_grad()
def _predict_direct(
    model: "_MantisAdapterPlusICL",
    *,
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_query: np.ndarray,
    query_batch_size: int,
    softmax_temperature: float,
    mgr_config=None,
) -> np.ndarray:
    """Directly predict labels for X_query using (support + query) ICL tables.

    This avoids sklearn pipeline and caches support representations.
    """

    device = next(model.adapter.parameters()).device
    X_support_t = torch.from_numpy(X_support.astype(np.float32)).to(device)
    X_query_t = torch.from_numpy(X_query.astype(np.float32)).to(device)
    y_support_t = torch.from_numpy(y_support.astype(np.float32)).to(device)

    # Shapes: support (S, L) -> (1, S, L), query (N, L) -> (B, 1, L)
    X_support_t = X_support_t.unsqueeze(0)
    y_support_t = y_support_t.unsqueeze(0)

    support_rep = model.adapter(model._encode(X_support_t))  # (1, S, D)

    preds: list[np.ndarray] = []
    bs = max(1, int(query_batch_size))
    for i in range(0, X_query_t.shape[0], bs):
        q = X_query_t[i : i + bs].unsqueeze(1)  # (B, 1, L)
        query_rep = model.adapter(model._encode(q))  # (B, 1, D)

        B = query_rep.shape[0]
        reps_all = torch.cat([support_rep.repeat(B, 1, 1), query_rep], dim=1)  # (B, S+1, D)
        y_train = y_support_t.repeat(B, 1)  # (B, S)

        logits = model.icl_predictor(
            reps_all,
            y_train=y_train,
            return_logits=True,
            softmax_temperature=float(softmax_temperature),
            mgr_config=mgr_config,
        )
        # logits: (B, 1, num_classes)
        pred = torch.argmax(logits[:, 0, :], dim=-1).detach().cpu().numpy()
        preds.append(pred)

    return np.concatenate(preds, axis=0)


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
        if X.shape[1] == 1:
            return X[:, 0, :]
        return X.mean(axis=1)
    raise ValueError(f"Unexpected X shape: {X.shape}")


def _find_latest_adapter_ckpt(dir_path: str) -> str:
    p = Path(dir_path)
    if not p.is_dir():
        raise FileNotFoundError(f"adapter ckpt dir not found: {dir_path}")

    candidates = list(p.glob("*_epoch*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No adapter ckpt files matching '*_epoch*.pt' under {dir_path}")

    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(candidates[0])


class _MantisAdapterPlusICL(nn.Module):
    """Mantis encoder -> Adapter -> TabICL icl_predictor.

    Implements the forward signature expected by MantisICLClassifier._batch_forward.
    """

    def __init__(
        self,
        *,
        mantis_model: nn.Module,
        adapter: nn.Module,
        icl_predictor: nn.Module,
        mantis_seq_len: int = 512,
        mantis_batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.mantis_model = mantis_model
        self.adapter = adapter
        self.icl_predictor = icl_predictor
        self.mantis_seq_len = int(mantis_seq_len)
        self.mantis_batch_size = int(mantis_batch_size)

        self.max_classes = int(getattr(self.icl_predictor, "max_classes", 10))

    def train(self, mode: bool = True):
        super().train(mode)
        # evaluation-only: keep everything in eval
        self.mantis_model.eval()
        self.adapter.eval()
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
        B, T, _H = X.shape
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
        # feature_shuffles/embed_with_test/d are ignored for this path
        reps = self._encode(X)
        reps = reps.to(X.device)
        reps = self.adapter(reps)

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained adapter (mantis->adapter->icl_predictor) on UCR. "
            "Adapter checkpoint is produced by src/tabicl/train/train_mantis_icl_adapter_only_from_ckpts.py."
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="direct",
        choices=["direct", "classifier"],
        help="Evaluation mode: 'direct' uses _MantisAdapterPlusICL directly; 'classifier' uses MantisICLClassifier.",
    )

    parser.add_argument(
        "--adapter_ckpt",
        type=str,
        default=None,
        help="Path to adapter checkpoint (.pt) saved by the training script.",
    )
    parser.add_argument(
        "--adapter_ckpt_dir",
        type=str,
        default="/data0/fangjuntao2025/tabicl-main/checkpoints/mantis_icl_adapter_only",
        help="Directory to auto-pick the latest '*_epoch*.pt' if --adapter_ckpt is not provided.",
    )

    parser.add_argument(
        "--mantis_ckpt",
        type=str,
        default="/data0/fangjuntao2025/tabicl-main/mantis512checkpoints/CaukerImpro-data100k_emb512_100epochs.pt",
        help="Mantis checkpoint (only used if not present in adapter ckpt).",
    )
    parser.add_argument(
        "--tabicl_ckpt",
        type=str,
        default="/data0/fangjuntao2025/tabicl-main/tabICLOrignCheckpoint/tabicl-classifier-v1.1-0506.ckpt",
        help="TabICL checkpoint (only used if not present in adapter ckpt).",
    )

    parser.add_argument("--ucr_path", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/")
    parser.add_argument(
        "--uea_path",
        type=str,
        default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/",
        help="Only for DataReader initialization; UCR evaluation does not require UEA datasets.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--dataset", type=str, default=None, help="Evaluate a single UCR dataset name")

    parser.add_argument("--n_estimators", type=int, default=1)
    parser.add_argument("--feat_shuffle_method", type=str, default="latin")

    parser.add_argument(
        "--support_size",
        type=int,
        default=128,
        help="(direct mode) Number of training samples used as ICL support (auto-bumped to >= #classes).",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=64,
        help="(direct mode) Query batch size (number of test samples per forward).",
    )
    parser.add_argument(
        "--softmax_temperature",
        type=float,
        default=0.9,
        help="Softmax temperature passed to icl_predictor inference.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for support sampling (direct mode).")

    parser.add_argument("--mantis_hidden_dim", type=int, default=512)
    parser.add_argument("--mantis_seq_len", type=int, default=512)
    parser.add_argument("--mantis_batch_size", type=int, default=64)

    args = parser.parse_args()

    device = torch.device(args.device)

    adapter_ckpt_path = args.adapter_ckpt or _find_latest_adapter_ckpt(args.adapter_ckpt_dir)
    adapter_ckpt = torch.load(adapter_ckpt_path, map_location="cpu")
    if not isinstance(adapter_ckpt, dict) or "adapter_state_dict" not in adapter_ckpt:
        raise ValueError(f"Invalid adapter checkpoint: {adapter_ckpt_path}")

    train_args = adapter_ckpt.get("args") if isinstance(adapter_ckpt.get("args"), dict) else {}

    mantis_ckpt = str(adapter_ckpt.get("mantis_ckpt", args.mantis_ckpt))
    tabicl_ckpt = str(adapter_ckpt.get("tabicl_ckpt", args.tabicl_ckpt))

    # Load TabICL and take icl_predictor only
    tabicl_model, tabicl_cfg = _load_tabicl_checkpoint(tabicl_ckpt)
    icl_predictor = tabicl_model.icl_predictor
    for p in icl_predictor.parameters():
        p.requires_grad_(False)
    icl_predictor.to(device)
    icl_predictor.eval()

    embed_dim = int(tabicl_cfg.get("embed_dim", 128))
    row_num_cls = int(tabicl_cfg.get("row_num_cls", 2))
    icl_dim = int(adapter_ckpt.get("icl_dim", embed_dim * row_num_cls))

    # Load mantis encoder
    mantis_model = build_mantis_encoder(
        mantis_checkpoint=Path(mantis_ckpt),
        device=device,
        hidden_dim=int(args.mantis_hidden_dim),
        seq_len=int(args.mantis_seq_len),
    )
    for p in mantis_model.parameters():
        p.requires_grad_(False)
    mantis_model.eval()

    mantis_dim = int(adapter_ckpt.get("mantis_dim", getattr(mantis_model, "hidden_dim", int(args.mantis_hidden_dim))))

    adapter = TokenMLPAdapter(
        mantis_dim=int(mantis_dim),
        icl_dim=int(icl_dim),
        hidden_dim=(
            None
            if train_args.get("adapter_hidden_dim") is None
            else int(train_args.get("adapter_hidden_dim"))
        ),
        dropout=float(train_args.get("adapter_dropout", 0.0)),
        use_layernorm=not bool(train_args.get("adapter_no_layernorm", False)),
    )
    adapter.load_state_dict(adapter_ckpt["adapter_state_dict"], strict=True)
    adapter.to(device)
    adapter.eval()

    custom_model = _MantisAdapterPlusICL(
        mantis_model=mantis_model,
        adapter=adapter,
        icl_predictor=icl_predictor,
        mantis_seq_len=int(args.mantis_seq_len),
        mantis_batch_size=int(args.mantis_batch_size),
    )
    for p in custom_model.parameters():
        p.requires_grad_(False)
    custom_model.eval()

    reader = DataReader(
        UEA_data_path=str(args.uea_path),
        UCR_data_path=str(args.ucr_path),
        transform_ts_size=int(args.mantis_seq_len),
    )
    if args.dataset is not None:
        dataset_names = [args.dataset]
    else:
        dataset_names = list(reader.dataset_list_ucr)

    print(f"[Eval] adapter_ckpt: {adapter_ckpt_path}")
    print(f"[Eval] mantis_ckpt: {mantis_ckpt}")
    print(f"[Eval] tabicl_ckpt: {tabicl_ckpt}")
    print(f"[Eval] mode: {args.mode}")

    accs: list[float] = []
    for name in dataset_names:
        try:
            X_tr, y_tr = reader.read_dataset(name, which_set="train")
            X_te, y_te = reader.read_dataset(name, which_set="test")
            X_tr_2d = _ensure_2d_timeseries(X_tr)
            X_te_2d = _ensure_2d_timeseries(X_te)

            y_tr_m, y_te_m, _classes = _remap_labels(y_tr, y_te)

            if args.mode == "classifier":
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
                clf.fit(X_tr_2d, y_tr_m)
                y_pred = clf.predict(X_te_2d)
                acc = float(np.mean(y_pred == y_te_m))
            else:
                sup_idx = _select_support_indices(y_tr_m, support_size=int(args.support_size), seed=int(args.seed))
                X_sup = X_tr_2d[sup_idx]
                y_sup = y_tr_m[sup_idx]

                y_pred = _predict_direct(
                    custom_model,
                    X_support=X_sup,
                    y_support=y_sup,
                    X_query=X_te_2d,
                    query_batch_size=int(args.query_batch_size),
                    softmax_temperature=float(args.softmax_temperature),
                    mgr_config=None,
                )
                acc = float(np.mean(y_pred == y_te_m))

            print(f"{name}: {acc:.4f}")
            accs.append(acc)
        except Exception as e:
            print(f"{name}: failed: {e}")

    if accs:
        print(f"\nEvaluated {len(accs)} UCR datasets | mean accuracy: {float(np.mean(accs)):.4f}")
    else:
        print("No datasets evaluated successfully.")


if __name__ == "__main__":
    main()
