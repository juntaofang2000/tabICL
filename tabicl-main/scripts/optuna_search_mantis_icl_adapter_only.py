#!/usr/bin/env python3
"""Optuna hyperparameter search for Mantis->(TokenMLPAdapter)->TabICL(icl_predictor) training.

This script targets the training logic in:
  src/tabicl/train/train_mantis_icl_adapter_only_from_ckpts.py

It runs a lightweight meta-training loop and uses validation meta-loss as the
Optuna objective (faster than full training + evaluation).

Typical usage:
  python scripts/optuna_search_mantis_icl_adapter_only.py \
    --n_trials 50 --timeout_sec 0 \
    --epochs 2 --steps_per_epoch 10 --val_steps 5 \
    --study_name mantis_icl_adapter_only_search \
    --storage sqlite:///optuna_mantis_icl_adapter_only.db

Notes:
- Requires optuna (recommended: pip install optuna).
- Saves per-trial artifacts under --out_dir.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim


def _require_optuna():
    try:
        import optuna  # type: ignore

        return optuna
    except Exception as e:
        raise RuntimeError(
            "Optuna is required for this script. Install it with: pip install optuna\n"
            f"Original import error: {e}"
        ) from e


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _device_from_arg(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    return device


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _limit_list(items: list[str], limit: int) -> list[str]:
    if limit <= 0:
        return items
    return items[: min(len(items), int(limit))]


def _sample_batch(rng: random.Random, names: list[str], k: int) -> list[str]:
    if not names:
        return []
    k = max(1, min(int(k), len(names)))
    return rng.sample(names, k)


@dataclass
class SearchSpace:
    # Training hyperparameters
    lr_range: tuple[float, float]
    weight_decay_range: tuple[float, float]
    n_augmentations_range: tuple[int, int]
    meta_batch_size_choices: list[int]
    train_size_choices: list[int]
    max_icl_len_choices: list[int]
    mantis_batch_size_choices: list[int]

    # Adapter hyperparameters
    adapter_dropout_range: tuple[float, float]
    adapter_hidden_dim_choices: list[int]
    use_layernorm_choices: list[bool]


def _default_search_space() -> SearchSpace:
    return SearchSpace(
        lr_range=(1e-5, 3e-3),
        weight_decay_range=(1e-6, 1e-2),
        n_augmentations_range=(1, 8),
        meta_batch_size_choices=[2, 4, 8, 12, 16],
        train_size_choices=[64, 128, 256],
        max_icl_len_choices=[256, 384, 512],
        mantis_batch_size_choices=[32, 64, 128],
        adapter_dropout_range=(0.0, 0.3),
        adapter_hidden_dim_choices=[0, 256, 512, 1024],
        use_layernorm_choices=[True, False],
    )


def _add_bool_optional(parser: argparse.ArgumentParser, name: str, *, default: bool, help_text: str):
    action_cls = getattr(argparse, "BooleanOptionalAction", None)
    if action_cls is not None:
        parser.add_argument(f"--{name}", action=action_cls, default=default, help=help_text)
    else:
        parser.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
        parser.add_argument(f"--no-{name}", dest=name, action="store_false", help=f"Disable: {help_text}")
        parser.set_defaults(**{name: default})


def validate_step(model, criterion, batch_datasets, device: torch.device, args) -> float | None:
    """Compute mean query CE loss over a meta-batch (no backprop)."""
    # Keep icl_predictor in train mode (see training script notes).
    model.train(True)
    model.adapter.eval()

    prepared = None
    with torch.no_grad():
        prepared = args._prepare_meta_tasks(model, batch_datasets, device, args)
        if prepared is None:
            return None

        adapter_out, y_sup_batch_list, y_qry_batch_list, mask_batch_list, _n_support = prepared

        losses: list[float] = []
        for i in range(int(adapter_out.size(0))):
            emb = adapter_out[i].unsqueeze(0).clone()  # (1,L,D)
            y_sup = y_sup_batch_list[i].unsqueeze(0)
            y_qry = y_qry_batch_list[i].unsqueeze(0)
            mask = mask_batch_list[i].unsqueeze(0)

            if not mask.any():
                continue

            logits = model.icl_predictor(emb, y_train=y_sup)

            qry_len = int(y_qry.size(1))
            logits_qry = logits[:, -qry_len:, :] if logits.size(1) == emb.size(1) else logits

            logits_flat = logits_qry.reshape(-1, logits_qry.size(-1))
            y_flat = y_qry.reshape(-1)
            mask_flat = mask.reshape(-1)
            if not mask_flat.any():
                continue

            y_sel = y_flat[mask_flat]
            C = int(logits_flat.size(-1))
            if int(y_sel.min().item()) < 0 or int(y_sel.max().item()) >= C:
                continue

            loss = criterion(logits_flat[mask_flat], y_sel)
            losses.append(float(loss.item()))

        if not losses:
            return None
        return float(np.mean(losses))


def main():
    optuna = _require_optuna()

    parser = argparse.ArgumentParser(description="Optuna search for Mantis ICL adapter-only training")

    # Paths / runtime
    parser.add_argument(
        "--tabicl_ckpt",
        type=str,
        default="/data0/fangjuntao2025/tabicl-main/tabICLOrignCheckpoint/tabicl-classifier-v1.1-0506.ckpt",
    )
    parser.add_argument(
        "--mantis_ckpt",
        type=str,
        default="/data0/fangjuntao2025/tabicl-main/mantis512checkpoints/CaukerImpro-data100k_emb512_100epochs.pt",
    )
    parser.add_argument(
        "--uea_path",
        type=str,
        default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/",
    )
    parser.add_argument(
        "--ucr_path",
        type=str,
        default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Keep these fixed for comparability (you can still pass different values)
    parser.add_argument("--mantis_hidden_dim", type=int, default=512)
    parser.add_argument("--mantis_seq_len", type=int, default=512)

    # Optuna
    parser.add_argument("--study_name", type=str, default=f"mantis_icl_adapter_only_search_{_now_tag()}")
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. sqlite:///optuna_mantis_icl_adapter_only.db. If omitted, uses in-memory study.",
    )
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--timeout_sec", type=int, default=0, help="0 means no timeout")
    parser.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random"], help="Optuna sampler")
    parser.add_argument(
        "--pruner",
        type=str,
        default="median",
        choices=["none", "median"],
        help="Optuna pruner",
    )

    # Data scope
    _add_bool_optional(
        parser,
        "use_uea_pretrain",
        default=True,
        help_text="Whether to include UEA benchmark datasets during pretraining/validation.",
    )
    parser.add_argument(
        "--limit_datasets",
        type=int,
        default=0,
        help="Limit total dataset names for faster search (0=all). Applied after sorting.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Held-out dataset ratio for validation.",
    )

    # Budget per trial
    parser.add_argument("--epochs", type=int, default=2, help="Max epochs per trial")
    parser.add_argument("--steps_per_epoch", type=int, default=10, help="Meta-batches per epoch")
    parser.add_argument("--val_steps", type=int, default=5, help="Validation meta-batches per epoch")

    # Output
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path("checkpoints") / "optuna_mantis_icl_adapter_only"),
        help="Directory to save study/trial artifacts",
    )

    args = parser.parse_args()

    # Ensure we can import tabicl from repo root even without installation
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from tabicl.prior.data_reader import DataReader
    from tabicl.model.tabicl import TabICL
    from tabicl.model.mantis_tabicl import build_mantis_encoder
    from tabicl.model.mantis_adapter_icl import TokenMLPAdapter

    # Import training utilities from the target script
    from tabicl.train import train_mantis_icl_adapter_only_from_ckpts as base

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist top-level run args
    run_tag = _now_tag()
    run_meta_path = out_dir / f"run_{run_tag}_meta.json"
    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump({"argv": sys.argv, "args": vars(args)}, f, indent=2, ensure_ascii=False)

    _set_seed(int(args.seed))
    device = _device_from_arg(args.device)

    # Dataset lists
    reader = DataReader(UEA_data_path=args.uea_path, UCR_data_path=args.ucr_path)
    if args.use_uea_pretrain:
        all_names = sorted(reader.dataset_list_ucr + reader.dataset_list_uea)
    else:
        all_names = sorted(reader.dataset_list_ucr)

    all_names = _limit_list(all_names, int(args.limit_datasets))
    if len(all_names) < 2:
        raise RuntimeError("Not enough datasets to perform train/val split.")

    # Deterministic split by seed
    split_rng = random.Random(int(args.seed))
    shuffled = all_names.copy()
    split_rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * float(args.val_ratio)))
    val_names = sorted(shuffled[:val_count])
    train_names = sorted(shuffled[val_count:])

    # Cache TabICL checkpoint on CPU once
    tabicl_state = torch.load(args.tabicl_ckpt, map_location="cpu")
    if not isinstance(tabicl_state, dict) or "config" not in tabicl_state:
        raise RuntimeError("TabICL checkpoint must be a dict containing 'config'.")

    # Try to find a usable state_dict key
    state_dict = tabicl_state.get("state_dict")
    if state_dict is None:
        for k in ("model_state_dict", "model"):
            if k in tabicl_state and isinstance(tabicl_state[k], dict):
                state_dict = tabicl_state[k]
                break
    if state_dict is None or not isinstance(state_dict, dict):
        raise RuntimeError("TabICL checkpoint must contain a model state dict ('state_dict' or similar).")

    cleaned_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    tabicl_cfg = dict(tabicl_state["config"])

    embed_dim = int(tabicl_cfg.get("embed_dim", 128))
    row_num_cls = int(tabicl_cfg.get("row_num_cls", 2))
    icl_dim = int(embed_dim * row_num_cls)

    space = _default_search_space()

    def make_sampler():
        if args.sampler == "random":
            return optuna.samplers.RandomSampler(seed=int(args.seed))
        return optuna.samplers.TPESampler(seed=int(args.seed))

    def make_pruner():
        if args.pruner == "none":
            return optuna.pruners.NopPruner()
        return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=make_sampler(),
        pruner=make_pruner(),
        storage=args.storage,
        load_if_exists=True,
    )

    def objective(trial) -> float:
        t0 = time.time()

        # --- Sample hyperparameters ---
        lr = trial.suggest_float("lr", space.lr_range[0], space.lr_range[1], log=True)
        weight_decay = trial.suggest_float("weight_decay", space.weight_decay_range[0], space.weight_decay_range[1], log=True)
        n_augmentations = trial.suggest_int("n_augmentations", space.n_augmentations_range[0], space.n_augmentations_range[1])
        meta_batch_size = trial.suggest_categorical("meta_batch_size", space.meta_batch_size_choices)
        train_size = trial.suggest_categorical("train_size", space.train_size_choices)
        max_icl_len = trial.suggest_categorical("max_icl_len", space.max_icl_len_choices)
        mantis_batch_size = trial.suggest_categorical("mantis_batch_size", space.mantis_batch_size_choices)

        adapter_dropout = trial.suggest_float(
            "adapter_dropout", space.adapter_dropout_range[0], space.adapter_dropout_range[1]
        )
        adapter_hidden_dim_raw = trial.suggest_categorical("adapter_hidden_dim", space.adapter_hidden_dim_choices)
        adapter_hidden_dim = None if int(adapter_hidden_dim_raw) <= 0 else int(adapter_hidden_dim_raw)
        use_layernorm = trial.suggest_categorical("adapter_use_layernorm", space.use_layernorm_choices)

        # --- Build models ---
        mantis_model = build_mantis_encoder(
            mantis_checkpoint=Path(str(args.mantis_ckpt)),
            device=device,
            hidden_dim=int(args.mantis_hidden_dim),
            seq_len=int(args.mantis_seq_len),
        )
        for p in mantis_model.parameters():
            p.requires_grad_(False)
        mantis_model.eval()

        mantis_dim = int(getattr(mantis_model, "hidden_dim", int(args.mantis_hidden_dim)))

        tabicl_model = TabICL(**tabicl_cfg)
        tabicl_model.load_state_dict(cleaned_state_dict, strict=False)
        tabicl_model.to(device)
        tabicl_model.eval()

        icl_predictor = tabicl_model.icl_predictor
        for p in icl_predictor.parameters():
            p.requires_grad_(False)
        icl_predictor.eval()

        adapter = TokenMLPAdapter(
            mantis_dim=int(mantis_dim),
            icl_dim=int(icl_dim),
            hidden_dim=adapter_hidden_dim,
            dropout=float(adapter_dropout),
            use_layernorm=bool(use_layernorm),
        ).to(device)

        model = base.MantisAdapterICLOnly(
            mantis_model=mantis_model,
            icl_predictor=icl_predictor,
            adapter=adapter,
            mantis_seq_len=int(args.mantis_seq_len),
            mantis_batch_size=int(mantis_batch_size),
        ).to(device)
        model.freeze_mantis_and_icl()

        optimizer = optim.AdamW(model.adapter.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        criterion = nn.CrossEntropyLoss()

        # --- Local args shim for base.train_step ---
        class _Args:
            pass

        local_args = _Args()
        local_args.train_size = int(train_size)
        local_args.max_icl_len = int(max_icl_len)
        local_args.n_augmentations = int(n_augmentations)

        # HACK: let validate_step call base._prepare_meta_tasks via local_args
        local_args._prepare_meta_tasks = base._prepare_meta_tasks

        # Deterministic per-trial sampler
        rng = random.Random(int(args.seed) + int(trial.number) * 1009)

        best_val = float("inf")
        best_epoch = -1

        for epoch in range(int(args.epochs)):
            # train epoch
            train_losses: list[float] = []
            for _step in range(int(args.steps_per_epoch)):
                batch_names = _sample_batch(rng, train_names, int(meta_batch_size))
                batch_data = []
                for name in batch_names:
                    loaded = base._load_dataset_tensors(reader, name, seq_len=int(args.mantis_seq_len))
                    if loaded is None:
                        continue
                    X_tr, y_tr, X_te, y_te = loaded
                    batch_data.append((X_tr.to(device), y_tr.to(device), X_te.to(device), y_te.to(device)))

                if not batch_data:
                    continue

                try:
                    loss = base.train_step(model, optimizer, criterion, batch_data, device, local_args)
                    train_losses.append(float(loss))
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        _cleanup_cuda()
                        continue
                    raise

            # validation epoch
            val_losses: list[float] = []
            for _step in range(int(args.val_steps)):
                batch_names = _sample_batch(rng, val_names, int(meta_batch_size))
                batch_data = []
                for name in batch_names:
                    loaded = base._load_dataset_tensors(reader, name, seq_len=int(args.mantis_seq_len))
                    if loaded is None:
                        continue
                    X_tr, y_tr, X_te, y_te = loaded
                    batch_data.append((X_tr.to(device), y_tr.to(device), X_te.to(device), y_te.to(device)))

                if not batch_data:
                    continue

                try:
                    vloss = validate_step(model, criterion, batch_data, device, local_args)
                    if vloss is not None:
                        val_losses.append(float(vloss))
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        _cleanup_cuda()
                        continue
                    raise

            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            trial.report(val_loss, step=epoch)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch

            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at epoch={epoch}, val_loss={val_loss}")

        # --- Save trial artifacts ---
        trial_dir = out_dir / f"trial_{trial.number:05d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "trial": int(trial.number),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "params": dict(trial.params),
            "config": {
                "seed": int(args.seed),
                "use_uea_pretrain": bool(args.use_uea_pretrain),
                "limit_datasets": int(args.limit_datasets),
                "val_ratio": float(args.val_ratio),
                "epochs": int(args.epochs),
                "steps_per_epoch": int(args.steps_per_epoch),
                "val_steps": int(args.val_steps),
                "mantis_seq_len": int(args.mantis_seq_len),
                "mantis_hidden_dim": int(args.mantis_hidden_dim),
                "tabicl_ckpt": str(args.tabicl_ckpt),
                "mantis_ckpt": str(args.mantis_ckpt),
            },
            "timing_sec": float(time.time() - t0),
            "icl_dim": int(icl_dim),
            "mantis_dim": int(mantis_dim),
            "adapter_state_dict": {k: v.detach().cpu() for k, v in model.adapter.state_dict().items()},
        }

        torch.save(ckpt, trial_dir / "adapter_best.pt")
        with open(trial_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "trial": int(trial.number),
                    "best_epoch": int(best_epoch),
                    "best_val_loss": float(best_val),
                    "params": dict(trial.params),
                    "timing_sec": float(ckpt["timing_sec"]),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Cleanup
        del model, adapter, tabicl_model, mantis_model
        _cleanup_cuda()

        return float(best_val)

    timeout = int(args.timeout_sec)
    timeout = None if timeout <= 0 else timeout

    study.optimize(objective, n_trials=int(args.n_trials), timeout=timeout, gc_after_trial=True)

    # Save best trial / study summary
    best = study.best_trial
    summary = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "best_value": float(best.value) if best.value is not None else None,
        "best_params": dict(best.params),
        "best_trial_number": int(best.number),
        "n_trials": int(len(study.trials)),
        "storage": args.storage,
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(out_dir / "study_best.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[Optuna] Finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
