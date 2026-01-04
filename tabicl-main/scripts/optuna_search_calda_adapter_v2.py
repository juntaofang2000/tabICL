#!/usr/bin/env python3
"""Optuna hyperparameter search for CALDA_AdapterV2 adapter pretraining.

This script runs a lightweight meta-pretraining loop and uses validation meta-loss
as the Optuna objective. It is intentionally faster than the full training script
and does NOT run the expensive full benchmark evaluation per trial.

It reuses the existing training utilities from:
  tabicl.train.train_adapter_with_classifierOrignv2

Typical usage:
  python scripts/optuna_search_calda_adapter_v2.py \
    --n_trials 50 --timeout_sec 0 \
    --steps_per_epoch 10 --val_steps 5 \
    --use_uea_pretrain --no-use_uea_pretrain \
    --study_name calda_v2_search --storage sqlite:///optuna_calda_v2.db

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
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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


@dataclass
class SearchSpace:
    # Adapter hyperparameters
    mcm_num_heads: list[int]
    mcm_dropout_range: tuple[float, float]
    bottleneck_dropout_range: tuple[float, float]

    # Training hyperparameters
    lr_range: tuple[float, float]
    weight_decay_range: tuple[float, float]
    n_augmentations_range: tuple[int, int]
    meta_batch_size_choices: list[int]
    train_size_choices: list[int]
    max_icl_len_choices: list[int]
    mantis_batch_size_choices: list[int]


def _default_search_space() -> SearchSpace:
    return SearchSpace(
        mcm_num_heads=[2, 4, 8],
        mcm_dropout_range=(0.0, 0.2),
        bottleneck_dropout_range=(0.0, 0.2),
        lr_range=(1e-5, 3e-3),
        weight_decay_range=(1e-6, 1e-2),
        n_augmentations_range=(1, 8),
        meta_batch_size_choices=[2, 4, 8, 12, 16],
        train_size_choices=[64, 128, 256],
        max_icl_len_choices=[256, 384, 512],
        mantis_batch_size_choices=[32, 64, 128],
    )


def _limit_list(items: list[str], limit: int) -> list[str]:
    if limit <= 0:
        return items
    return items[: min(len(items), int(limit))]


def _sample_batch(rng: random.Random, names: list[str], k: int) -> list[str]:
    if not names:
        return []
    k = max(1, min(int(k), len(names)))
    return rng.sample(names, k)


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    optuna = _require_optuna()

    parser = argparse.ArgumentParser()

    def _add_bool_optional(name: str, *, default: bool, help_text: str):
        action_cls = getattr(argparse, "BooleanOptionalAction", None)
        if action_cls is not None:
            parser.add_argument(f"--{name}", action=action_cls, default=default, help=help_text)
        else:
            parser.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
            parser.add_argument(f"--no-{name}", dest=name, action="store_false", help=f"Disable: {help_text}")
            parser.set_defaults(**{name: default})

    # Paths / runtime
    parser.add_argument(
        "--tabicl_ckpt",
        type=str,
        default="/data0/fangjuntao2025/tabicl-main/tabICLOrignCheckpoint/tabicl-classifier-v1.1-0506.ckpt",
    )
    parser.add_argument(
        "--mantis_ckpt",
        type=str,
        default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/Models/Mantis/Mantis_cheickpoint/",
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

    # Optuna
    parser.add_argument("--study_name", type=str, default=f"calda_v2_search_{_now_tag()}")
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. sqlite:///optuna_calda_v2.db. If omitted, uses in-memory study.",
    )
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--timeout_sec", type=int, default=0, help="0 means no timeout")
    parser.add_argument(
        "--sampler",
        type=str,
        default="tpe",
        choices=["tpe", "random"],
        help="Optuna sampler type",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="median",
        choices=["none", "median"],
        help="Optuna pruner type",
    )

    # Data scope
    _add_bool_optional(
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
    parser.add_argument(
        "--pretrain_flatten_channels",
        action="store_true",
        help="Flatten multichannel series into single-channel samples during pretraining/validation.",
    )
    parser.add_argument(
        "--use_var_selector",
        action="store_true",
        help="Enable variance-based channel selector for UEA datasets.",
    )
    parser.add_argument(
        "--var_num_channels",
        type=int,
        default=10,
        help="Target number of channels after variance-based selection (UEA only).",
    )

    # Budget per trial
    parser.add_argument("--epochs", type=int, default=2, help="Max epochs per trial")
    parser.add_argument("--steps_per_epoch", type=int, default=10, help="Meta-batches per epoch")
    parser.add_argument("--val_steps", type=int, default=5, help="Validation meta-batches per epoch")

    # Output
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path("checkpoints") / "optuna_calda_v2"),
        help="Directory to save study/trial artifacts",
    )

    args = parser.parse_args()

    # Make sure we can import tabicl from repo root even without installation
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from tabicl.prior.data_reader import DataReader
    from tabicl.model.mantis_tabicl import build_mantis_encoder
    from tabicl.model.tabicl import TabICL
    from tabicl.model.adapterOrign import CALDA_AdapterV2
    from tabicl.train import train_adapter_with_classifierOrign as base

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

    space = _default_search_space()

    def make_sampler():
        if args.sampler == "random":
            return optuna.samplers.RandomSampler(seed=int(args.seed))
        return optuna.samplers.TPESampler(seed=int(args.seed))

    def make_pruner():
        if args.pruner == "none":
            return optuna.pruners.NopPruner()
        # Median pruner works well for noisy objectives
        return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=make_sampler(),
        pruner=make_pruner(),
        storage=args.storage,
        load_if_exists=True,
    )

    # Cache tabicl checkpoint on CPU once
    tabicl_state = torch.load(args.tabicl_ckpt, map_location="cpu")

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()

        # --- Sample hyperparameters ---
        mcm_num_heads = trial.suggest_categorical("mcm_num_heads", space.mcm_num_heads)
        mcm_dropout = trial.suggest_float("mcm_dropout", space.mcm_dropout_range[0], space.mcm_dropout_range[1])
        bottleneck_dropout = trial.suggest_float(
            "bottleneck_dropout",
            space.bottleneck_dropout_range[0],
            space.bottleneck_dropout_range[1],
        )

        lr = trial.suggest_float("lr", space.lr_range[0], space.lr_range[1], log=True)
        weight_decay = trial.suggest_float(
            "weight_decay", space.weight_decay_range[0], space.weight_decay_range[1], log=True
        )
        n_augmentations = trial.suggest_int("n_augmentations", space.n_augmentations_range[0], space.n_augmentations_range[1])
        meta_batch_size = trial.suggest_categorical("meta_batch_size", space.meta_batch_size_choices)
        train_size = trial.suggest_categorical("train_size", space.train_size_choices)
        max_icl_len = trial.suggest_categorical("max_icl_len", space.max_icl_len_choices)
        mantis_batch_size = trial.suggest_categorical("mantis_batch_size", space.mantis_batch_size_choices)

        # --- Build models ---
        # Note: MantIS + TabICL are frozen in base.MantisAdapterTabICL implementation
        mantis_model = build_mantis_encoder(args.mantis_ckpt, device=device)

        tabicl_model = TabICL(**tabicl_state["config"])
        tabicl_model.load_state_dict(tabicl_state["state_dict"])
        tabicl_model.to(device)

        tabicl_dim = 256
        mantis_dim = int(getattr(mantis_model, "hidden_dim", 256))

        # Ensure heads divides embedding dim for MultiheadAttention
        if mantis_dim % int(mcm_num_heads) != 0:
            raise optuna.TrialPruned(f"mantis_dim={mantis_dim} not divisible by heads={mcm_num_heads}")

        adapter = CALDA_AdapterV2(
            mantis_emb_dim=mantis_dim,
            tabicl_input_dim=tabicl_dim,
            out_dim=tabicl_dim,
            mcm_num_heads=int(mcm_num_heads),
            mcm_dropout=float(mcm_dropout),
            bottleneck_dropout=float(bottleneck_dropout),
        ).to(device)

        model = base.MantisAdapterTabICL(
            mantis_model,
            tabicl_model,
            adapter,
            mantis_batch_size=int(mantis_batch_size),
            mantis_fusion="concat",
        ).to(device)

        # --- Local args shim for base.* functions ---
        class _Args:
            pass

        local_args = _Args()
        local_args.train_size = int(train_size)
        local_args.max_icl_len = int(max_icl_len)
        local_args.n_augmentations = int(n_augmentations)
        local_args.debug_grad = False
        local_args.debug_stats = False
        local_args.debug_oob = False

        # Data-related flags used by load_dataset_data
        local_args.pretrain_flatten_channels = bool(args.pretrain_flatten_channels)
        local_args.use_var_selector = bool(args.use_var_selector)
        local_args.var_num_channels = int(args.var_num_channels)

        optimizer = optim.AdamW(model.adapter.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        criterion = nn.CrossEntropyLoss()

        # Deterministic per-trial sampler
        rng = random.Random(int(args.seed) + int(trial.number) * 1009)

        # --- Training loop (lightweight) ---
        best_val = float("inf")
        best_epoch = -1
        local_args.train_no_feat_perm = False      # ✅ 必须：否则 base.train_step 会 AttributeError
        local_args.infer_no_feat_shuffle = False   # 可选：未来如果复用 eval 相关逻辑更稳
        for epoch in range(int(args.epochs)):
            # train epoch
            train_losses: list[float] = []
            for _step in range(int(args.steps_per_epoch)):
                batch_names = _sample_batch(rng, train_names, int(meta_batch_size))
                batch_data = []
                for name in batch_names:
                    is_uea = name in reader.dataset_list_uea
                    loaded = base.load_dataset_data(
                        reader,
                        name,
                        is_uea=is_uea,
                        use_var_selector=bool(args.use_var_selector),
                        var_num_channels=int(args.var_num_channels),
                    )
                    if loaded is None:
                        continue
                    X_tr, y_tr, X_te, y_te = loaded
                    if args.pretrain_flatten_channels:
                        flat = base._flatten_multichannel_as_single_channel(X_tr, y_tr)
                        if flat is not None:
                            X_tr, y_tr = flat
                        flat = base._flatten_multichannel_as_single_channel(X_te, y_te)
                        if flat is not None:
                            X_te, y_te = flat
                    batch_data.append((X_tr, y_tr, X_te, y_te))

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
                    is_uea = name in reader.dataset_list_uea
                    loaded = base.load_dataset_data(
                        reader,
                        name,
                        is_uea=is_uea,
                        use_var_selector=bool(args.use_var_selector),
                        var_num_channels=int(args.var_num_channels),
                    )
                    if loaded is None:
                        continue
                    X_tr, y_tr, X_te, y_te = loaded
                    if args.pretrain_flatten_channels:
                        flat = base._flatten_multichannel_as_single_channel(X_tr, y_tr)
                        if flat is not None:
                            X_tr, y_tr = flat
                        flat = base._flatten_multichannel_as_single_channel(X_te, y_te)
                        if flat is not None:
                            X_te, y_te = flat
                    batch_data.append((X_tr, y_tr, X_te, y_te))

                if not batch_data:
                    continue

                try:
                    vloss = base.validate_step(model, criterion, batch_data, device, local_args)
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
                "pretrain_flatten_channels": bool(args.pretrain_flatten_channels),
                "use_var_selector": bool(args.use_var_selector),
                "var_num_channels": int(args.var_num_channels),
            },
            "timing_sec": float(time.time() - t0),
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
    # Reduce tokenizer thread spam in some environments
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
