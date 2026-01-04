"""train_adapter_with_classifierOrignv2

Changes vs train_adapter_with_classifierOrign:
- During adapter pretraining/validation ONLY, treat multichannel time series as a set
  of single-channel samples by flattening channels into the sample dimension.

Example:
  X shape (125, 900, 512) -> (125*900, 1, 512)
  Labels are repeated per channel (same label for all channels of a sample).

Evaluation keeps the original dataset shapes/labels (no flattening).
"""

import argparse
import os
import sys
import json
import copy
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

# Add src to path (keep consistent with original script)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

import tabicl.train.train_adapter_with_classifierOrign as base


def _flatten_multichannel_as_single_channel(X: torch.Tensor, y: torch.Tensor):
    """Flatten (N, C, L) -> (N*C, 1, L) and repeat labels.

    If C==1, returns inputs unchanged.
    """
    if X.dim() != 3:
        return X, y
    N, C, L = X.shape
    if C <= 1:
        return X, y

    # (N, C, L) -> (N*C, 1, L)
    X_flat = X.contiguous().reshape(N * C, L).unsqueeze(1)
    # (N,) -> (N*C,)
    y_flat = y.repeat_interleave(C)
    return X_flat, y_flat


def load_dataset_data_pretrain_v2(reader, dataset_name, *, is_uea: bool, use_var_selector: bool, var_num_channels: int | None):
    """Load dataset for pretraining with channel-flattening (multichannel -> many single-channel samples)."""
    X_train, y_train, X_test, y_test = base.load_dataset_data(
        reader,
        dataset_name,
        is_uea=is_uea,
        use_var_selector=use_var_selector,
        var_num_channels=var_num_channels,
    )
    if X_train is None:
        return None, None, None, None

    X_train, y_train = _flatten_multichannel_as_single_channel(X_train, y_train)
    X_test, y_test = _flatten_multichannel_as_single_channel(X_test, y_test)
    return X_train, y_train, X_test, y_test


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--uea_path", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/")
    parser.add_argument("--ucr_path", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--div_weight", type=float, default=0.1)
    parser.add_argument("--max_icl_len", type=int, default=512, help="Max sequence length for ICL training to avoid OOM")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_adapter", action="store_true", help="Disable adapter and use raw Mantis embeddings")
    parser.add_argument("--mantis_batch_size", type=int, default=16, help="Batch size for Mantis encoder")
    parser.add_argument("--meta_batch_size", type=int, default=16, help="Number of datasets per training step")
    parser.add_argument("--train_size", type=int, default=100, help="Number of support samples (context size)")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--n_augmentations", type=int, default=5, help="Number of augmentations per dataset")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio (held-out datasets) during adapter pretraining")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="/data0/fangjuntao2025/tabicl-main/checkpoints/mantis_adapter_pretrainNew",
        help="Directory to save adapter checkpoints",
    )
    parser.add_argument("--ckpt_prefix", type=str, default="adapter", help="Checkpoint filename prefix")
    parser.add_argument("--save_last", action="store_true", help="Also save last checkpoint each epoch")

    parser.add_argument(
        "--mantis_fusion",
        type=str,
        default="concat",
        choices=["concat", "sum"],
        help="How to fuse per-channel Mantis embeddings. 'concat' flattens channels; 'sum' adds them elementwise (v2).",
    )

    parser.add_argument(
        "--eval_adapter_ckpt",
        type=str,
        default=None,
        help="Path to an adapter checkpoint (.pt) to load for evaluation. If not set, auto-load best from ckpt_dir.",
    )

    parser.add_argument(
        "--use_var_selector",
        action="store_true",
        help="Enable VarianceBasedSelector channel compression for UEA datasets only (UCR unchanged).",
    )
    parser.add_argument(
        "--var_num_channels",
        type=int,
        default=10,
        help="Target number of channels after variance-based selection (UEA only).",
    )

    parser.add_argument(
        "--infer_no_feat_shuffle",
        action="store_true",
        help="During evaluation/inference, disable TabICLClassifier feature-dimension shuffling (D-dim permutation). Keeps norm + class shift.",
    )
    parser.add_argument(
        "--train_no_feat_perm",
        action="store_true",
        help="During adapter pretraining/meta-task augmentation, disable feature-dimension permutation (D-dim randperm). Keep norm + class shift.",
    )
    parser.add_argument(
        "--debug_grad",
        action="store_true",
        help="Print requires_grad flags and grad_fn for TabICL logits during training.",
    )
    parser.add_argument(
        "--debug_oob",
        action="store_true",
        help="Print details when CrossEntropy targets are out-of-bounds (prevents CUDA assert).",
    )

    args = parser.parse_args()

    base.set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    print("Loading models...")
    mantis_model = base.build_mantis_encoder(args.mantis_ckpt, device=device)

    tabicl_state = torch.load(args.tabicl_ckpt, map_location="cpu")
    tabicl_model = base.TabICL(**tabicl_state["config"])
    tabicl_model.load_state_dict(tabicl_state["state_dict"])
    tabicl_model.to(device)

    tabicl_dim = 256
    mantis_dim = mantis_model.hidden_dim
    print(f"Mantis Dim: {mantis_dim}, TabICL Dim: {tabicl_dim}")

    # Keep original behavior for sum-fusion
    if args.mantis_fusion == "sum":
        if not args.no_adapter:
            print("[Info] mantis_fusion='sum' (v2) -> forcing --no_adapter (no learnable adapter).")
        args.no_adapter = True

    if args.no_adapter:
        adapter = None
    else:
        adapter = base.CALDA_Adapter(mantis_emb_dim=256, tabicl_input_dim=256).to(device)

    run_tag = base._adapter_run_tag(adapter, args)
    ckpt_base = f"{args.ckpt_prefix}_{run_tag}" if args.ckpt_prefix else run_tag

    model = base.MantisAdapterTabICL(
        mantis_model,
        tabicl_model,
        adapter,
        mantis_batch_size=args.mantis_batch_size,
        mantis_fusion=args.mantis_fusion,
    ).to(device)

    reader = base.DataReader(UEA_data_path=args.uea_path, UCR_data_path=args.ucr_path)

    # Combine UCR and UEA datasets (keep original list selection)
    datasets = sorted(reader.dataset_list_ucr)

    train_datasets = datasets
    val_datasets = []
    if (not args.no_adapter) and args.val_ratio > 0:
        rng = random.Random(args.seed)
        shuffled = datasets.copy()
        rng.shuffle(shuffled)
        val_count = max(1, int(len(shuffled) * args.val_ratio)) if len(shuffled) > 1 else 0
        val_datasets = sorted(shuffled[:val_count])
        train_datasets = sorted(shuffled[val_count:]) if val_count > 0 else datasets
        print(f"Meta split: train={len(train_datasets)}, val={len(val_datasets)} (val_ratio={args.val_ratio})")

    # --- Pretraining Phase ---
    if not args.no_adapter:
        print(
            f"Starting Pretraining (single-channel sampling) on {len(train_datasets)} datasets for {args.epochs} epochs..."
        )
        optimizer = optim.AdamW(model.adapter.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        os.makedirs(args.ckpt_dir, exist_ok=True)
        best_val = float("inf")
        best_epoch = -1

        for epoch in range(args.epochs):
            random.shuffle(train_datasets)
            epoch_loss = 0.0
            count = 0

            num_batches = (len(train_datasets) + args.meta_batch_size - 1) // args.meta_batch_size
            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{args.epochs}")

            for i in pbar:
                batch_names = train_datasets[i * args.meta_batch_size : (i + 1) * args.meta_batch_size]

                batch_data = []
                for name in batch_names:
                    is_uea = name in reader.dataset_list_uea
                    X_tr, y_tr, X_te, y_te = load_dataset_data_pretrain_v2(
                        reader,
                        name,
                        is_uea=is_uea,
                        use_var_selector=args.use_var_selector,
                        var_num_channels=args.var_num_channels,
                    )
                    if X_tr is not None:
                        batch_data.append((X_tr, y_tr, X_te, y_te))

                if not batch_data:
                    continue

                try:
                    loss = base.train_step(model, optimizer, criterion, batch_data, device, args)
                    epoch_loss += loss
                    count += 1
                    pbar.set_postfix({"avg_loss": epoch_loss / count if count > 0 else 0})
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("\nSkipping batch due to OOM")
                        torch.cuda.empty_cache()
                        continue
                    raise

            avg_train_loss = epoch_loss / count if count > 0 else 0.0

            # Validation (also uses single-channel sampling)
            avg_val_loss = None
            if val_datasets:
                val_loss_sum = 0.0
                val_steps = 0
                num_val_batches = (len(val_datasets) + args.meta_batch_size - 1) // args.meta_batch_size
                for i in tqdm(range(num_val_batches), desc=f"Val {epoch+1}/{args.epochs}", leave=False):
                    batch_names = val_datasets[i * args.meta_batch_size : (i + 1) * args.meta_batch_size]
                    batch_data = []
                    for name in batch_names:
                        is_uea = name in reader.dataset_list_uea
                        X_tr, y_tr, X_te, y_te = load_dataset_data_pretrain_v2(
                            reader,
                            name,
                            is_uea=is_uea,
                            use_var_selector=args.use_var_selector,
                            var_num_channels=args.var_num_channels,
                        )
                        if X_tr is not None:
                            batch_data.append((X_tr, y_tr, X_te, y_te))
                    if not batch_data:
                        continue
                    try:
                        vloss = base.validate_step(model, criterion, batch_data, device, args)
                        if vloss is None:
                            continue
                        val_loss_sum += vloss
                        val_steps += 1
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            continue
                        raise
                avg_val_loss = val_loss_sum / val_steps if val_steps > 0 else None

            ckpt_common = {
                "epoch": epoch,
                "adapter_state_dict": copy.deepcopy(model.adapter.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "config": vars(args),
                "run_tag": run_tag,
                "tabicl_ckpt": args.tabicl_ckpt,
                "mantis_ckpt": args.mantis_ckpt,
                "seed": args.seed,
+                "pretrain_sampling": "flatten_channels_to_single_channel",
            }

            if args.save_last:
                last_path = os.path.join(args.ckpt_dir, f"{ckpt_base}_last.pt")
                torch.save(ckpt_common, last_path)

            if avg_val_loss is not None and avg_val_loss < best_val:
                best_val = avg_val_loss
                best_epoch = epoch
                best_path = os.path.join(args.ckpt_dir, f"{ckpt_base}_best.pt")
                ckpt_best = dict(ckpt_common)
                ckpt_best["best_val_loss"] = best_val
                ckpt_best["best_epoch"] = best_epoch
                torch.save(ckpt_best, best_path)
                print(f"Saved best checkpoint: {best_path} (val_loss={best_val:.6f})")
            elif not val_datasets:
                train_path = os.path.join(args.ckpt_dir, f"{ckpt_base}_best_trainloss.pt")
                torch.save(ckpt_common, train_path)
                print(f"Saved checkpoint (no val): {train_path} (train_loss={avg_train_loss:.6f})")
            else:
                if avg_val_loss is not None:
                    print(
                        f"Epoch {epoch+1}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}, best_val={best_val:.6f}"
                    )
                else:
                    print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.6f}, val_loss=N/A, best_val={best_val:.6f}")

        print("Pretraining finished.")

    # --- Evaluation Phase (unchanged; no channel-flattening) ---
    print("Starting Evaluation...")
    results: dict[str, float] = {}

    if (not args.no_adapter) and (model.adapter is not None):
        ckpt_to_load = None
        if args.eval_adapter_ckpt:
            ckpt_to_load = args.eval_adapter_ckpt
        else:
            adapter_name = model.adapter.__class__.__name__
            ckpt_to_load = base._find_latest_ckpt(
                args.ckpt_dir,
                patterns=[
                    f"*{adapter_name}*_best.pt",
                    f"*{adapter_name}*_best_trainloss.pt",
                    f"*{adapter_name}*_last.pt",
                    "*_best.pt",
                    "*_best_trainloss.pt",
                    "*_last.pt",
                ],
            )

        if ckpt_to_load is not None and os.path.isfile(ckpt_to_load):
            ckpt = torch.load(ckpt_to_load, map_location="cpu")
            adapter_state = ckpt.get("adapter_state_dict", ckpt)
            model.adapter.load_state_dict(adapter_state)
            model.adapter.to(device)
            print(f"[Eval] Loaded adapter weights: {ckpt_to_load}")
        else:
            print("[Eval] No adapter checkpoint found; using current adapter weights.")

    infer_feat_shuffle_method = "none" if args.infer_no_feat_shuffle else "latin"
    if args.infer_no_feat_shuffle:
        print("[Eval] TabICLClassifier: feature shuffle disabled (feat_shuffle_method='none').")

    clf = base.TabICLClassifier(
        model_path=args.tabicl_ckpt,
        n_estimators=32,
        feat_shuffle_method=infer_feat_shuffle_method,
        device=device,
        verbose=False,
        mantis_checkpoint=None,
        batch_size=8,
    )

    all_datasets = sorted(reader.dataset_list_ucr)
    for dataset_name in tqdm(all_datasets, desc="Evaluating"):
        try:
            is_uea = dataset_name in reader.dataset_list_uea
            X_train, y_train, X_test, y_test = base.load_dataset_data(
                reader,
                dataset_name,
                is_uea=is_uea,
                use_var_selector=args.use_var_selector,
                var_num_channels=args.var_num_channels,
            )
            if X_train is None:
                continue

            X_train_emb = base.get_embeddings(model, X_train, device)
            X_test_emb = base.get_embeddings(model, X_test, device)

            clf.fit(X_train_emb, y_train.numpy())
            y_pred = clf.predict(X_test_emb)
            acc = float(np.mean(y_pred == y_test.numpy()))
            results[dataset_name] = acc
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nSkipping {dataset_name} due to OOM")
                torch.cuda.empty_cache()
                continue
            raise
        except Exception as e:
            print(f"\nError evaluating {dataset_name}: {e}")
            continue

    print("\nFinal Results:")

    uea_results = {name: acc for name, acc in results.items() if name in reader.dataset_list_uea}
    ucr_results = {name: acc for name, acc in results.items() if name in reader.dataset_list_ucr}

    if uea_results:
        print(f"\n--- UEA Benchmark ({len(uea_results)} datasets) ---")
        for name in sorted(uea_results.keys()):
            print(f"{name}: {uea_results[name]:.4f}")
        print(f"Average UEA Accuracy: {np.mean(list(uea_results.values())):.4f}")

    if ucr_results:
        print(f"\n--- UCR Benchmark ({len(ucr_results)} datasets) ---")
        for name in sorted(ucr_results.keys()):
            print(f"{name}: {ucr_results[name]:.4f}")
        print(f"Average UCR Accuracy: {np.mean(list(ucr_results.values())):.4f}")

    if results:
        print(f"\nOverall Average Accuracy: {np.mean(list(results.values())):.4f}")

    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        structured_results = {"UEA": uea_results, "UCR": ucr_results}
        with open(args.output_file, "w") as f:
            json.dump(structured_results, f, indent=4)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
