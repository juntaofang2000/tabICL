import argparse, torch, torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from tabicl.model.mantis_tabicl import MantisTabICL
from tabicl.prior.data_reader import DataReader

# 这个finetune 代码有问题，每一个数据集就loss 更新一次，相当于batchsize 为1, 这是不对的

def build_parser():
    p = argparse.ArgumentParser(description="Fine-tune MantisTabICL on UCR/UEA.")
    p.add_argument(
        "--tabicl-ckpt",
        default="/data0/fangjuntao2025/tabicl-main/tabICLOrignCheckpoint/tabicl-classifier-v1.1-0506.ckpt",
        help="Path to TabICL checkpoint",
    )
    p.add_argument(
        "--mantis-ckpt",
        default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/Models/Mantis/Mantis_cheickpoint/",
        help="Path to Mantis checkpoint file or directory",
    )
    p.add_argument("--uea-path", default= "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/"  )
    p.add_argument("--ucr-path", default= "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/" )
    p.add_argument("--save-dir", default="ft_checkpoints")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--train-size", type=int, default=100, help="前多少样本当作上下文标签")
    p.add_argument("--repr-cache", default=None, help="Directory to cache Mantis representations for faster epochs")
    return p

def build_context_and_targets(y_train, y_test, train_size, device):
    """Return context labels from training split only and targets for remaining rows."""

    if y_train.shape[0] < train_size:
        raise ValueError(
            f"train split has only {y_train.shape[0]} rows, which is fewer than train_size={train_size}"
        )

    ctx = torch.from_numpy(y_train[:train_size]).long().to(device)
    remainder = []
    if y_train.shape[0] > train_size:
        remainder.append(torch.from_numpy(y_train[train_size:]).long())
    if y_test.shape[0] > 0:
        remainder.append(torch.from_numpy(y_test).long())

    if remainder:
        tgt = torch.cat(remainder, dim=0).to(device)
    else:
        tgt = torch.empty(0, dtype=torch.long, device=device)

    return ctx, tgt


def _sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def maybe_load_cached_repr(cache_dir: Path | None, dataset_name: str):
    if not cache_dir:
        return None
    cache_file = cache_dir / f"{_sanitize_name(dataset_name)}.pt"
    if cache_file.is_file():
        return torch.load(cache_file, map_location="cpu")
    return None


def maybe_save_cached_repr(cache_dir: Path | None, dataset_name: str, tensor: torch.Tensor):
    if not cache_dir:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{_sanitize_name(dataset_name)}.pt"
    torch.save(tensor.cpu(), cache_file)
    


def collate_batch(batch_samples, train_size, device):
    """Truncate to the shortest table in the batch for uniform shapes."""

    batch_size = len(batch_samples)
    min_seq_len = min(sample["repr"].shape[0] for sample in batch_samples)
    if min_seq_len <= train_size:
        raise ValueError("All tables in a batch must be longer than train_size to keep targets non-empty.")

    tgt_window = min_seq_len - train_size
    max_feat_dim = max(sample["repr"].shape[1] for sample in batch_samples)

    repr_batch = torch.zeros(batch_size, min_seq_len, max_feat_dim, device=device)
    ctx_batch = torch.zeros(batch_size, train_size, device=device, dtype=torch.long)
    tgt_batch = torch.full((batch_size, tgt_window), fill_value=-100, device=device, dtype=torch.long)
    tgt_mask = torch.zeros(batch_size, tgt_window, device=device, dtype=torch.bool)

    for idx, sample in enumerate(batch_samples):
        seq = sample["repr"]
        feat_dim = seq.shape[1]
        repr_batch[idx, :, :feat_dim] = seq[:min_seq_len]

        ctx = sample["ctx"]
        ctx_batch[idx] = ctx[:train_size]

        tgt = sample["tgt"]
        tgt_len = min(tgt.shape[0], tgt_window)
        if tgt_len > 0:
            tgt_batch[idx, :tgt_len] = tgt[:tgt_len]
            tgt_mask[idx, :tgt_len] = True

    return repr_batch, ctx_batch, tgt_batch, tgt_mask


def run_batch(model, batch_samples, train_size, device, use_amp: bool):
    try:
        repr_batch, ctx_batch, tgt_batch, tgt_mask = collate_batch(batch_samples, train_size, device)
    except torch.cuda.OutOfMemoryError:
        print("[OOM] Skipping batch during collation (insufficient GPU memory)")
        torch.cuda.empty_cache()
        return None

    try:
        with autocast(enabled=use_amp):
            logits = model.tabicl_model(repr_batch, ctx_batch, return_logits=True)
            logits = logits[:, : tgt_batch.shape[1], :]

            if not tgt_mask.any():
                return None

            loss = F.cross_entropy(logits[tgt_mask], tgt_batch[tgt_mask])
    except torch.cuda.OutOfMemoryError:
        print("[OOM] Skipping batch during forward pass (insufficient GPU memory)")
        torch.cuda.empty_cache()
        return None
    return loss

def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # 载入模型（包含冻结的 TabICL + 可训练的顶层）
    model = MantisTabICL(
        tabicl_checkpoint=args.tabicl_ckpt,
        mantis_checkpoint=args.mantis_ckpt,
        mantis_batch_size=32,
        device=device,
    ).to(device)
    model.train()
    if hasattr(model, "mantis_model"):
        model.mantis_model.eval()

    # 只微调 TabICL 最后 ICL 层
    for p in model.tabicl_model.parameters():
        p.requires_grad = False
    for p in model.tabicl_model.icl_predictor.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scaler = GradScaler(enabled=use_amp)

    reader = DataReader(UEA_data_path=args.uea_path, UCR_data_path=args.ucr_path, transform_ts_size=512)

    datasets = reader.dataset_list_uea + reader.dataset_list_ucr
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.repr_cache) if args.repr_cache else None

    for epoch in range(args.epochs):
        total_loss = 0.0
        num_steps = 0
        optimizer.zero_grad(set_to_none=True)
        batch_samples = []
        for name in tqdm(datasets, desc=f"Epoch {epoch+1}/{args.epochs}"):
            try:
                X_train, y_train = reader.read_dataset(name, "train")
                X_test, y_test = reader.read_dataset(name, "test")

                X_all = np.concatenate([X_train, X_test], axis=0)
                y_all = np.concatenate([y_train, y_test], axis=0)

                if np.unique(y_all).size >= 10:
                    continue

                if X_train.shape[0] < args.train_size:
                    print(
                        f"[SKIP] {name}: train split has only {X_train.shape[0]} rows, need at least {args.train_size}"
                    )
                    continue

                seq_tensor = torch.from_numpy(X_all).float().to(device)

                mantis_repr = maybe_load_cached_repr(cache_dir, name)
                if mantis_repr is None:
                    with torch.no_grad():
                        encoded = model._encode_with_mantis(seq_tensor)
                    mantis_repr = torch.as_tensor(encoded).cpu()
                    maybe_save_cached_repr(cache_dir, name, mantis_repr)

                mantis_repr = mantis_repr.to(device)
                try:
                    ctx_y, tgt_y = build_context_and_targets(y_train, y_test, args.train_size, device)
                except ValueError as err:
                    print(f"[SKIP] {name}: {err}")
                    continue

                if tgt_y.numel() == 0:
                    continue

                batch_samples.append({
                    "repr": mantis_repr,    
                    "ctx": ctx_y,
                    "tgt": tgt_y,
                })

                if len(batch_samples) == args.batch_size:
                    loss = run_batch(model, batch_samples, args.train_size, device, use_amp)
                    if loss is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        total_loss += loss.detach().item()
                        num_steps += 1
                    batch_samples.clear()

            except torch.cuda.OutOfMemoryError:
                print(f"[OOM] Skipping dataset {name} due to CUDA OOM")
                torch.cuda.empty_cache()
                continue

        if batch_samples:
            loss = run_batch(model, batch_samples, args.train_size, device, use_amp)
            if loss is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                total_loss += loss.detach().item()
                num_steps += 1
            batch_samples.clear()

        checkpoint = {
            "epoch": epoch + 1,
            "config": getattr(model, "tabicl_config", None),
            "tabicl_state_dict": model.tabicl_model.state_dict(),
            "mantis_state_dict": model.mantis_model.state_dict(),
            "mantis_tabicl_state_dict": model.state_dict(),
            "state_dict": model.tabicl_model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, save_dir / f"mantis_tabicl_ft_epoch{epoch+1}.ckpt")
        avg_loss = total_loss / max(1, num_steps)
        print(f"Epoch {epoch+1}: avg loss={avg_loss:.4f}")

if __name__ == "__main__":
    main()