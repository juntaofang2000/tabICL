import argparse, torch, torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tabicl.model.mantis_tabicl import MantisTabICL
from tabicl.prior.data_reader import DataReader

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
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--train-size", type=int, default=5, help="前多少样本当作上下文标签")
    return p

@torch.no_grad()
def pad_split(X, y, train_size):
    if y.ndim == 0:
        y = y.unsqueeze(0)

    y_train = y[:train_size]
    y_test = y[train_size:]  # 修改：直接取剩余部分，不需要指定结束索引
    return X, y_train, y_test

def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 载入模型（包含冻结的 TabICL + 可训练的顶层）
    model = MantisTabICL(
        tabicl_checkpoint=args.tabicl_ckpt,
        mantis_checkpoint=args.mantis_ckpt,
        mantis_batch_size=args.batch_size,
        device=device,
    ).to(device)
    model.train()

    # 只微调 TabICL 最后 ICL 层
    for p in model.tabicl_model.parameters():
        p.requires_grad = False
    for p in model.tabicl_model.icl_predictor.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    reader = DataReader(UEA_data_path=args.uea_path, UCR_data_path=args.ucr_path, transform_ts_size=512)

    datasets = reader.dataset_list_uea + reader.dataset_list_ucr
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        total_loss = 0.0
        num_steps = 0
        for name in tqdm(datasets, desc=f"Epoch {epoch+1}/{args.epochs}"):
            X_train, y_train = reader.read_dataset(name, "train")
            X_test, y_test = reader.read_dataset(name, "test")

            X_all = np.concatenate([X_train, X_test], axis=0)
            y_all = np.concatenate([y_train, y_test], axis=0)

            if np.unique(y_all).size >= 10:
                continue

            seq_tensor = torch.from_numpy(X_all).float()
            y_vec = torch.from_numpy(y_all).long()
            # if seq_tensor.ndim == 3:
            #     n, c, l = seq_tensor.shape
            #     seq_tensor = seq_tensor.reshape(n * c, l)
            #     y_vec = y_vec.repeat_interleave(c)
            # elif seq_tensor.ndim == 1:
            #     seq_tensor = seq_tensor.unsqueeze(0)
            seq_tensor = seq_tensor.to(device)
            y_tensor = y_vec.to(device)

            mantis_repr = model._encode_with_mantis(seq_tensor)
            mantis_repr = torch.as_tensor(mantis_repr).to(device)
            mantis_repr, ctx_y, tgt_y = pad_split(mantis_repr, y_tensor, args.train_size)

            mantis_repr = mantis_repr.unsqueeze(0)
            ctx_y = ctx_y.unsqueeze(0)
            tgt_y = tgt_y.unsqueeze(0)

            logits = model.tabicl_model(
                mantis_repr,
                ctx_y,
                return_logits=True,
            )
            logits = logits[:, : tgt_y.shape[1], :]
            pred = logits.flatten(end_dim=-2)
            true = tgt_y.long().flatten()
            loss = F.cross_entropy(pred, true)
            loss.backward()
            total_loss += loss.item()
            num_steps += 1

            if (len(datasets) * epoch + datasets.index(name) + 1) % args.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            save_dir / f"mantis_tabicl_ft_epoch{epoch+1}.ckpt",
        )
        avg_loss = total_loss / max(1, num_steps)
        print(f"Epoch {epoch+1}: avg loss={avg_loss:.4f}")

if __name__ == "__main__":
    main()