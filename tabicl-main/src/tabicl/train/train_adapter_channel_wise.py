import argparse
import os
import sys
import json
import torch
import numpy as np
import random
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
import copy

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from tabicl.model.mantis_tabicl import MantisTabICL, build_mantis_encoder
from tabicl.model.adapterOrign import CALDA_Adapter, DistributionDiversityLoss
from tabicl.prior.data_reader import DataReader
from tabicl.model.tabicl import TabICL
from tabicl.sklearn.classifier import TabICLClassifier

def load_dataset_names_from_file(filepath):
    """从结果文件读取所有数据集名称（每行格式：name: acc）"""
    names = []
    with open(filepath, "r") as f:
        for line in f:
            if ":" in line:
                name = line.split(":")[0].strip()
                if name:
                    names.append(name)
    return names

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _ensure_three_dim(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        return arr[None, None, :]
    if arr.ndim == 2:
        return arr[:, None, :]
    return arr

def resize_series(X, target_len=512):
    # X: (N, C, L)
    # Resize L to target_len using interpolation
    if X.shape[2] == target_len:
        return torch.from_numpy(X).float()
    
    X_tensor = torch.from_numpy(X).float()
    # F.interpolate expects (N, C, L)
    X_resized = torch.nn.functional.interpolate(X_tensor, size=target_len, mode='linear', align_corners=False)
    return X_resized

class MantisAdapterTabICL(nn.Module):
    def __init__(self, mantis_model, tabicl_model, adapter, mantis_batch_size=16):
        super().__init__()
        self.mantis_model = mantis_model
        self.tabicl_model = tabicl_model
        self.adapter = adapter
        self.mantis_batch_size = mantis_batch_size
        
        # Freeze Mantis and TabICL
        for param in self.mantis_model.parameters():
            param.requires_grad = False
        for param in self.tabicl_model.parameters():
            param.requires_grad = False
            
    def train(self, mode=True):
        """
        Override train mode to keep Mantis and TabICL in eval mode.
        """
        super().train(mode)
        self.mantis_model.eval()
        self.tabicl_model.eval()
        return self

    def forward(self, X, y_train, return_logits=True):
        # X: (Batch, Samples, Channels, Length)
        B, N, C, L = X.shape
        
        # 1. Encode with Mantis
        X_in = X.reshape(-1, L).unsqueeze(1) # (B*N*C, 1, L)
        
        # Batch processing for Mantis to avoid OOM
        mantis_outs = []
        total_samples = X_in.size(0)
        
        # Get device from mantis model
        device = next(self.mantis_model.parameters()).device
        
        with torch.no_grad():
            for i in range(0, total_samples, self.mantis_batch_size):
                batch = X_in[i : i + self.mantis_batch_size]
                batch = batch.to(device)
                out = self.mantis_model(batch)
                mantis_outs.append(out)
            mantis_out = torch.cat(mantis_outs, dim=0) # (B*N*C, d_mantis)
            
        # 2. Adapter
        # Modified: Channel-wise processing
        # mantis_out is (B*N*C, d_mantis)
        # Reshape to (B*N*C, 1, d_mantis) for adapter input
        mantis_out_reshaped = mantis_out.unsqueeze(1)
        
        if self.adapter is not None:
            # Batch execution for Adapter
            adapter_outs = []
            adapter_batch_size = 32
            for i in range(0, mantis_out_reshaped.size(0), adapter_batch_size):
                batch_slice = mantis_out_reshaped[i : i + adapter_batch_size]
                out_slice = self.adapter(batch_slice) # (Batch_slice, d_adapter)
                adapter_outs.append(out_slice)
            adapter_out_flat = torch.cat(adapter_outs, dim=0) # (B*N*C, d_adapter)
        else:
            adapter_out_flat = mantis_out # (B*N*C, d_mantis)
        
        # Reshape and Concat
        # We want (B, N, C * d_adapter)
        d_out = adapter_out_flat.shape[-1]
        adapter_out = adapter_out_flat.reshape(B, N, C, d_out)
        adapter_out = adapter_out.reshape(B, N, C * d_out)
        
        # 3. TabICL
        tabicl_in = adapter_out # (B, N, C*d_out)
        out = self.tabicl_model(tabicl_in, y_train, return_logits=return_logits)
        
        return out, adapter_out

    def get_adapter_output(self, X):
        """
        Get embeddings from Mantis + Adapter without passing through TabICL.
        Useful for applying augmentations before TabICL.
        """
        # X: (Batch, Samples, Channels, Length)
        B, N, C, L = X.shape
        
        # 1. Encode with Mantis
        X_in = X.reshape(-1, L).unsqueeze(1) # (B*N*C, 1, L)
        
        # Batch processing for Mantis to avoid OOM
        mantis_outs = []
        total_samples = X_in.size(0)
        
        # Get device from mantis model
        device = next(self.mantis_model.parameters()).device
        
        with torch.no_grad():
            for i in range(0, total_samples, self.mantis_batch_size):
                batch = X_in[i : i + self.mantis_batch_size]
                batch = batch.to(device)
                out = self.mantis_model(batch)
                mantis_outs.append(out)
            mantis_out = torch.cat(mantis_outs, dim=0)
            
        # 2. Adapter
        # Modified: Channel-wise processing
        mantis_out_reshaped = mantis_out.unsqueeze(1) # (B*N*C, 1, d_mantis)
        
        if self.adapter is not None:
            # Batch execution for Adapter
            adapter_outs = []
            adapter_batch_size = 32
            for i in range(0, mantis_out_reshaped.size(0), adapter_batch_size):
                batch_slice = mantis_out_reshaped[i : i + adapter_batch_size]
                out_slice = self.adapter(batch_slice)
                adapter_outs.append(out_slice)
            adapter_out_flat = torch.cat(adapter_outs, dim=0)
        else:
            adapter_out_flat = mantis_out
            
        # Reshape and Concat
        d_out = adapter_out_flat.shape[-1]
        adapter_out = adapter_out_flat.reshape(B, N, C, d_out)
        adapter_out = adapter_out.reshape(B, N, C * d_out)
            
        return adapter_out

def augment_batch(X, y_support, y_query, device, n_classes):
    """
    Apply TabICLClassifier-like augmentations:
    1. Normalization (StandardScaler as proxy for PowerTransform)
    2. Feature Shuffling
    3. Class Shift
    """
    # X: (B, N_total, D)
    # y_support: (B, N_support)
    # y_query: (B, N_query)
    
    # 1. Normalization (Randomly apply)
    if torch.rand(1).item() > 0.5:
        mean = X.mean(dim=1, keepdim=True)
        std = X.std(dim=1, keepdim=True) + 1e-5
        X = (X - mean) / std
    
    # 2. Feature Shuffling
    D = X.shape[-1]
    perm = torch.randperm(D, device=device)
    X = X[..., perm]
    
    # 3. Class Shift
    shift = torch.randint(0, n_classes, (1,)).item()
    y_support = (y_support + shift) % n_classes
    y_query = (y_query + shift) % n_classes
    
    return X, y_support, y_query


def get_embeddings(model, X_data, device, batch_size=64):
    """
    Helper to get embeddings from Mantis + Adapter.
    X_data: (N, C, L) tensor or numpy array
    """
    if isinstance(X_data, np.ndarray):
        X_data = torch.from_numpy(X_data).float()
        
    embs = []
    N = X_data.size(0)
    
    model.eval()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = X_data[i:i+batch_size].to(device)
            B, C, L = batch.shape
            batch_in = batch.reshape(-1, L).unsqueeze(1)
            
            # Process batch_in in smaller chunks for Mantis to avoid OOM
            mantis_outs = []
            sub_batch_size = 16 
            for j in range(0, batch_in.size(0), sub_batch_size):
                sub_batch = batch_in[j : j + sub_batch_size]
                mantis_outs.append(model.mantis_model(sub_batch))
            mantis_out = torch.cat(mantis_outs, dim=0) # (B*C, Mantis_Dim)
            
            # Adapter
            # Modified: Channel-wise processing
            mantis_out_reshaped = mantis_out.unsqueeze(1) # (B*C, 1, Mantis_Dim)
            
            if model.adapter is not None:
                adapter_out_flat = model.adapter(mantis_out_reshaped) # (B*C, TabICL_Dim)
            else:
                adapter_out_flat = mantis_out
            
            d_out = adapter_out_flat.shape[-1]
            adapter_out = adapter_out_flat.reshape(B, C, d_out)
            adapter_out = adapter_out.reshape(B, C * d_out)
                
            embs.append(adapter_out.cpu().numpy())
            
    return np.concatenate(embs, axis=0)

def load_dataset_data(reader, dataset_name):
    try:
        X_train_raw, y_train_raw = reader.read_dataset(dataset_name, which_set="train")
        X_test_raw, y_test_raw = reader.read_dataset(dataset_name, which_set="test")
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None, None, None, None

    X_train = _ensure_three_dim(X_train_raw)
    X_test = _ensure_three_dim(X_test_raw)
    
    X_train = resize_series(X_train, target_len=512)
    X_test = resize_series(X_test, target_len=512)
    
    y_train = torch.from_numpy(y_train_raw).long()
    y_test = torch.from_numpy(y_test_raw).long()
    
    return X_train, y_train, X_test, y_test


def _build_meta_tasks(model, batch_datasets, device, args):
    """
    Build augmented meta-learning tasks for adapter training/eval.

    Returns:
        X_aug_batch: (T, L, D)
        y_sup_aug_batch: (T, n_support)
        y_qry_aug_batch: (T, qry_len)
        valid_mask_batch: (T, qry_len)
        n_support: int
        qry_len: int
    """
    # 1) Determine n_support (context size) for this batch
    min_train_len = min(d[0].size(0) for d in batch_datasets)
    n_support = min(args.train_size, min_train_len)
    if n_support < 1:
        return None

    X_seq_list = []
    y_sup_mapped_list = []
    y_qry_mapped_list = []
    valid_mask_list = []

    for X_train, y_train, X_test, y_test in batch_datasets:
        X_sup = X_train[:n_support]
        y_sup = y_train[:n_support]

        X_qry = torch.cat([X_train[n_support:], X_test], dim=0)
        y_qry = torch.cat([y_train[n_support:], y_test], dim=0)

        X_seq = torch.cat([X_sup, X_qry], dim=0)
        X_seq_list.append(X_seq)

        # Map classes based on support-set classes
        unique_classes, inverse_indices = torch.unique(y_sup, return_inverse=True)
        y_sup_mapped = inverse_indices.to(device)

        max_label = max(y_sup.max(), y_qry.max()).item()
        mapper = torch.full((max_label + 1,), -1, dtype=torch.long, device=device)
        mapper[unique_classes.to(device)] = torch.arange(len(unique_classes), device=device)

        y_qry_mapped = mapper[y_qry.to(device)]
        valid_mask = (y_qry_mapped != -1)

        # Safe mapping for augmentation (replace -1 with 0)
        y_qry_mapped_safe = y_qry_mapped.clone()
        y_qry_mapped_safe[~valid_mask] = 0

        y_sup_mapped_list.append(y_sup_mapped)
        y_qry_mapped_list.append(y_qry_mapped_safe)
        valid_mask_list.append(valid_mask)

    # 2) Collate: truncate to min length in batch and args.max_icl_len
    min_len = min(x.size(0) for x in X_seq_list)
    target_len = min(min_len, args.max_icl_len)
    if target_len <= n_support:
        return None

    adapter_out_list = []
    y_sup_batch_list = []
    y_qry_batch_list = []
    mask_batch_list = []

    for i in range(len(batch_datasets)):
        x_item = X_seq_list[i][:target_len]
        emb = model.get_adapter_output(x_item.unsqueeze(0))  # (1, L, D)
        adapter_out_list.append(emb.squeeze(0))

        y_sup_batch_list.append(y_sup_mapped_list[i])

        qry_len = target_len - n_support
        y_qry_batch_list.append(y_qry_mapped_list[i][:qry_len])
        mask_batch_list.append(valid_mask_list[i][:qry_len])

    adapter_out = torch.stack(adapter_out_list)  # (B, L, D)

    # 3) Augment
    aug_emb_list = []
    aug_y_sup_list = []
    aug_y_qry_list = []
    aug_mask_list = []

    for i in range(len(batch_datasets)):
        emb = adapter_out[i].unsqueeze(0)  # (1, L, D)
        y_sup = y_sup_batch_list[i].unsqueeze(0)  # (1, n_support)
        y_qry = y_qry_batch_list[i].unsqueeze(0)  # (1, qry_len)
        mask = mask_batch_list[i]

        n_classes = y_sup.max().item() + 1
        # Handle degenerate tasks
        if n_classes < 1:
            continue

        for _ in range(args.n_augmentations):
            X_aug, y_sup_aug, y_qry_aug = augment_batch(emb, y_sup, y_qry, device, n_classes)
            aug_emb_list.append(X_aug.squeeze(0))
            aug_y_sup_list.append(y_sup_aug.squeeze(0))
            aug_y_qry_list.append(y_qry_aug.squeeze(0))
            aug_mask_list.append(mask)

    if not aug_emb_list:
        return None

    X_aug_batch = torch.stack(aug_emb_list)
    y_sup_aug_batch = torch.stack(aug_y_sup_list)
    y_qry_aug_batch = torch.stack(aug_y_qry_list)
    valid_mask_batch = torch.stack(aug_mask_list)

    qry_len = y_qry_aug_batch.size(1)
    return X_aug_batch, y_sup_aug_batch, y_qry_aug_batch, valid_mask_batch, n_support, qry_len


def _compute_meta_loss(model, criterion, X_aug_batch, y_sup_aug_batch, y_qry_aug_batch, valid_mask_batch):
    """Compute averaged meta loss across tasks in X_aug_batch."""
    total_loss = 0.0
    denom = 0

    for i in range(X_aug_batch.size(0)):
        X_in = X_aug_batch[i].unsqueeze(0)  # (1, L, D)
        y_sup_in = y_sup_aug_batch[i].unsqueeze(0)  # (1, n_support)
        y_qry_in = y_qry_aug_batch[i].unsqueeze(0)  # (1, qry_len)
        mask_in = valid_mask_batch[i]  # (qry_len,)

        if not mask_in.any():
            continue

        logits = model.tabicl_model(X_in, y_sup_in, return_logits=True)

        qry_len = y_qry_in.size(1)
        if logits.size(1) == X_in.size(1):
            logits_qry = logits[:, -qry_len:, :]
        else:
            logits_qry = logits

        logits_flat = logits_qry.reshape(-1, logits_qry.size(-1))
        y_flat = y_qry_in.reshape(-1)
        mask_flat = mask_in.reshape(-1)

        loss = criterion(logits_flat[mask_flat], y_flat[mask_flat])
        total_loss += loss.item()
        denom += 1

    if denom == 0:
        return None
    return total_loss / denom


def train_step(model, optimizer, criterion, batch_datasets, device, args):
    model.train()
    optimizer.zero_grad()

    built = _build_meta_tasks(model, batch_datasets, device, args)
    if built is None:
        return 0.0
    X_aug_batch, y_sup_aug_batch, y_qry_aug_batch, valid_mask_batch, _, _ = built

    # Forward/backward across tasks (manual to keep per-task class counts independent)
    total_loss = 0.0
    denom = 0

    for i in range(X_aug_batch.size(0)):
        X_in = X_aug_batch[i].unsqueeze(0)
        y_sup_in = y_sup_aug_batch[i].unsqueeze(0)
        y_qry_in = y_qry_aug_batch[i].unsqueeze(0)
        mask_in = valid_mask_batch[i]

        if not mask_in.any():
            continue

        logits = model.tabicl_model(X_in, y_sup_in, return_logits=True)

        qry_len = y_qry_in.size(1)
        if logits.size(1) == X_in.size(1):
            logits_qry = logits[:, -qry_len:, :]
        else:
            logits_qry = logits

        logits_flat = logits_qry.reshape(-1, logits_qry.size(-1))
        y_flat = y_qry_in.reshape(-1)
        mask_flat = mask_in.reshape(-1)

        loss = criterion(logits_flat[mask_flat], y_flat[mask_flat])
        loss = loss / X_aug_batch.size(0)

        # Ensure graph exists even if something odd happens
        if not loss.requires_grad:
            dummy = sum(p.sum() for p in model.adapter.parameters()) * 0.0
            loss = loss + dummy

        loss.backward()
        total_loss += loss.item()
        denom += 1

    optimizer.step()
    return total_loss if denom > 0 else 0.0


@torch.no_grad()
def validate_step(model, criterion, batch_datasets, device, args):
    model.eval()
    built = _build_meta_tasks(model, batch_datasets, device, args)
    if built is None:
        return None
    X_aug_batch, y_sup_aug_batch, y_qry_aug_batch, valid_mask_batch, _, _ = built
    return _compute_meta_loss(model, criterion, X_aug_batch, y_sup_aug_batch, y_qry_aug_batch, valid_mask_batch)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabicl_ckpt", type=str, default="/data0/fangjuntao2025/tabicl-main/tabICLOrignCheckpoint/tabicl-classifier-v1.1-0506.ckpt")
    parser.add_argument("--mantis_ckpt", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/Models/Mantis/Mantis_cheickpoint/")
    parser.add_argument("--uea_path", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/")
    parser.add_argument("--ucr_path", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--div_weight", type=float, default=0.1)
    parser.add_argument("--max_icl_len", type=int, default=512, help="Max sequence length for ICL training to avoid OOM")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_adapter", action="store_true", help="Disable adapter and use raw Mantis embeddings")
    parser.add_argument("--mantis_batch_size", type=int, default=16, help="Batch size for Mantis encoder")
    parser.add_argument("--meta_batch_size", type=int, default=8, help="Number of datasets per training step")
    parser.add_argument("--train_size", type=int, default=100, help="Number of support samples (context size)")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--seed", type=int, default=666, help="Random seed for reproducibility")
    parser.add_argument("--n_augmentations", type=int, default=5, help="Number of augmentations per dataset")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio (held-out datasets) during adapter pretraining")
    parser.add_argument("--ckpt_dir", type=str, default="/data0/fangjuntao2025/tabicl-main/checkpoints/mantis_adapter_pretrain", help="Directory to save adapter checkpoints")
    parser.add_argument("--ckpt_prefix", type=str, default="adapter", help="Checkpoint filename prefix")
    parser.add_argument("--save_last", action="store_true", help="Also save last checkpoint each epoch")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device(args.device)
    if device.type == 'cuda' and device.index is not None:
        torch.cuda.set_device(device)
    
    # 1. Build Models
    print("Loading models...")
    # Load Mantis
    mantis_model = build_mantis_encoder(args.mantis_ckpt, device=device)
    
    # Load TabICL (for training adapter)
    tabicl_state = torch.load(args.tabicl_ckpt, map_location="cpu")
    tabicl_model = TabICL(**tabicl_state["config"])
    tabicl_model.load_state_dict(tabicl_state["state_dict"])
    tabicl_model.to(device)
    
    # Initialize Adapter
    tabicl_dim = 256
    
    mantis_dim = mantis_model.hidden_dim
    
    print(f"Mantis Dim: {mantis_dim}, TabICL Dim: {tabicl_dim}")
    
    if args.no_adapter:
        adapter = None
    else:
        adapter = CALDA_Adapter(mantis_emb_dim=mantis_dim, tabicl_input_dim=tabicl_dim).to(device)
    
    model = MantisAdapterTabICL(mantis_model, tabicl_model, adapter, mantis_batch_size=args.mantis_batch_size).to(device)
    
    reader = DataReader(UEA_data_path=args.uea_path, UCR_data_path=args.ucr_path)
    
    # Combine UCR and UEA datasets
    datasets = sorted(reader.dataset_list_ucr)

    # Split datasets into train/val for meta-pretraining
    train_datasets = datasets
    val_datasets = []
    if not args.no_adapter and args.val_ratio > 0:
        rng = random.Random(args.seed)
        shuffled = datasets.copy()
        rng.shuffle(shuffled)
        val_count = max(1, int(len(shuffled) * args.val_ratio)) if len(shuffled) > 1 else 0
        val_datasets = sorted(shuffled[:val_count])
        train_datasets = sorted(shuffled[val_count:]) if val_count > 0 else datasets
        print(f"Meta split: train={len(train_datasets)}, val={len(val_datasets)} (val_ratio={args.val_ratio})")
    
    # --- Pretraining Phase ---
    if not args.no_adapter:
        print(f"Starting Pretraining on {len(train_datasets)} datasets for {args.epochs} epochs...")
        optimizer = optim.AdamW(model.adapter.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        os.makedirs(args.ckpt_dir, exist_ok=True)
        best_val = float("inf")
        best_epoch = -1
        
        for epoch in range(args.epochs):
            random.shuffle(train_datasets)
            epoch_loss = 0.0
            count = 0
            
            # Create batches
            num_batches = (len(train_datasets) + args.meta_batch_size - 1) // args.meta_batch_size
            
            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{args.epochs}")
            for i in pbar:
                batch_names = train_datasets[i*args.meta_batch_size : (i+1)*args.meta_batch_size]
                
                batch_data = []
                for name in batch_names:
                    X_tr, y_tr, X_te, y_te = load_dataset_data(reader, name)
                    if X_tr is not None:
                        batch_data.append((X_tr, y_tr, X_te, y_te))
                
                if not batch_data:
                    continue
                
                try:
                    loss = train_step(model, optimizer, criterion, batch_data, device, args)
                    epoch_loss += loss
                    count += 1
                    pbar.set_postfix({'avg_loss': epoch_loss / count if count > 0 else 0})
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nSkipping batch due to OOM")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

            avg_train_loss = epoch_loss / count if count > 0 else 0.0

            # Validation
            avg_val_loss = None
            if val_datasets:
                val_loss_sum = 0.0
                val_steps = 0
                num_val_batches = (len(val_datasets) + args.meta_batch_size - 1) // args.meta_batch_size
                for i in tqdm(range(num_val_batches), desc=f"Val {epoch+1}/{args.epochs}", leave=False):
                    batch_names = val_datasets[i*args.meta_batch_size : (i+1)*args.meta_batch_size]
                    batch_data = []
                    for name in batch_names:
                        X_tr, y_tr, X_te, y_te = load_dataset_data(reader, name)
                        if X_tr is not None:
                            batch_data.append((X_tr, y_tr, X_te, y_te))
                    if not batch_data:
                        continue
                    try:
                        vloss = validate_step(model, criterion, batch_data, device, args)
                        if vloss is None:
                            continue
                        val_loss_sum += vloss
                        val_steps += 1
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            continue
                        raise e
                avg_val_loss = val_loss_sum / val_steps if val_steps > 0 else None

            # Save checkpoints
            ckpt_common = {
                "epoch": epoch,
                "adapter_state_dict": copy.deepcopy(model.adapter.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "config": vars(args),
                "tabicl_ckpt": args.tabicl_ckpt,
                "mantis_ckpt": args.mantis_ckpt,
                "seed": args.seed,
            }

            if args.save_last:
                last_path = os.path.join(args.ckpt_dir, f"{args.ckpt_prefix}_last.pt")
                torch.save(ckpt_common, last_path)

            if avg_val_loss is not None and avg_val_loss < best_val:
                best_val = avg_val_loss
                best_epoch = epoch
                best_path = os.path.join(args.ckpt_dir, f"{args.ckpt_prefix}_best.pt")
                ckpt_best = dict(ckpt_common)
                ckpt_best["best_val_loss"] = best_val
                ckpt_best["best_epoch"] = best_epoch
                torch.save(ckpt_best, best_path)
                print(f"Saved best checkpoint: {best_path} (val_loss={best_val:.6f})")
            elif not val_datasets:
                # If no validation, save best by train loss
                train_path = os.path.join(args.ckpt_dir, f"{args.ckpt_prefix}_best_trainloss.pt")
                torch.save(ckpt_common, train_path)
                print(f"Saved checkpoint (no val): {train_path} (train_loss={avg_train_loss:.6f})")
            else:
                if avg_val_loss is not None:
                    print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}, best_val={best_val:.6f}")
                else:
                    print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.6f}, val_loss=N/A, best_val={best_val:.6f}")
        
        print("Pretraining finished.")
    
    # --- Evaluation Phase ---
    print("Starting Evaluation...")
    results = {}
    
    # Initialize Classifier for evaluation
    clf = TabICLClassifier(
        model_path=args.tabicl_ckpt,
        n_estimators=32,
        device=device,
        verbose=False,
        mantis_checkpoint=None,
        batch_size=8,
    )
    
    all_datasets = sorted(reader.dataset_list_ucr + reader.dataset_list_uea)
    for dataset_name in tqdm(all_datasets, desc="Evaluating"):
        try:
            X_train, y_train, X_test, y_test = load_dataset_data(reader, dataset_name)
            if X_train is None:
                continue
                
            # Extract embeddings
            X_train_emb = get_embeddings(model, X_train, device)
            X_test_emb = get_embeddings(model, X_test, device)
            
            # Fit and Predict
            clf.fit(X_train_emb, y_train.numpy())
            y_pred = clf.predict(X_test_emb)
            acc = np.mean(y_pred == y_test.numpy())
            
            results[dataset_name] = acc
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nSkipping {dataset_name} due to OOM")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
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
        
    print(f"\nOverall Average Accuracy: {np.mean(list(results.values())):.4f}")

    if args.output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        
        # Save structured results separating UEA and UCR
        structured_results = {
            "UEA": uea_results,
            "UCR": ucr_results
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(structured_results, f, indent=4)
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
