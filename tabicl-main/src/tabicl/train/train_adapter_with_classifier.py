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
from tabicl.model.adapter import StructuralCausalAdapter, ICLAlignmentLoss
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
    def __init__(self, mantis_model, tabicl_model, adapter, projector=None, mantis_batch_size=16):
        super().__init__()
        self.mantis_model = mantis_model
        self.tabicl_model = tabicl_model
        self.adapter = adapter
        self.projector = projector
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
            mantis_out = torch.cat(mantis_outs, dim=0)
            
        # 2. Adapter
        mantis_out_reshaped = mantis_out.reshape(B*N, C, -1)
        
        if self.adapter is not None:
            # Batch execution for Adapter
            adapter_outs = []
            adapter_batch_size = 32
            for i in range(0, mantis_out_reshaped.size(0), adapter_batch_size):
                batch_slice = mantis_out_reshaped[i : i + adapter_batch_size]
                out_slice = self.adapter(batch_slice)
                if isinstance(out_slice, (tuple, list)):
                    out_slice = out_slice[0]
                if out_slice.dim() == 3:
                    out_slice = out_slice.reshape(out_slice.size(0), -1)
                if self.projector is not None:
                    out_slice = self.projector(out_slice)
                adapter_outs.append(out_slice)
            adapter_out = torch.cat(adapter_outs, dim=0)
        else:
            adapter_out = mantis_out_reshaped.reshape(B*N, -1)
        
        # 3. TabICL
        tabicl_in = adapter_out.reshape(B, N, -1)
        out = self.tabicl_model(tabicl_in, y_train, return_logits=return_logits)
        
        return out, adapter_out

    def get_adapter_output(self, X, return_aux: bool = False):
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
        mantis_out_reshaped = mantis_out.reshape(B*N, C, -1)
        
        aux_sum = None
        aux_count = 0

        if self.adapter is not None:
            # Batch execution for Adapter
            adapter_outs = []
            adapter_batch_size = 32
            for i in range(0, mantis_out_reshaped.size(0), adapter_batch_size):
                batch_slice = mantis_out_reshaped[i : i + adapter_batch_size]
                out_slice = self.adapter(batch_slice)
                aux_slice = None
                if isinstance(out_slice, (tuple, list)):
                    out_slice, aux_slice = out_slice
                if out_slice.dim() == 3:
                    out_slice = out_slice.reshape(out_slice.size(0), -1)
                if self.projector is not None:
                    out_slice = self.projector(out_slice)
                adapter_outs.append(out_slice)

                if return_aux and aux_slice is not None:
                    # Only aggregate scalar losses here; adjacency is global and can be logged elsewhere.
                    indep = aux_slice.get("independence_loss", None)
                    sparse = aux_slice.get("sparsity_loss", None)
                    if indep is not None and sparse is not None:
                        if aux_sum is None:
                            aux_sum = {
                                "independence_loss": indep,
                                "sparsity_loss": sparse,
                            }
                        else:
                            aux_sum["independence_loss"] = aux_sum["independence_loss"] + indep
                            aux_sum["sparsity_loss"] = aux_sum["sparsity_loss"] + sparse
                        aux_count += 1
            adapter_out = torch.cat(adapter_outs, dim=0)
        else:
            adapter_out = mantis_out_reshaped.reshape(B*N, -1)

        out = adapter_out.reshape(B, N, -1)
        if not return_aux:
            return out

        if aux_sum is None or aux_count == 0:
            aux = {
                "independence_loss": torch.tensor(0.0, device=out.device, requires_grad=True),
                "sparsity_loss": torch.tensor(0.0, device=out.device, requires_grad=True),
            }
        else:
            aux = {
                "independence_loss": aux_sum["independence_loss"] / float(aux_count),
                "sparsity_loss": aux_sum["sparsity_loss"] / float(aux_count),
            }
        return out, aux

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
            mantis_out_reshaped = mantis_out.reshape(B, C, -1)
            if model.adapter is not None:
                adapter_out = model.adapter(mantis_out_reshaped)
                if isinstance(adapter_out, (tuple, list)):
                    adapter_out = adapter_out[0]
                if adapter_out.dim() == 3:
                    adapter_out = adapter_out.reshape(adapter_out.size(0), -1)
                if getattr(model, "projector", None) is not None:
                    adapter_out = model.projector(adapter_out)
            else:
                adapter_out = mantis_out_reshaped.reshape(B, -1)
                
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

def train_step(model, optimizer, icl_loss_fn, batch_datasets, device, args):
    model.train()
    optimizer.zero_grad()
    
    # 1. Determine n_support (context size) for this batch
    # Use args.train_size but capped by the smallest training set in the batch
    min_train_len = min(d[0].size(0) for d in batch_datasets)
    n_support = min(args.train_size, min_train_len)
    
    # Ensure n_support is at least 1
    if n_support < 1:
        return None

    X_seq_list = []
    y_sup_mapped_list = []
    y_qry_mapped_list = []
    valid_mask_list = []
    
    for X_train, y_train, X_test, y_test in batch_datasets:
        # Construct sequence
        # Support: X_train[:n_support]
        # Query: X_train[n_support:] + X_test
        
        X_sup = X_train[:n_support]
        y_sup = y_train[:n_support]
        
        # Remaining training data + test data
        X_qry = torch.cat([X_train[n_support:], X_test], dim=0)
        y_qry = torch.cat([y_train[n_support:], y_test], dim=0)
        
        X_seq = torch.cat([X_sup, X_qry], dim=0)
        # y_seq is not strictly needed as one tensor, we process sup and qry separately for mapping
        
        X_seq_list.append(X_seq)
        
        # Mapping (needed for TabICL/Loss)
        # We map based on Support set classes
        unique_classes, inverse_indices = torch.unique(y_sup, return_inverse=True)
        y_sup_mapped = inverse_indices.to(device)
        
        # Map query labels
        max_label = max(y_sup.max(), y_qry.max()).item()
        mapper = torch.full((max_label + 1,), -1, dtype=torch.long, device=device)
        mapper[unique_classes] = torch.arange(len(unique_classes), device=device)
        
        y_qry_mapped = mapper[y_qry.to(device)]
        valid_mask = (y_qry_mapped != -1)
        
        # Safe mapping for augmentation (replace -1 with 0)
        y_qry_mapped_safe = y_qry_mapped.clone()
        y_qry_mapped_safe[~valid_mask] = 0
        
        y_sup_mapped_list.append(y_sup_mapped)
        y_qry_mapped_list.append(y_qry_mapped_safe)
        valid_mask_list.append(valid_mask)

    # 2. Collate (Truncate to min length in batch, and max_icl_len)
    min_len = min(x.size(0) for x in X_seq_list)
    target_len = min(min_len, args.max_icl_len)
    
    # Ensure target_len > n_support (so we have at least some query samples)
    if target_len <= n_support:
        return None

    adapter_out_list = []
    y_sup_batch_list = []
    y_qry_batch_list = []
    mask_batch_list = []
    
    indep_losses = []
    sparse_losses = []

    for i in range(len(batch_datasets)):
        # Truncate X and get embeddings individually to handle variable channels
        x_item = X_seq_list[i][:target_len]
        emb_aux = model.get_adapter_output(x_item.unsqueeze(0), return_aux=True)  # (1, L, D), aux
        emb, aux = emb_aux
        adapter_out_list.append(emb.squeeze(0))
        indep_losses.append(aux["independence_loss"])
        #sparse_losses.append(aux["sparsity_loss"])

        # y_sup is fixed size n_support
        y_sup_batch_list.append(y_sup_mapped_list[i])
        
        # y_qry needs to be truncated. 
        # The query part starts at n_support.
        # Total length is target_len.
        # So query length is target_len - n_support.
        qry_len = target_len - n_support
        y_qry_batch_list.append(y_qry_mapped_list[i][:qry_len])
        mask_batch_list.append(valid_mask_list[i][:qry_len])

    adapter_out = torch.stack(adapter_out_list) # (B, L, D)
    
    # Augmentation loop
    aug_emb_list = []
    aug_y_sup_list = []
    aug_y_qry_list = []
    aug_mask_list = []
    
    for i in range(len(batch_datasets)):
        emb = adapter_out[i].unsqueeze(0) # (1, L, D)
        y_sup = y_sup_batch_list[i].unsqueeze(0) # (1, n_support)
        y_qry = y_qry_batch_list[i].unsqueeze(0) # (1, qry_len)
        mask = mask_batch_list[i]
        
        # Determine n_classes for this task
        n_classes = y_sup.max().item() + 1
        
        # Multiple augmentations
        for _ in range(args.n_augmentations):
            X_aug, y_sup_aug, y_qry_aug = augment_batch(
                emb, y_sup, y_qry, device, n_classes
            )
            
            aug_emb_list.append(X_aug.squeeze(0))
            aug_y_sup_list.append(y_sup_aug.squeeze(0))
            aug_y_qry_list.append(y_qry_aug.squeeze(0))
            aug_mask_list.append(mask)
        
    X_aug_batch = torch.stack(aug_emb_list)
    y_sup_aug_batch = torch.stack(aug_y_sup_list)
    y_qry_aug_batch = torch.stack(aug_y_qry_list)
    valid_mask_batch = torch.stack(aug_mask_list)
    
    # ProtoNet + KL (ICLAlignmentLoss) over augmented episodes
    total_loss = 0.0
    used = 0
    for i in range(X_aug_batch.size(0)):
        X_full = X_aug_batch[i]  # (L, D)
        y_sup = y_sup_aug_batch[i]  # (n_support,)
        y_qry = y_qry_aug_batch[i]  # (qry_len,)
        mask_in = valid_mask_batch[i]  # (qry_len,)

        if not mask_in.any():
            continue

        z_support = X_full[:n_support]  # (n_support, D)
        z_query_all = X_full[n_support:]  # (qry_len, D)
        z_query = z_query_all[mask_in]
        y_query = y_qry[mask_in]

        if z_query.size(0) < 1:
            continue

        loss_i = icl_loss_fn.forward_episode(z_support, y_sup, z_query, y_query)
        total_loss = total_loss + loss_i
        used += 1

    if used == 0:
        return None

    proto_loss_tensor = total_loss / used
    loss = proto_loss_tensor

    # Add causal auxiliary losses from StructuralCausalAdapter (averaged over datasets in meta-batch).
    if len(indep_losses) > 0:
        indep_mean_tensor = torch.stack(indep_losses).mean()
        loss = loss + args.indep_weight * indep_mean_tensor
    else:
        indep_mean_tensor = torch.tensor(0.0, device=proto_loss_tensor.device)
    # if len(sparse_losses) > 0:
    #     sparse_mean_tensor = torch.stack(sparse_losses).mean()
    #     loss = loss + args.sparsity_weight * sparse_mean_tensor
    # else:
    #     sparse_mean_tensor = torch.tensor(0.0, device=proto_loss_tensor.device)

    loss.backward()  
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 防止梯度爆炸
    # 打印日志时，分别打印 cls 和 kl，看看是谁在捣乱
    optimizer.step()
    return {
        "loss": float(loss.detach().item()),
        "proto_loss": float(proto_loss_tensor.detach().item()),
        "independence_loss": float(indep_mean_tensor.detach().item()),
        #"sparsity_loss": float(sparse_mean_tensor.detach().item()),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabicl_ckpt", type=str, default="/data0/fangjuntao2025/tabicl-main/tabICLOrignCheckpoint/tabicl-classifier-v1.1-0506.ckpt")
    parser.add_argument("--mantis_ckpt", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/Models/Mantis/Mantis_cheickpoint/")
    parser.add_argument("--uea_path", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/")
    parser.add_argument("--ucr_path", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/")
    parser.add_argument("--epochs", type=int, default=20)
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
    parser.add_argument("--num_latents", type=int, default=10, help="Number of causal latents extracted by cross-attention")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads in the adapter")
    parser.add_argument("--adapter_norm", type=str, default="bn", choices=["bn", "ln", "none"], help="Normalization for adapter outputs")
    parser.add_argument("--kl_weight", type=float, default=1e-3, help="KL regularization weight toward N(0, I)")

    # StructuralCausalAdapter hyperparams
    parser.add_argument("--independence", type=str, default="orth", choices=["orth", "hsic"], help="Independence regularizer among K latents")
    parser.add_argument("--hsic_kernel", type=str, default="rbf", choices=["rbf", "linear"], help="HSIC kernel type (if independence=hsic)")
    parser.add_argument("--hsic_sigma", type=float, default=1.0, help="RBF sigma for HSIC (if independence=hsic)")
    parser.add_argument("--gumbel_tau", type=float, default=1.0, help="Gumbel-Sigmoid temperature for adjacency sampling")
    parser.add_argument("--gumbel_hard", action="store_true", help="Use hard (straight-through) adjacency sampling")
    parser.add_argument("--allow_self_edges", action="store_true", help="Allow self edges in adjacency matrix")
    parser.add_argument("--sparsity_on", type=str, default="prob", choices=["prob", "sample"], help="Apply sparsity loss on prob or sampled adjacency")
    parser.add_argument("--indep_weight", type=float, default=1e-2, help="Weight for independence_loss")
    parser.add_argument("--sparsity_weight", type=float, default=1e-3, help="Weight for sparsity_loss")
    
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
        projector = None
    else:
        adapter = StructuralCausalAdapter(
            emb_dim=mantis_dim,
            num_latents=args.num_latents,
            num_heads=args.num_heads,
            dropout=0.0,
            norm=args.adapter_norm,
            use_affine_norm=False,
            independence=args.independence,
            hsic_kernel=args.hsic_kernel,
            hsic_sigma=args.hsic_sigma,
            gumbel_tau=args.gumbel_tau,
            gumbel_hard=args.gumbel_hard,
            allow_self_edges=args.allow_self_edges,
            sparsity_on=args.sparsity_on,
        ).to(device)
        projector = nn.Linear(args.num_latents * mantis_dim, tabicl_dim).to(device)

    model = MantisAdapterTabICL(
        mantis_model,
        tabicl_model,
        adapter,
        projector=projector,
        mantis_batch_size=args.mantis_batch_size,
    ).to(device)
    
    reader = DataReader(UEA_data_path=args.uea_path, UCR_data_path=args.ucr_path)
    
    # Combine UCR and UEA datasets
    datasets = sorted(reader.dataset_list_ucr)
    
    # --- Pretraining Phase ---
    if not args.no_adapter:
        print(f"Starting Pretraining on {len(datasets)} datasets for {args.epochs} epochs...")
        optimizer = optim.AdamW(
            list(model.adapter.parameters()) + list(model.projector.parameters()),
            lr=args.lr,
            weight_decay=1e-4,
        )
        icl_loss_fn = ICLAlignmentLoss(n_support=128, kl_weight=args.kl_weight).to(device)
        
        for epoch in range(args.epochs):
            random.shuffle(datasets)
            epoch_loss = 0.0
            count = 0
            
            # Create batches
            num_batches = (len(datasets) + args.meta_batch_size - 1) // args.meta_batch_size
            
            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{args.epochs}")
            for i in pbar:
                batch_names = datasets[i*args.meta_batch_size : (i+1)*args.meta_batch_size]
                
                batch_data = []
                for name in batch_names:
                    X_tr, y_tr, X_te, y_te = load_dataset_data(reader, name)
                    if X_tr is not None:
                        batch_data.append((X_tr, y_tr, X_te, y_te))
                
                if not batch_data:
                    continue
                
                try:
                    metrics = train_step(model, optimizer, icl_loss_fn, batch_data, device, args)
                    if metrics is None:
                        continue
                    epoch_loss += metrics["loss"]
                    count += 1

                    # Track running means for easy ablation screenshots.
                    if count == 1:
                        run_indep = metrics["independence_loss"]
                        #run_sparse = metrics["sparsity_loss"]
                    else:
                        run_indep = run_indep + metrics["independence_loss"]
                        #run_sparse = run_sparse + metrics["sparsity_loss"]

                    pbar.set_postfix({
                        'avg_loss': epoch_loss / count if count > 0 else 0,
                        'avg_indep': run_indep / count if count > 0 else 0,
                        #'avg_sparse': run_sparse / count if count > 0 else 0,
                    })
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nSkipping batch due to OOM")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
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
