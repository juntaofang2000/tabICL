import argparse
import os
import sys
import json
import torch
import numpy as np
import random
from torch import nn, optim
import torch.nn.functional as F
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

# -----------------------------
# Meta-ICL view sampling & transforms
# -----------------------------
def sample_view_config():
    """Sample one view (norm / perm / class_shift) for invariance training.

    This is a lightweight proxy of TabICLClassifier's EnsembleGenerator.
    """
    return {
        # 'none' keeps original scale; 'zscore' uses support-only stats to avoid leakage;
        # 'signed_log1p' provides a cheap heavy-tail squash (often helpful for embeddings).
        "norm": random.choice(["none", "zscore", "signed_log1p"]),
        # TabICL is not strictly permutation-invariant, so we explicitly sample permutations.
        "perm": random.choice(["none", "random", "shift"]),
        # Class shift prevents overfitting to absolute class index patterns.
        "class_shift": (random.random() < 0.5),
    }


def apply_view_transform(X, y_support, y_query, n_support, n_classes, cfg):
    """Apply a view transform to (X, y_support, y_query).

    Parameters
    ----------
    X : Tensor, shape (B=1, L, D)
    y_support : Tensor, shape (B=1, n_support)
    y_query : Tensor, shape (B=1, qry_len) (safe labels; invalid already set to 0 and masked outside)
    n_support : int
    n_classes : int
    cfg : dict with keys: norm, perm, class_shift

    Returns
    -------
    X_t, y_sup_t, y_qry_t, meta
        meta contains 'class_shift_offset' for logits alignment.
    """
    X_t = X
    y_sup_t = y_support
    y_qry_t = y_query

    # 1) Normalization (support-only statistics to avoid leakage)
    norm = cfg.get("norm", "none")
    if norm == "zscore":
        X_sup = X_t[:, :n_support, :]
        mean = X_sup.mean(dim=1, keepdim=True)
        std = X_sup.std(dim=1, keepdim=True) + 1e-6
        X_t = (X_t - mean) / std
    elif norm == "signed_log1p":
        X_t = torch.sign(X_t) * torch.log1p(torch.abs(X_t))

    # 2) Feature permutation / shift
    perm = cfg.get("perm", "none")
    D = X_t.size(-1)
    if perm == "random":
        feat_perm = torch.randperm(D, device=X_t.device)
        X_t = X_t[..., feat_perm]
    elif perm == "shift":
        feat_shift = int(torch.randint(0, D, (1,), device=X_t.device).item())
        X_t = torch.roll(X_t, shifts=feat_shift, dims=-1)

    # 3) Class shift
    class_shift_offset = 0
    if cfg.get("class_shift", False) and n_classes > 1:
        class_shift_offset = int(torch.randint(0, n_classes, (1,), device=X_t.device).item())
        y_sup_t = (y_sup_t + class_shift_offset) % n_classes
        y_qry_t = (y_qry_t + class_shift_offset) % n_classes

    meta = {"class_shift_offset": class_shift_offset}
    return X_t, y_sup_t, y_qry_t, meta


def _extract_query_logits(logits, x_full, qry_len):
    """TabICL sometimes returns logits for full sequence, sometimes only for query."""
    if logits.dim() == 3 and logits.size(1) == x_full.size(1):
        return logits[:, -qry_len:, :]
    return logits

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
    """Meta-ICL training for the projector/adapter (Pθ).

    D 主线实现：
    - 冻结 Mantis、冻结 TabICL，只训 adapter/projector (由 optimizer 控制)
    - Episodic 直接优化 TabICL query CE
    - 对不同 (norm/perm/class_shift) 的输出做一致性正则 (symmetric KL)
    """
    model.train()
    optimizer.zero_grad()

    # We process each dataset independently; n_support can vary per task.
    if args.train_size < 1:
        return None

    total_loss = 0.0
    total_ce = 0.0
    total_cons = 0.0
    used = 0

    indep_losses = []
    sparse_losses = []

    for X_train, y_train, X_test, y_test in batch_datasets:
        n_support = min(args.train_size, X_train.size(0))
        if n_support < 1:
            continue

        # 1) Build one episode (TRAIN split only; do NOT touch test labels in pretraining)
        # Shuffle train indices so support/query change every step
        perm_idx = torch.randperm(X_train.size(0))
        X_train_shuf = X_train[perm_idx]
        y_train_shuf = y_train[perm_idx]

        # Optional: enforce K-way episodes (helps match TabICL's <=10-class training regime)
        max_k = int(getattr(args, "max_episode_classes", 0))
        if max_k and max_k > 0:
            all_classes = torch.unique(y_train_shuf)
            if all_classes.numel() > max_k:
                sel = all_classes[torch.randperm(all_classes.numel())[:max_k]]
                keep = (y_train_shuf.view(-1, 1) == sel.view(1, -1)).any(dim=1)
                X_train_shuf = X_train_shuf[keep]
                y_train_shuf = y_train_shuf[keep]
                # Recompute n_support to keep at least 1 query sample
                if X_train_shuf.size(0) <= 1:
                    continue
                n_support = min(int(n_support), max(1, int(X_train_shuf.size(0)) - 1))

        X_sup = X_train_shuf[:n_support]
        y_sup = y_train_shuf[:n_support]

        # Optional: cap multivariate channels for efficiency (especially important for UEA with very large C)
        if getattr(args, "channel_cap", 0) and X_sup.size(1) > int(args.channel_cap):
            # choose channels using support-only variance (avoid leakage)
            ch_var = X_sup.float().var(dim=(0, 2))  # (C,)
            topk = torch.topk(ch_var, k=int(args.channel_cap), largest=True).indices
            X_train_shuf = X_train_shuf[:, topk, :]
            # refresh support after selection
            X_sup = X_train_shuf[:n_support]
            y_sup = y_train_shuf[:n_support]

        # Query pool from the remaining TRAIN samples
        X_qry_pool = X_train_shuf[n_support:]
        y_qry_pool = y_train_shuf[n_support:]
        if X_qry_pool.size(0) < 1:
            continue

        # Optional: sample a fixed number of query points to control compute
        max_q = max(1, int(args.max_icl_len) - int(n_support))
        q_take = min(int(args.query_size), max_q, int(X_qry_pool.size(0)))
        if q_take < int(X_qry_pool.size(0)):
            q_idx = torch.randperm(X_qry_pool.size(0))[:q_take]
            X_qry = X_qry_pool[q_idx]
            y_qry = y_qry_pool[q_idx]
        else:
            X_qry = X_qry_pool
            y_qry = y_qry_pool

        X_seq = torch.cat([X_sup, X_qry], dim=0)

        # 2) Map labels based on support classes (TabICL expects labels 0..K-1)
        unique_classes, inverse_indices = torch.unique(y_sup, return_inverse=True)
        n_classes = int(unique_classes.numel())
        if n_classes < 2:
            continue

        y_sup_mapped = inverse_indices.to(device)  # (n_support,)

        max_label = max(y_sup.max(), y_qry.max()).item()
        mapper = torch.full((max_label + 1,), -1, dtype=torch.long, device=device)
        mapper[unique_classes.to(device)] = torch.arange(n_classes, device=device)

        y_qry_mapped = mapper[y_qry.to(device)]  # (len_qry_total,)
        valid_mask = (y_qry_mapped != -1)

        # 3) Truncate per-dataset (IMPORTANT: do NOT truncate all tasks to the batch min)
        target_len = min(X_seq.size(0), args.max_icl_len)
        if target_len <= n_support:
            continue

        X_seq = X_seq[:target_len]
        qry_len = target_len - n_support

        y_qry_mapped = y_qry_mapped[:qry_len]
        valid_mask = valid_mask[:qry_len]

        if not valid_mask.any():
            continue

        # Safe labels for augmentation (invalid labels are masked anyway)
        y_qry_mapped_safe = y_qry_mapped.clone()
        y_qry_mapped_safe[~valid_mask] = 0

        # 4) Compute embeddings once (Mantis frozen)
        emb_aux = model.get_adapter_output(X_seq.unsqueeze(0), return_aux=True)  # (1, L, D), aux
        X_feat, aux = emb_aux

        # Only log/use aux losses when the episode is used
        if isinstance(aux, dict):
            indep_losses.append(aux.get("independence_loss", torch.tensor(0.0, device=X_feat.device)))
            sparse_losses.append(aux.get("sparsity_loss", torch.tensor(0.0, device=X_feat.device)))

        y_sup_mapped = y_sup_mapped.unsqueeze(0)          # (1, n_support)
        y_qry_mapped_safe = y_qry_mapped_safe.unsqueeze(0)  # (1, qry_len)

        # 5) Two stochastic views
        cfg_a = sample_view_config()
        cfg_b = sample_view_config()

        Xa, ya_sup, ya_qry, meta_a = apply_view_transform(
            X_feat, y_sup_mapped, y_qry_mapped_safe, n_support, n_classes, cfg_a
        )
        Xb, yb_sup, yb_qry, meta_b = apply_view_transform(
            X_feat, y_sup_mapped, y_qry_mapped_safe, n_support, n_classes, cfg_b
        )

        # 6) Forward TabICL (frozen) and extract query logits
        logits_a = model.tabicl_model(Xa, ya_sup, return_logits=True)
        logits_b = model.tabicl_model(Xb, yb_sup, return_logits=True)

        logits_a_q = _extract_query_logits(logits_a, Xa, qry_len)  # (1, qry_len, K)
        logits_b_q = _extract_query_logits(logits_b, Xb, qry_len)

        # 7) Query CE on both views
        ce_a = F.cross_entropy(logits_a_q[0][valid_mask], ya_qry[0][valid_mask])
        ce_b = F.cross_entropy(logits_b_q[0][valid_mask], yb_qry[0][valid_mask])
        loss_ce = 0.5 * (ce_a + ce_b)

        # 8) Invariance consistency (symmetric KL) with class-shift alignment
        shift_a = int(meta_a.get("class_shift_offset", 0))
        shift_b = int(meta_b.get("class_shift_offset", 0))

        # Undo class-shift so both are in the same canonical label space
        logits_a_u = torch.roll(logits_a_q, shifts=-shift_a, dims=-1)
        logits_b_u = torch.roll(logits_b_q, shifts=-shift_b, dims=-1)

        T = float(getattr(args, "cons_temperature", 1.0))
        logp_a = F.log_softmax(logits_a_u[0][valid_mask] / T, dim=-1)
        logp_b = F.log_softmax(logits_b_u[0][valid_mask] / T, dim=-1)

        kl_ab = F.kl_div(logp_a, logp_b.exp(), reduction="batchmean")
        kl_ba = F.kl_div(logp_b, logp_a.exp(), reduction="batchmean")
        loss_cons = 0.5 * (kl_ab + kl_ba) * (T * T)

        loss_i = loss_ce + float(getattr(args, "lambda_cons", 1.0)) * loss_cons
        total_loss = total_loss + loss_i
        total_ce += float(loss_ce.detach().item())
        total_cons += float(loss_cons.detach().item())
        used += 1

    if used == 0:
        return None

    loss = total_loss / used

    # 9) Add causal auxiliary losses (optional regularizers)
    if len(indep_losses) > 0:
        indep_mean_tensor = torch.stack(indep_losses).mean()
        loss = loss + args.indep_weight * indep_mean_tensor
    else:
        indep_mean_tensor = torch.tensor(0.0, device=device)

    if len(sparse_losses) > 0:
        sparse_mean_tensor = torch.stack(sparse_losses).mean()
        loss = loss + args.sparsity_weight * sparse_mean_tensor
    else:
        sparse_mean_tensor = torch.tensor(0.0, device=device)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "loss": float(loss.detach().item()),
        "ce": total_ce / used,
        "cons": total_cons / used,
        "independence_loss": float(indep_mean_tensor.detach().item()),
        "sparsity_loss": float(sparse_mean_tensor.detach().item()),
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
    parser.add_argument("--lambda_cons", type=float, default=1.0, help="Weight for invariance consistency (symmetric KL)")
    parser.add_argument("--cons_temperature", type=float, default=1.0, help="Temperature for consistency KL")
    parser.add_argument("--query_size", type=int, default=256, help="Number of query samples per episode (from TRAIN split only)")
    parser.add_argument("--channel_cap", type=int, default=0, help="If >0, select top-variance channels to cap multivariate channels (helps UEA speed/oom)")
    parser.add_argument("--max_episode_classes", type=int, default=10, help="Max #classes per episode (TabICL is typically trained for <=10 classes)")
    parser.add_argument("--tabicl_dim", type=int, default=256, help="Projector output dim (#features for TabICL)")
    parser.add_argument("--eval_n_estimators", type=int, default=32, help="TabICLClassifier n_estimators during evaluation")
    parser.add_argument("--no_mix_uea", action="store_true", help="Disable mixing UEA datasets in pretraining (only UCR)")
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
    tabicl_dim = args.tabicl_dim
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
    datasets = sorted(reader.dataset_list_ucr + ([] if args.no_mix_uea else reader.dataset_list_uea))
    
    # --- Pretraining Phase ---
    if not args.no_adapter:
        print(f"Starting Pretraining on {len(datasets)} datasets for {args.epochs} epochs...")
        optimizer = optim.AdamW(
            list(model.adapter.parameters()) + list(model.projector.parameters()),
            lr=args.lr,
            weight_decay=1e-4,
        )
        icl_loss_fn = None  # D-mainline: use TabICL query CE + invariance consistency
        
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
                        run_sparse = metrics["sparsity_loss"]
                        run_ce = metrics.get("ce", 0.0)
                        run_cons = metrics.get("cons", 0.0)
                    else:
                        run_indep = run_indep + metrics["independence_loss"]
                        run_sparse = run_sparse + metrics["sparsity_loss"]
                        run_ce = run_ce + metrics.get("ce", 0.0)
                        run_cons = run_cons + metrics.get("cons", 0.0)

                    pbar.set_postfix({
                        'avg_loss': epoch_loss / count if count > 0 else 0,
                        'avg_ce': run_ce / count if count > 0 else 0,
                        'avg_cons': run_cons / count if count > 0 else 0,
                        'avg_indep': run_indep / count if count > 0 else 0,
                        'avg_sparse': run_sparse / count if count > 0 else 0,
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
        n_estimators=args.eval_n_estimators,
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

            # Optional: cap channels in evaluation using TRAIN-set variance, then apply to both train/test
            if getattr(args, "channel_cap", 0) and X_train.size(1) > int(args.channel_cap):
                ch_var = X_train.float().var(dim=(0, 2))
                topk = torch.topk(ch_var, k=int(args.channel_cap), largest=True).indices
                X_train = X_train[:, topk, :]
                X_test = X_test[:, topk, :]
                
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
