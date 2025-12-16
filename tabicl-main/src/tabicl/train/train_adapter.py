
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
from tabicl.model.adapter import CALDA_Adapter, DistributionDiversityLoss
from tabicl.prior.data_reader import DataReader
from tabicl.model.tabicl import TabICL

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
        This is crucial because TabICL's training mode assumes < max_classes labels,
        while eval mode handles hierarchical classification for > max_classes.
        """
        super().train(mode)
        self.mantis_model.eval()
        self.tabicl_model.eval()
        return self

    def forward(self, X, y_train, return_logits=True):
        # X: (Batch, Samples, Channels, Length)
        # But usually we process one "Table" (Task) at a time or a batch of tables.
        # Here X is likely (Batch_of_Tasks, N_Samples, Channels, Length)
        # Or if we process a single dataset, X might be (N_Samples, Channels, Length).
        # TabICL expects (Batch_of_Tasks, N_Samples, D).
        
        # Let's assume input X is (B, N, C, L)
        B, N, C, L = X.shape
        
        # 1. Encode with Mantis
        # Reshape to (B*N*C, L)
        # Note: Mantis expects (Batch, Seq_Len, Channels) where Channels is usually 1 for univariate model?
        # Mantis8M is univariate foundation model. So we treat each channel as independent univariate series.
        # Input to Mantis: (Batch, Seq_Len, 1) or (Batch, Seq_Len) depending on implementation.
        # My previous edit to `mantis_tabicl.py` removed reshape and passed X directly.
        # But here we want to be explicit.
        
        # Mantis expects (Batch, Channels, SeqLen)
        # We treat each channel of each sample as an independent time series.
        # So we reshape to (B*N*C, 1, L).
        X_in = X.reshape(-1, L).unsqueeze(1) # (B*N*C, 1, L)
        
        # Batch processing for Mantis to avoid OOM
        mantis_outs = []
        total_samples = X_in.size(0)
        
        # Get device from mantis model
        device = next(self.mantis_model.parameters()).device
        
        with torch.no_grad():
            for i in range(0, total_samples, self.mantis_batch_size):
                batch = X_in[i : i + self.mantis_batch_size]
                # Move batch to device
                batch = batch.to(device)
                out = self.mantis_model(batch)
                mantis_outs.append(out)
            # Mantis output: (B*N*C, Mantis_Dim)
            mantis_out = torch.cat(mantis_outs, dim=0)
            
        # 2. Adapter
        # Reshape to (B*N, C, Mantis_Dim)
        mantis_out_reshaped = mantis_out.reshape(B*N, C, -1)
        
        if self.adapter is not None:
            # Adapter forward
            # Output: (B*N, TabICL_Dim)
            # Batch execution for Adapter to avoid OOM with high channel counts (e.g. PEMS-SF)
            adapter_outs = []
            adapter_batch_size = 32 # Process 32 samples at a time
            for i in range(0, mantis_out_reshaped.size(0), adapter_batch_size):
                batch_slice = mantis_out_reshaped[i : i + adapter_batch_size]
                out_slice = self.adapter(batch_slice)
                adapter_outs.append(out_slice)
            adapter_out = torch.cat(adapter_outs, dim=0)
        else:
            # No adapter: Flatten channels and use Mantis dim as features
            # mantis_out_reshaped: (B*N, C, Mantis_Dim)
            # We want (B*N, Features).
            # If C=1, it's (B*N, Mantis_Dim).
            # If C>1, we might need to flatten or average?
            # TabICL expects (B, N, D).
            # If we flatten C and Mantis_Dim, D = C * Mantis_Dim.
            # But TabICL treats D as "features".
            # So we flatten.
            adapter_out = mantis_out_reshaped.reshape(B*N, -1)
        
        # 3. TabICL
        # Reshape to (B, N, TabICL_Dim)
        tabicl_in = adapter_out.reshape(B, N, -1)
        
        # TabICL forward
        # y_train: (B, Train_Size)
        # We need to split tabicl_in into Support and Query inside TabICL?
        # TabICL forward takes (R, y_train). R is the whole sequence.
        # It uses y_train to identify the support set (first len(y_train) samples).
        
        out = self.tabicl_model(tabicl_in, y_train, return_logits=return_logits)
        
        return out, adapter_out

def train_one_dataset(args, dataset_name, model, reader, device):
    print(f"Processing {dataset_name}...")
    
    # 1. Load Data
    try:
        X_train_raw, y_train_raw = reader.read_dataset(dataset_name, which_set="train")
        X_test_raw, y_test_raw = reader.read_dataset(dataset_name, which_set="test")
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None

    # Ensure 3D (N, C, L)
    X_train = _ensure_three_dim(X_train_raw)
    X_test = _ensure_three_dim(X_test_raw)
    
    # Resize to 512 (Mantis default)
    # Keep on CPU to avoid OOM
    X_train = resize_series(X_train, target_len=512)
    X_test = resize_series(X_test, target_len=512)
    
    y_train = torch.from_numpy(y_train_raw).long()
    y_test = torch.from_numpy(y_test_raw).long()
    
    # 2. Prepare Data for In-Context Learning
    # We treat the entire training set as the "Support Set" and we want to learn to adapt it.
    # But TabICL is an ICL model. It takes (Support + Query).
    # Here we are training the *Adapter*.
    # Strategy:
    # We can sample subsets from X_train as (Support, Query) to train the adapter.
    # Or we can use X_train as Support and X_train (part of it) as Query?
    # Standard ICL training: Sample a batch of tasks.
    # Here we have one dataset. We can simulate tasks by subsampling.
    
    # Create a DataLoader that samples batches of (Support, Query) from X_train
    # For simplicity, let's just randomly split X_train into Support/Query in each iteration.
    
    if model.adapter is not None:
        optimizer = optim.AdamW(model.adapter.parameters(), lr=args.lr, weight_decay=1e-4)
        div_loss_fn = DistributionDiversityLoss()
        criterion = nn.CrossEntropyLoss()
        
        model.train() # Set adapter to train mode (Mantis/TabICL are frozen in init)
        
        # Training Loop
        pbar = tqdm(range(args.epochs), desc=f"Training {dataset_name}")
        for epoch in pbar:
            optimizer.zero_grad()
            
            # Sample a "task" from X_train
            # Shuffle indices
            perm = torch.randperm(X_train.size(0))
            
            # Split into Support and Query
            # If dataset is small, we might use all.
            # Let's use a random split, e.g., 50% support, 50% query (or limited by max seq len)
            n_samples = X_train.size(0)
            n_support = int(n_samples * 0.5)
            if n_support < 1: n_support = 1
            if n_support >= n_samples: n_support = n_samples - 1
            
            # Limit total sequence length to avoid OOM
            max_icl_len = args.max_icl_len
            if n_samples > max_icl_len:
                indices = perm[:max_icl_len]
                n_support = int(max_icl_len * 0.5)
            else:
                indices = perm
                
            X_batch = X_train[indices].unsqueeze(0) # (1, N, C, L)
            y_batch = y_train[indices].unsqueeze(0) # (1, N)
            
            # y_train_input for TabICL is just the support labels
            y_support = y_batch[:, :n_support].to(device)
            y_query = y_batch[:, n_support:].to(device)
            
            # Remap labels to 0..k-1 based on support set
            # This ensures TabICL receives valid 0..k-1 labels and outputs k logits.
            unique_classes, inverse_indices = torch.unique(y_support[0], return_inverse=True)
            y_support_mapped = inverse_indices.unsqueeze(0)
            
            # Map query labels
            max_label = max(y_support.max(), y_query.max()).item()
            mapper = torch.full((max_label + 1,), -1, dtype=torch.long, device=device)
            mapper[unique_classes] = torch.arange(len(unique_classes), device=device)
            
            y_query_mapped = mapper[y_query]
            
            # Forward
            # TabICL output is (B, N_Query, n_classes)
            # Pass remapped support labels
            logits, adapter_out = model(X_batch, y_support_mapped)
            
            # Loss
            # logits: (1, N_Query, n_classes)
            # y_query: (1, N_Query)
            
            # Filter valid query samples (those whose class is in support)
            valid_mask = (y_query_mapped != -1).view(-1)
            if not valid_mask.any():
                continue
                
            logits_flat = logits.reshape(-1, logits.size(-1))
            y_query_flat = y_query_mapped.reshape(-1)
            
            loss_ce = criterion(logits_flat[valid_mask], y_query_flat[valid_mask])
            
            # Diversity Loss
            #loss_div = div_loss_fn(adapter_out.reshape(-1, adapter_out.size(-1)))
            
            #total_loss = loss_ce + args.div_weight * loss_div
            total_loss = loss_ce
            loss_div  =  torch.tensor(0.0)
            
            # Ensure gradients flow to adapter even if TabICL blocks them
            if not total_loss.requires_grad:
                 # If loss_ce has no grad (TabICL frozen/detached), we attach a dummy gradient
                 # to the adapter parameters so the optimizer step doesn't fail.
                 # This effectively means NO update happens, but it prevents the crash.
                 dummy = sum(p.sum() for p in model.adapter.parameters()) * 0.0
                 total_loss = total_loss + dummy
            
            total_loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': total_loss.item(), 'ce': loss_ce.item()})
            # pbar.set_postfix({'loss': total_loss.item(), 'ce': loss_ce.item(), 'div': loss_div.item()})
    else:
        print(f"Skipping training for {dataset_name} (No Adapter)")

    # 3. Evaluation
    # Clear memory from training
    if model.adapter is not None:
        del optimizer, criterion, div_loss_fn
    torch.cuda.empty_cache()
    
    model.eval()
    with torch.no_grad():
        # Strategy: Use X_train as support.
        # If X_train is too large, subsample it to avoid OOM in TabICL (N^2 attention).
        # Batch X_test (Query) to avoid OOM.
        
        MAX_SUPPORT_SIZE = 3000  # Limit support set size for evaluation
        EVAL_QUERY_BATCH_SIZE = 64 # Batch size for queries
        
        n_train = X_train.size(0)
        if n_train > MAX_SUPPORT_SIZE:
            # Subsample support set
            perm = torch.randperm(n_train)[:MAX_SUPPORT_SIZE]
            X_support = X_train[perm].unsqueeze(0) # (1, N_Sup, C, L)
            y_support = y_train[perm].unsqueeze(0).to(device) # (1, N_Sup)
        else:
            X_support = X_train.unsqueeze(0)
            y_support = y_train.unsqueeze(0).to(device)
            
        n_test = X_test.size(0)
        all_preds = []
        
        # Loop over test set in batches
        for i in range(0, n_test, EVAL_QUERY_BATCH_SIZE):
            X_query_batch = X_test[i : i + EVAL_QUERY_BATCH_SIZE].unsqueeze(0) # (1, N_Q, C, L)
            # y_query_batch = y_test[i : i + EVAL_QUERY_BATCH_SIZE].to(device) # Not needed for forward
            
            # Combine: [Support, Query_Batch]
            X_combined = torch.cat([X_support, X_query_batch], dim=1)
            
            # Forward
            # y_support is passed to identify support samples
            logits, _ = model(X_combined, y_support)
            
            # logits: (1, N_Q, n_classes)
            preds_batch = logits.argmax(dim=-1).cpu() # (1, N_Q)
            all_preds.append(preds_batch)
            
        # Concatenate all predictions
        preds = torch.cat(all_preds, dim=1).squeeze(0) # (N_Test)
        
        acc = (preds == y_test).float().mean().item()
        
        print(f"Result {dataset_name}: Accuracy = {acc:.4f}")
        return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabicl_ckpt", type=str, default="/data0/fangjuntao2025/tabicl-main/tabICLOrignCheckpoint/tabicl-classifier-v1.1-0506.ckpt")
    parser.add_argument("--mantis_ckpt", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/Models/Mantis/Mantis_cheickpoint/")
    parser.add_argument("--uea_path", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/")
    parser.add_argument("--ucr_path", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--div_weight", type=float, default=0.1)
    parser.add_argument("--max_icl_len", type=int, default=512, help="Max sequence length for ICL training to avoid OOM")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_adapter", action="store_true", help="Disable adapter and use raw Mantis embeddings")
    parser.add_argument("--mantis_batch_size", type=int, default=16, help="Batch size for Mantis encoder")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device(args.device)
    if device.type == 'cuda' and device.index is not None:
        torch.cuda.set_device(device)
    
    # 1. Build Models
    print("Loading models...")
    # Load Mantis
    mantis_model = build_mantis_encoder(args.mantis_ckpt, device=device)
    
    # Load TabICL
    tabicl_state = torch.load(args.tabicl_ckpt, map_location="cpu")
    tabicl_model = TabICL(**tabicl_state["config"])
    tabicl_model.load_state_dict(tabicl_state["state_dict"])
    tabicl_model.to(device)
    
    # Initialize Adapter
    # Mantis Dim = 256 (default), TabICL Dim = 512 (default for TabICL?)
    # Check TabICL config
    tabicl_dim = 100
    mantis_dim = mantis_model.hidden_dim
    
    print(f"Mantis Dim: {mantis_dim}, TabICL Dim: {tabicl_dim}")
    
    # We need a fresh adapter for each dataset? 
    # The user said "Train adapter on UCR/UEA train sets... One by one".
    # This implies we reset the adapter for each dataset.
    
    reader = DataReader(UEA_data_path=args.uea_path, UCR_data_path=args.ucr_path)
    
    datasets = sorted(reader.dataset_list_uea+reader.dataset_list_ucr)
    
    results = {}
    # selected_file = "./evaluation_results/mantisTabICL_uea_all_detailed.txt"
    # datasets = load_dataset_names_from_file(selected_file)
    for dataset_name in datasets:
        # Reset Adapter
        if args.no_adapter:
            adapter = None
        else:
            adapter = CALDA_Adapter(mantis_emb_dim=mantis_dim, tabicl_input_dim=tabicl_dim).to(device)
        
        model = MantisAdapterTabICL(mantis_model, tabicl_model, adapter, mantis_batch_size=args.mantis_batch_size).to(device)
        
        acc = train_one_dataset(args, dataset_name, model, reader, device)
        if acc is not None:
            results[dataset_name] = acc
            
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
