
import argparse
import os
import sys
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
import copy

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../model"))

from tabicl.model.mantis_tabicl import MantisTabICL, build_mantis_encoder
from tabicl.model.adapter import CALDA_Adapter, DistributionDiversityLoss
from tabicl.prior.data_reader import DataReader
from tabicl.model.tabicl import TabICL

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
    def __init__(self, mantis_model, tabicl_model, adapter):
        super().__init__()
        self.mantis_model = mantis_model
        self.tabicl_model = tabicl_model
        self.adapter = adapter
        
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
        
        with torch.no_grad():
            # Mantis output: (B*N*C, Mantis_Dim)
            mantis_out = self.mantis_model(X_in)
            
        # 2. Adapter
        # Reshape to (B*N, C, Mantis_Dim)
        mantis_out_reshaped = mantis_out.reshape(B*N, C, -1)
        
        if self.adapter is not None:
            # Adapter forward
            # Output: (B*N, TabICL_Dim)
            adapter_out = self.adapter(mantis_out_reshaped)
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
    X_train = resize_series(X_train, target_len=512).to(device)
    X_test = resize_series(X_test, target_len=512).to(device)
    
    y_train = torch.from_numpy(y_train_raw).long().to(device)
    y_test = torch.from_numpy(y_test_raw).long().to(device)
    
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
            y_support = y_batch[:, :n_support]
            y_query = y_batch[:, n_support:]
            
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
            loss_div = div_loss_fn(adapter_out.reshape(-1, adapter_out.size(-1)))
            
            total_loss = loss_ce + args.div_weight * loss_div
            
            total_loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': total_loss.item(), 'ce': loss_ce.item(), 'div': loss_div.item()})
    else:
        print(f"Skipping training for {dataset_name} (No Adapter)")

    # 3. Evaluation
    model.eval()
    with torch.no_grad():
        # Use all X_train as support, predict X_test
        # We might need to batch X_test if it's too large
        
        # Support
        X_support = X_train.unsqueeze(0)
        y_support = y_train.unsqueeze(0)
        
        # Query
        X_query = X_test.unsqueeze(0)
        y_query = y_test.unsqueeze(0)
        
        # Combine for TabICL input: [Support, Query]
        X_combined = torch.cat([X_support, X_query], dim=1)
        
        # Forward
        # We need to handle OOM here too if X_combined is huge.
        # But for now let's try direct forward.
        
        logits, _ = model(X_combined, y_support)
        
        preds = logits.argmax(dim=-1)
        acc = (preds == y_query).float().mean().item()
        
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
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
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
    
    datasets = sorted(reader.dataset_list_ucr + reader.dataset_list_uea)
    
    results = {}
    
    for dataset_name in datasets:
        # Reset Adapter
        if args.no_adapter:
            adapter = None
        else:
            adapter = CALDA_Adapter(mantis_emb_dim=mantis_dim, tabicl_input_dim=tabicl_dim).to(device)
        
        model = MantisAdapterTabICL(mantis_model, tabicl_model, adapter).to(device)
        
        acc = train_one_dataset(args, dataset_name, model, reader, device)
        if acc is not None:
            results[dataset_name] = acc
            
    print("\nFinal Results:")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")
    print(f"Average Accuracy: {np.mean(list(results.values())):.4f}")

if __name__ == "__main__":
    main()
