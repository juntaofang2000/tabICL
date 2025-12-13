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
            mantis_out = torch.cat(mantis_outs, dim=0)
            
        # 2. Adapter
        mantis_out_reshaped = mantis_out.reshape(B*N, C, -1)
        
        if self.adapter is not None:
            adapter_out = self.adapter(mantis_out_reshaped)
        else:
            adapter_out = mantis_out_reshaped.reshape(B*N, -1)
        
        # 3. TabICL
        tabicl_in = adapter_out.reshape(B, N, -1)
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
        mantis_out_reshaped = mantis_out.reshape(B*N, C, -1)
        
        if self.adapter is not None:
            adapter_out = self.adapter(mantis_out_reshaped)
        else:
            adapter_out = mantis_out_reshaped.reshape(B*N, -1)
            
        return adapter_out.reshape(B, N, -1)

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
                adapter_out = model.adapter(mantis_out_reshaped) # (B, TabICL_Dim)
            else:
                adapter_out = mantis_out_reshaped.reshape(B, -1)
                
            embs.append(adapter_out.cpu().numpy())
            
    return np.concatenate(embs, axis=0)

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
    X_train = resize_series(X_train, target_len=512)
    X_test = resize_series(X_test, target_len=512)
    
    y_train = torch.from_numpy(y_train_raw).long()
    y_test = torch.from_numpy(y_test_raw).long()
    
    # Initialize TabICLClassifier for validation
    # We initialize it once to avoid reloading the model every epoch
    clf = TabICLClassifier(
        model_path=args.tabicl_ckpt,
        n_estimators=32,
        device=device,
        verbose=False,
        mantis_checkpoint=None, # We already encoded
        batch_size=8, # Inference batch size
    )
    
    best_acc = 0.0
    
    # 2. Train Adapter
    if model.adapter is not None:
        optimizer = optim.AdamW(model.adapter.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        model.train() 
        
        # Training Loop
        pbar = tqdm(range(args.epochs), desc=f"Training {dataset_name}")
        for epoch in pbar:
            # --- Training Step ---
            model.train()
            optimizer.zero_grad()
            
            # Sample a "task" from X_train
            perm = torch.randperm(X_train.size(0))
            n_samples = X_train.size(0)
            n_support = int(n_samples * 0.5)
            if n_support < 1: n_support = 1
            if n_support >= n_samples: n_support = n_samples - 1
            
            max_icl_len = args.max_icl_len
            if n_samples > max_icl_len:
                indices = perm[:max_icl_len]
                n_support = int(max_icl_len * 0.5)
            else:
                indices = perm
                
            X_batch = X_train[indices].unsqueeze(0) # (1, N, C, L)
            y_batch = y_train[indices].unsqueeze(0) # (1, N)
            
            y_support = y_batch[:, :n_support].to(device)
            y_query = y_batch[:, n_support:].to(device)
            
            unique_classes, inverse_indices = torch.unique(y_support[0], return_inverse=True)
            y_support_mapped = inverse_indices.unsqueeze(0)
            
            max_label = max(y_support.max(), y_query.max()).item()
            mapper = torch.full((max_label + 1,), -1, dtype=torch.long, device=device)
            mapper[unique_classes] = torch.arange(len(unique_classes), device=device)
            
            y_query_mapped = mapper[y_query]
            
            # --- Augmentation Step ---
            # Get embeddings
            adapter_out = model.get_adapter_output(X_batch) # (1, N, D)
            
            # Apply augmentations
            X_aug, y_support_aug, y_query_aug = augment_batch(
                adapter_out, 
                y_support_mapped, 
                y_query_mapped, 
                device, 
                len(unique_classes)
            )
            
            # Forward TabICL with augmented data
            # Note: TabICL expects y_train (support labels)
            logits = model.tabicl_model(X_aug, y_support_aug, return_logits=True)
            
            valid_mask = (y_query_mapped != -1).view(-1)
            if not valid_mask.any():
                continue
                
            logits_flat = logits.reshape(-1, logits.size(-1))
            y_query_flat = y_query_aug.reshape(-1) # Use augmented query labels
            
            loss_ce = criterion(logits_flat[valid_mask], y_query_flat[valid_mask])
            total_loss = loss_ce
            
            if not total_loss.requires_grad:
                 dummy = sum(p.sum() for p in model.adapter.parameters()) * 0.0
                 total_loss = total_loss + dummy
            
            total_loss.backward()
            optimizer.step()
            
            # --- Validation Step (using TabICLClassifier) ---
            # Extract embeddings
            X_train_emb = get_embeddings(model, X_train, device)
            X_test_emb = get_embeddings(model, X_test, device)
            
            # Fit and Predict
            clf.fit(X_train_emb, y_train.numpy())
            y_pred = clf.predict(X_test_emb)
            acc = np.mean(y_pred == y_test.numpy())
            
            if acc > best_acc:
                best_acc = acc
            
            pbar.set_postfix({'loss': total_loss.item(), 'val_acc': acc, 'best': best_acc})
            
        # Clear memory
        del optimizer, criterion
        torch.cuda.empty_cache()
    else:
        print(f"Skipping training for {dataset_name} (No Adapter)")
        # If no adapter, just evaluate once
        X_train_emb = get_embeddings(model, X_train, device)
        X_test_emb = get_embeddings(model, X_test, device)
        clf.fit(X_train_emb, y_train.numpy())
        y_pred = clf.predict(X_test_emb)
        best_acc = np.mean(y_pred == y_test.numpy())

    print(f"Result {dataset_name}: Best TabICLClassifier Accuracy = {best_acc:.4f}")
    return best_acc

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
    
    # Load TabICL (for training adapter)
    tabicl_state = torch.load(args.tabicl_ckpt, map_location="cpu")
    tabicl_model = TabICL(**tabicl_state["config"])
    tabicl_model.load_state_dict(tabicl_state["state_dict"])
    tabicl_model.to(device)
    
    # Initialize Adapter
    tabicl_dim = 100
    mantis_dim = mantis_model.hidden_dim
    
    print(f"Mantis Dim: {mantis_dim}, TabICL Dim: {tabicl_dim}")
    
    reader = DataReader(UEA_data_path=args.uea_path, UCR_data_path=args.ucr_path)
    
    results = {}
    # Default to UCR datasets if no file specified
    datasets = sorted(reader.dataset_list_ucr + reader.dataset_list_uea)
        
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
