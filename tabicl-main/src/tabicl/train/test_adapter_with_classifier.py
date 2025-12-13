
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

def get_embeddings(model, X_data, device, batch_size=64):
    """
    Helper to get embeddings from Mantis + Adapter.
    X_data: (N, C, L) tensor or numpy array
    """
    # Ensure X_data is tensor
    if isinstance(X_data, np.ndarray):
        X_data = torch.from_numpy(X_data).float()
        
    embs = []
    N = X_data.size(0)
    
    model.eval()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = X_data[i:i+batch_size].to(device)
            # Mantis forward
            # batch: (B, C, L) -> (B*C, 1, L)
            B, C, L = batch.shape
            batch_in = batch.reshape(-1, L).unsqueeze(1)
            
            # Mantis expects (Batch, Channels, SeqLen)
            # We treat each channel as independent
            
            # Note: MantisAdapterTabICL.forward does batching internally for Mantis.
            # But here we are already batching X_data.
            # We can call model.mantis_model directly.
            
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
    # Keep on CPU to avoid OOM
    X_train = resize_series(X_train, target_len=512)
    X_test = resize_series(X_test, target_len=512)
    
    y_train = torch.from_numpy(y_train_raw).long()
    y_test = torch.from_numpy(y_test_raw).long()
    
    # 2. Train Adapter (Same as train_adapter.py)
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
            perm = torch.randperm(X_train.size(0))
            
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
            
            y_support = y_batch[:, :n_support].to(device)
            y_query = y_batch[:, n_support:].to(device)
            
            unique_classes, inverse_indices = torch.unique(y_support[0], return_inverse=True)
            y_support_mapped = inverse_indices.unsqueeze(0)
            
            max_label = max(y_support.max(), y_query.max()).item()
            mapper = torch.full((max_label + 1,), -1, dtype=torch.long, device=device)
            mapper[unique_classes] = torch.arange(len(unique_classes), device=device)
            
            y_query_mapped = mapper[y_query]
            
            logits, adapter_out = model(X_batch, y_support_mapped)
            
            valid_mask = (y_query_mapped != -1).view(-1)
            if not valid_mask.any():
                continue
                
            logits_flat = logits.reshape(-1, logits.size(-1))
            y_query_flat = y_query_mapped.reshape(-1)
            
            loss_ce = criterion(logits_flat[valid_mask], y_query_flat[valid_mask])
            
            # Diversity Loss (Disabled as per user request in previous turn, but keeping logic consistent)
            # loss_div = div_loss_fn(adapter_out.reshape(-1, adapter_out.size(-1)))
            # total_loss = loss_ce + args.div_weight * loss_div
            
            total_loss = loss_ce
            
            if not total_loss.requires_grad:
                 dummy = sum(p.sum() for p in model.adapter.parameters()) * 0.0
                 total_loss = total_loss + dummy
            
            total_loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': total_loss.item(), 'ce': loss_ce.item()})
            
        # Clear memory
        del optimizer, criterion, div_loss_fn
        torch.cuda.empty_cache()
    else:
        print(f"Skipping training for {dataset_name} (No Adapter)")

    # 3. Evaluation with TabICLClassifier
    print("Evaluating with TabICLClassifier...")
    
    # Extract embeddings
    X_train_emb = get_embeddings(model, X_train, device)
    X_test_emb = get_embeddings(model, X_test, device)
    
    # Initialize TabICLClassifier
    # Note: We use the same checkpoint path.
    # TabICLClassifier will load the model again.
    
    clf = TabICLClassifier(
        model_path=args.tabicl_ckpt,
        n_estimators=32,
        device=device,
        verbose=False,
        mantis_checkpoint=None, # We already encoded
        batch_size=8, # Inference batch size
    )
    
    # Fit on training embeddings
    # Note: y_train is tensor, convert to numpy
    clf.fit(X_train_emb, y_train.numpy())
    
    # Predict on test embeddings
    y_pred = clf.predict(X_test_emb)
    
    acc = np.mean(y_pred == y_test.numpy())
    print(f"Result {dataset_name}: TabICLClassifier Accuracy = {acc:.4f}")
    
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
    # selected_file = "./evaluation_results/mantisTabICL_uea_all_detailed.txt"
    # if os.path.exists(selected_file):
    #     datasets = load_dataset_names_from_file(selected_file)
    # else:
        # Fallback if file not found
    datasets = sorted(reader.dataset_list_ucr)
        
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
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")
    
    ucr_accs = [results[name] for name in results if name in reader.dataset_list_ucr]
    uea_accs = [results[name] for name in results if name in reader.dataset_list_uea]
    
    if ucr_accs:
        print(f"Average UCR Accuracy: {np.mean(ucr_accs):.4f}")
    if uea_accs:
        print(f"Average UEA Accuracy: {np.mean(uea_accs):.4f}")
        
    print(f"Overall Average Accuracy: {np.mean(list(results.values())):.4f}")

    if args.output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
