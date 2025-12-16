import argparse
import os
import sys
import json
import torch
import numpy as np
import random
from torch import nn, optim
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from tabicl.model.CausalAdapter import RobustCWA, RobustCausalLoss
# 允许直接运行该文件时也能 import 到 src/ 下的 tabicl 包
_SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)


# Placeholder imports if environment is different, user should verify these
try:
    from tabicl.model.mantis_tabicl import build_mantis_encoder
    from tabicl.model.mantis_dev.trainer.trainer import MantisTrainer
    from tabicl.prior.data_reader import DataReader
    from tabicl.model.tabicl import TabICL
    from tabicl.sklearn.classifier import TabICLClassifier
except ImportError:
    print("Warning: TabICL specific modules not found. Ensure project structure is correct.")

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
    if X.shape[2] == target_len:
        return torch.from_numpy(X).float()
    X_tensor = torch.from_numpy(X).float()
    return torch.nn.functional.interpolate(X_tensor, size=target_len, mode='linear', align_corners=False)

def get_mantis_embeddings(model, X, batch_size=64, device='cuda'):
    """Extract embeddings from frozen Mantis backbone.

    Note: In this repo, the recommended extraction path is via `MantisTrainer.transform`,
    which handles per-channel encoding and batching.
    """

    target_device = torch.device(device) if isinstance(device, str) else device
    was_training = model.training
    model.eval()
    try:
        trainer = MantisTrainer(device=target_device, network=model)
        with torch.no_grad():
            X_dev = X.to(target_device) if isinstance(X, torch.Tensor) else torch.as_tensor(X, dtype=torch.float32, device=target_device)
            emb = trainer.transform(X_dev, batch_size=batch_size, to_numpy=False)
            if isinstance(emb, tuple):
                emb = emb[0]
            return emb.detach().cpu()
    finally:
        model.train(was_training)

def load_dataset_data(reader, dataset_name):
    try:
        X_train_raw, y_train_raw = reader.read_dataset(dataset_name, which_set="train")
        X_test_raw, y_test_raw = reader.read_dataset(dataset_name, which_set="test")
    except Exception as e:
        # print(f"Error loading {dataset_name}: {e}")
        return None, None, None, None

    X_train = _ensure_three_dim(X_train_raw)
    X_test = _ensure_three_dim(X_test_raw)
    
    X_train = resize_series(X_train, target_len=512)
    X_test = resize_series(X_test, target_len=512)
    
    y_train = torch.from_numpy(y_train_raw).long()
    y_test = torch.from_numpy(y_test_raw).long()
    
    return X_train, y_train, X_test, y_test

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

def train_step(adapter, mantis_model, tabicl_model, criterion, optimizer, batch_datasets, device, args):
    adapter.train()
    mantis_model.eval()
    tabicl_model.eval()
    optimizer.zero_grad()
    
    # 1. Determine n_support (context size) for this batch
    min_train_len = min(d[0].size(0) for d in batch_datasets)
    n_support = min(args.train_size, min_train_len)
    
    if n_support < 1:
        return 0.0, 0.0, 0.0, 0.0

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
        
        unique_classes, inverse_indices = torch.unique(y_sup, return_inverse=True)
        y_sup_mapped = inverse_indices.to(device)
        
        max_label = max(y_sup.max(), y_qry.max()).item()
        mapper = torch.full((max_label + 1,), -1, dtype=torch.long, device=device)
        mapper[unique_classes] = torch.arange(len(unique_classes), device=device)
        
        y_qry_mapped = mapper[y_qry.to(device)]
        valid_mask = (y_qry_mapped != -1)
        
        y_qry_mapped_safe = y_qry_mapped.clone()
        y_qry_mapped_safe[~valid_mask] = 0
        
        y_sup_mapped_list.append(y_sup_mapped)
        y_qry_mapped_list.append(y_qry_mapped_safe)
        valid_mask_list.append(valid_mask)

    # 2. Collate
    min_len = min(x.size(0) for x in X_seq_list)
    target_len = min(min_len, args.max_icl_len)
    
    if target_len <= n_support:
        return 0.0, 0.0, 0.0, 0.0

    X_batch_list = []
    y_sup_batch_list = []
    y_qry_batch_list = []
    mask_batch_list = []
    
    for i in range(len(batch_datasets)):
        X_batch_list.append(X_seq_list[i][:target_len])
        y_sup_batch_list.append(y_sup_mapped_list[i])
        
        qry_len = target_len - n_support
        y_qry_batch_list.append(y_qry_mapped_list[i][:qry_len])
        mask_batch_list.append(valid_mask_list[i][:qry_len])

    X_batch = torch.stack(X_batch_list).to(device) # (B, L, C, T)
    
    # 3. Get Mantis Embeddings
    B, L, C, T = X_batch.shape
    X_in = X_batch.reshape(-1, T).unsqueeze(1) # (B*L*C, 1, T)
    
    mantis_outs = []
    mantis_batch_size = args.mantis_batch_size
    with torch.no_grad():
        for i in range(0, X_in.size(0), mantis_batch_size):
            batch = X_in[i : i + mantis_batch_size]
            out = mantis_model(batch)
            mantis_outs.append(out)
        mantis_out = torch.cat(mantis_outs, dim=0) # (B*L*C, D_mantis)
    
    mantis_out_reshaped = mantis_out.reshape(B*L, C, -1) # (B*L, C, D_mantis)
    z_mantis = mantis_out_reshaped.mean(dim=1) # (B*L, D_mantis)
    
    # 4. Adapter Forward (RobustCWA)
    z_all_flat, h_orth, W_orth, W_white = adapter(z_mantis, training_mode=True)
    
    # Reshape for TabICL: (B, L, D_out)
    z_all = z_all_flat.view(B, L, -1)
    
    # 5. Augmentation & TabICL Forward
    
    # Compute Reg Terms (Global)
    dummy_logits = torch.randn(1, 2, device=device)
    dummy_targets = torch.zeros(1, dtype=torch.long, device=device)
    _, reg_dict = criterion(dummy_logits, dummy_targets, h_orth, W_orth, z_all_flat)
    orth_loss = reg_dict['orth']
    indep_loss = reg_dict['indep']
    
    task_loss_accum = 0.0
    valid_tasks = 0
    
    for i in range(B):
        emb = z_all[i].unsqueeze(0) # (1, L, D)
        y_sup = y_sup_batch_list[i].unsqueeze(0)
        y_qry = y_qry_batch_list[i].unsqueeze(0)
        mask_in = mask_batch_list[i]
        
        if not mask_in.any():
            continue
            
        n_classes = y_sup.max().item() + 1
        
        X_aug, y_sup_aug, y_qry_aug = augment_batch(
            emb, y_sup, y_qry, device, n_classes
        )
        
        logits = tabicl_model(X_aug, y_sup_aug, return_logits=True)
        
        qry_len = y_qry_aug.size(1)
        if logits.size(1) == X_aug.size(1):
             logits_qry = logits[:, -qry_len:, :]
        else:
             logits_qry = logits
             
        logits_flat = logits_qry.reshape(-1, logits_qry.size(-1))
        y_flat = y_qry_aug.reshape(-1)
        mask_flat = mask_in.reshape(-1)
        
        ce_loss = criterion.ce_loss(logits_flat[mask_flat], y_flat[mask_flat])
        task_loss_accum += ce_loss
        valid_tasks += 1
        
    if valid_tasks > 0:
        avg_task_loss = task_loss_accum / valid_tasks
    else:
        avg_task_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
    final_loss = avg_task_loss + criterion.lambda_orth * orth_loss + criterion.lambda_indep * indep_loss
    
    if not final_loss.requires_grad:
         dummy = sum(p.sum() for p in adapter.parameters()) * 0.0
         final_loss = final_loss + dummy
         
    final_loss.backward()
    
    torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return final_loss.item(), avg_task_loss.item(), orth_loss.item(), indep_loss.item()

def evaluate_all(adapter, mantis_model, tabicl_ckpt, reader, datasets, device, args):
    print("\nEvaluating on all datasets...")
    adapter.eval()
    mantis_model.eval()
    
    clf = TabICLClassifier(
        model_path=tabicl_ckpt,
        n_estimators=32,
        device=device,
        verbose=False,
        mantis_checkpoint=None,
        batch_size=8
    )
    
    results = {}
    
    for name in tqdm(datasets, desc="Evaluating"):
        try:
            X_train_raw, y_train_raw = reader.read_dataset(name, which_set="train")
            X_test_raw, y_test_raw = reader.read_dataset(name, which_set="test")
        except:
            continue
            
        X_train = resize_series(_ensure_three_dim(X_train_raw))
        X_test = resize_series(_ensure_three_dim(X_test_raw))
        
        z_train = get_mantis_embeddings(mantis_model, X_train, device=device).to(device)
        z_test = get_mantis_embeddings(mantis_model, X_test, device=device).to(device)
        
        with torch.no_grad():
            x_train_tab = adapter(z_train, training_mode=False).cpu().numpy()
            x_test_tab = adapter(z_test, training_mode=False).cpu().numpy()
            
        x_train_tab = np.nan_to_num(x_train_tab)
        x_test_tab = np.nan_to_num(x_test_tab)
            
        clf.fit(x_train_tab, y_train_raw)
        y_pred = clf.predict(x_test_tab)
        
        acc = np.mean(y_pred == y_test_raw)
        results[name] = acc
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabicl_ckpt", type=str, default="/data0/fangjuntao2025/tabicl-main/tabICLOrignCheckpoint/tabicl-classifier-v1.1-0506.ckpt")
    parser.add_argument("--mantis_ckpt", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/Models/Mantis/Mantis_cheickpoint/")
    parser.add_argument("--ucr_path", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/")
    parser.add_argument("--uea_path", type=str, default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/")
    parser.add_argument("--save_adapter", type=str, default="./checkpoints/causal_adapter_ucr.pt")
    parser.add_argument("--output_file", type=str, default="evaluation_results/causal_adapter_ucr_results.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mantis_batch_size", type=int, default=16)
    parser.add_argument("--meta_batch_size", type=int, default=8)
    parser.add_argument("--train_size", type=int, default=100)
    parser.add_argument("--max_icl_len", type=int, default=512)

    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Load Models
    print("Loading Mantis...")
    mantis_model = build_mantis_encoder(args.mantis_ckpt, device=device)
    mantis_model.eval()
    
    print("Loading TabICL...")
    tabicl_state = torch.load(args.tabicl_ckpt, map_location="cpu")
    tabicl_model = TabICL(**tabicl_state["config"])
    tabicl_model.load_state_dict(tabicl_state["state_dict"])
    tabicl_model.to(device)
    
    # 2. Initialize Causal Adapter
    adapter = RobustCWA(input_dim=mantis_model.hidden_dim, hidden_dim=128, output_dim=64).to(device)
    
    criterion = RobustCausalLoss(lambda_orth=0.01, lambda_indep=0.01).to(device)
    
    optimizer = optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 3. Prepare Data
    reader = DataReader(UCR_data_path=args.ucr_path, UEA_data_path=args.uea_path)
    datasets = sorted(reader.dataset_list_ucr)
    
    # 4. Training Loop
    print(f"Starting Pre-training on {len(datasets)} datasets...")
    
    for epoch in range(args.epochs):
        random.shuffle(datasets)
        
        total_loss = 0.0
        total_task = 0.0
        total_orth = 0.0
        total_indep = 0.0
        count = 0
        
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
                
            loss, task, orth, indep = train_step(
                adapter, mantis_model, tabicl_model, criterion, optimizer, batch_data, device, args
            )
            
            total_loss += loss
            total_task += task
            total_orth += orth
            total_indep += indep
            count += 1
            
            pbar.set_postfix({
                'loss': f"{total_loss/count:.4f}", 
                'task': f"{total_task/count:.4f}",
                'orth': f"{total_orth/count:.4f}"
            })
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            os.makedirs(os.path.dirname(args.save_adapter), exist_ok=True)
            torch.save(adapter.state_dict(), args.save_adapter)
            
    torch.save(adapter.state_dict(), args.save_adapter)
    print(f"Training Complete. Adapter saved to {args.save_adapter}")

    # --- Evaluation ---
    results = evaluate_all(adapter, mantis_model, args.tabicl_ckpt, reader, datasets, device, args)
    
    avg_acc = np.mean(list(results.values()))
    print(f"\nAverage UCR Accuracy: {avg_acc:.4f}")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()