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

# Add src to path
_SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from tabicl.model.mantis_tabicl import build_mantis_encoder
from tabicl.model.CausalAdapter import RobustCWA, RobustCausalLoss
from tabicl.prior.data_reader import DataReader
from tabicl.model.tabicl import TabICL
from tabicl.sklearn.classifier import TabICLClassifier
from tabicl.model.mantis_dev.trainer.trainer import MantisTrainer

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
    """Extract embeddings from frozen Mantis backbone."""
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

def sample_episodic_batch(dataset_cache, min_samples=4, max_batch_size=128):
    """
    Sample an episodic batch from the cache of ALL datasets.
    """
    dataset_names = list(dataset_cache.keys())
    name = random.choice(dataset_names)
    X_emb, y_all = dataset_cache[name]
    
    unique_classes = torch.unique(y_all)
    if len(unique_classes) < 2:
        return None, None, None, None
        
    # Randomly select classes (2 to 10)
    num_classes = random.randint(2, min(len(unique_classes), 10))
    
    # Filter classes with enough samples
    perm = torch.randperm(len(unique_classes))
    candidates = unique_classes[perm]
    
    valid_classes = []
    for cls_val in candidates:
        indices = (y_all == cls_val).nonzero(as_tuple=True)[0]
        if len(indices) >= 2:
            valid_classes.append(cls_val)
            if len(valid_classes) == num_classes:
                break
    
    if len(valid_classes) < 2:
        return None, None, None, None
        
    selected_classes = torch.stack(valid_classes)
    
    # Remap labels
    remap_dict = {old_cls.item(): new_cls for new_cls, old_cls in enumerate(selected_classes)}
    
    support_indices = []
    query_indices = []
    
    for cls_val in selected_classes:
        indices = (y_all == cls_val).nonzero(as_tuple=True)[0]
        indices = indices[torch.randperm(len(indices))]
        
        n_samples = len(indices)
        n_support = max(1, min(n_samples // 2, 20))
        n_query = max(1, min(n_samples - n_support, 20))
        
        support_indices.append(indices[:n_support])
        query_indices.append(indices[n_support:n_support+n_query])
        
    if not support_indices or not query_indices:
        return None, None, None, None
        
    support_indices = torch.cat(support_indices)
    query_indices = torch.cat(query_indices)
    
    if len(support_indices) + len(query_indices) > max_batch_size:
        return None, None, None, None
        
    X_support = X_emb[support_indices]
    y_support = torch.tensor([remap_dict[y_all[i].item()] for i in support_indices], dtype=torch.long)
    
    X_query = X_emb[query_indices]
    y_query = torch.tensor([remap_dict[y_all[i].item()] for i in query_indices], dtype=torch.long)
    
    return X_support, y_support, X_query, y_query

def train_epoch(adapter, tabicl_model, criterion, optimizer, train_cache, device, steps_per_epoch=100):
    adapter.train()
    tabicl_model.eval()
    
    total_loss = 0
    total_task = 0
    total_orth = 0
    total_indep = 0
    valid_steps = 0
    
    pbar = tqdm(range(steps_per_epoch), desc="Training Steps", leave=False)
    
    for _ in pbar:
        X_supp_cpu, y_supp_cpu, X_qry_cpu, y_qry_cpu = sample_episodic_batch(train_cache)
        
        if X_supp_cpu is None:
            continue
            
        X_supp = X_supp_cpu.to(device)
        y_supp = y_supp_cpu.to(device)
        X_qry = X_qry_cpu.to(device)
        y_qry = y_qry_cpu.to(device)
        
        X_all = torch.cat([X_supp, X_qry], dim=0)
        
        optimizer.zero_grad()
        
        # Adapter Forward
        z_all, h_orth, W_orth, W_white = adapter(X_all, training_mode=True)
        
        n_supp = X_supp.shape[0]
        z_supp = z_all[:n_supp]
        z_qry = z_all[n_supp:]

        # TabICL Forward
        X_table = torch.cat([z_supp, z_qry], dim=0).unsqueeze(0).float()
        y_train = y_supp.unsqueeze(0)
        logits = tabicl_model(X_table, y_train)
        logits = logits.squeeze(0)
        
        loss, loss_dict = criterion(logits, y_qry, h_orth, W_orth, z_all)
        
        if torch.isnan(loss) or loss.item() == 0.0:
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_task += loss_dict['task'].item()
        total_orth += loss_dict['orth'].item()
        total_indep += loss_dict['indep'].item()
        valid_steps += 1
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    if valid_steps == 0:
        return 0, 0, 0, 0
        
    return total_loss/valid_steps, total_task/valid_steps, total_orth/valid_steps, total_indep/valid_steps

def evaluate_all(adapter, mantis_model, tabicl_ckpt, reader, datasets, device):
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
        
        # Get Mantis Embeddings
        z_train = get_mantis_embeddings(mantis_model, X_train, device=device).to(device)
        z_test = get_mantis_embeddings(mantis_model, X_test, device=device).to(device)
        
        # Apply Adapter
        with torch.no_grad():
            x_train_tab = adapter(z_train, training_mode=False).cpu().numpy()
            x_test_tab = adapter(z_test, training_mode=False).cpu().numpy()
            
        x_train_tab = np.nan_to_num(x_train_tab)
        x_test_tab = np.nan_to_num(x_test_tab)
            
        # TabICL Classification
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
    parser.add_argument("--save_adapter", type=str, default="./checkpoints/causal_adapter_ucr_all.pt")
    parser.add_argument("--output_file", type=str, default="evaluation_results/causal_adapter_ucr_all_results.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    
    set_seed(args.seed)
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
    
    # 2. Initialize Adapter
    adapter = RobustCWA(input_dim=mantis_model.hidden_dim, hidden_dim=128, output_dim=64).to(device)
    criterion = RobustCausalLoss(lambda_orth=0.01, lambda_indep=0.01).to(device)
    optimizer = optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 3. Prepare Data Cache (Train Only)
    print("Pre-computing Mantis embeddings for UCR Training datasets...")
    reader = DataReader(UCR_data_path=args.ucr_path, UEA_data_path=args.uea_path)
    ucr_datasets = reader.dataset_list_ucr
    
    train_cache = {}
    for name in tqdm(ucr_datasets):
        try:
            X_train_raw, y_train_raw = reader.read_dataset(name, which_set="train")
            
            if len(X_train_raw) > 5000: 
                indices = np.random.choice(len(X_train_raw), 5000, replace=False)
                X_train_raw = X_train_raw[indices]
                y_train_raw = y_train_raw[indices]
                
            X_train = resize_series(_ensure_three_dim(X_train_raw))
            y_train = torch.from_numpy(y_train_raw).long()
            
            emb = get_mantis_embeddings(mantis_model, X_train, device=device)
            train_cache[name] = (emb, y_train)
        except Exception:
            pass
            
    print(f"Cached {len(train_cache)} training datasets.")
    
    # 4. Training Loop
    print(f"Starting Pre-training on ALL datasets...")
    
    for epoch in range(args.epochs):
        loss, task, orth, indep = train_epoch(
            adapter, tabicl_model, criterion, optimizer, train_cache, device, 
            steps_per_epoch=args.steps_per_epoch
        )
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{args.epochs} | lr: {current_lr:.6f} | Loss: {loss:.4f} (Task: {task:.4f}, Orth: {orth:.4f}, Indep: {indep:.4f})")
        
        if (epoch + 1) % 5 == 0:
            os.makedirs(os.path.dirname(args.save_adapter), exist_ok=True)
            torch.save(adapter.state_dict(), args.save_adapter)
            
    torch.save(adapter.state_dict(), args.save_adapter)
    print(f"Training Complete. Adapter saved to {args.save_adapter}")

    # 5. Evaluation
    results = evaluate_all(adapter, mantis_model, args.tabicl_ckpt, reader, ucr_datasets, device)
    
    avg_acc = np.mean(list(results.values()))
    print(f"\nAverage UCR Accuracy: {avg_acc:.4f}")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
