import torch
import numpy as np
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from tabicl.model.mantis_tabicl import MantisTabICL
from tabicl.prior.data_reader import DataReader

# Constants
DEFAULT_TABICL_CKPT = "/data0/fangjuntao2025/tabicl-main/tabICLOrignCheckpoint/tabicl-classifier-v1.1-0506.ckpt"
DEFAULT_MANTIS_CKPT = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/Models/Mantis/Mantis_cheickpoint/"
FINE_TUNED_CKPT = "/data0/fangjuntao2025/tabicl-main/mantis_tabiclcheckpointsMixup/step-400.ckpt"

DEFAULT_UEA_PATH = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/"
DEFAULT_UCR_PATH = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/"

def load_model(fine_tuned_ckpt, device):
    print("Initializing MantisTabICL architecture...")
    # Initialize with base checkpoints to define architecture
    # Note: These base checkpoints are only used to initialize the structure.
    # The weights will be overwritten by the fine-tuned checkpoint.
    model = MantisTabICL(
        tabicl_checkpoint=DEFAULT_TABICL_CKPT,
        mantis_checkpoint=DEFAULT_MANTIS_CKPT,
        mantis_batch_size=64, 
        device=device
    )
    
    # Load fine-tuned weights
    print(f"Loading fine-tuned weights from {fine_tuned_ckpt}")
    checkpoint = torch.load(fine_tuned_ckpt, map_location="cpu")
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove DDP prefix 'module.' if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
        
    model.to(device)
    model.eval()
    return model

def evaluate_dataset(model, reader, dataset_name, device):
    try:
        # Read dataset
        X_train, y_train = reader.read_dataset(dataset_name, which_set='train')
        X_test, y_test = reader.read_dataset(dataset_name, which_set='test')
        
        # Preprocess X
        # DataReader returns (N, C, L) with L=512 (if transform_ts_size=512)
        # MantisTabICL expects (Batch, SeqLen, H)
        # We treat the dataset as a single batch (Batch=1)
        # SeqLen = N_train + N_test
        # H should be compatible with Mantis encoder.
        # Assuming Mantis takes flattened input or univariate 512.
        
        # Flatten channels if multivariate, or keep as is if Mantis handles it?
        # Based on previous context, we flatten to (N, -1)
        # X_train = X_train.reshape(X_train.shape[0], -1)
        # X_test = X_test.reshape(X_test.shape[0], -1)
        
        # Concatenate train and test to form the sequence
        # X_full = np.concatenate([X_train, X_test], axis=0)
        
        # # Convert to tensor
        # X_tensor = torch.from_numpy(X_full).float().to(device)
        # y_train_tensor = torch.from_numpy(y_train).to(device)
        
        # # Add batch dimension (Batch=1)
        # X_batch = X_tensor.unsqueeze(0) # (1, N_total, H)
        # y_train_batch = y_train_tensor.unsqueeze(0) # (1, N_train)
        
        # Forward pass
        with torch.no_grad():
            # model(X, y_train) returns logits for the query set (test part)
            logits = model(X_batch, y_train_batch)
            # logits shape: (1, N_test, n_classes)
            
            preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
            
        # Calculate accuracy
        acc = (preds == y_test).mean()
        return acc
        
    except torch.cuda.OutOfMemoryError:
        print(f"OOM for {dataset_name}")
        torch.cuda.empty_cache()
        return 0.0
    except Exception as e:
        print(f"Error evaluating {dataset_name}: {e}")
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Evaluate MantisTabICL on UCR/UEA")
    parser.add_argument("--ckpt", type=str, default=FINE_TUNED_CKPT, help="Path to fine-tuned checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--datasets", type=str, default="all", help="Comma separated list of datasets or 'all'")
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # Load model
    model = load_model(args.ckpt, device)
    
    # Initialize DataReader
    reader = DataReader(
        UEA_data_path=DEFAULT_UEA_PATH,
        UCR_data_path=DEFAULT_UCR_PATH,
        transform_ts_size=512,
        log_processing=False
    )
    
    # Get dataset list
    if args.datasets == "all":
        # Combine UCR and UEA
        datasets = sorted(list(reader.dataset_list_ucr) + list(reader.dataset_list_uea))
    else:
        datasets = args.datasets.split(",")

    print(f"Evaluating on {len(datasets)} datasets...")
    
    results = []
    for name in tqdm(datasets):
        acc = evaluate_dataset(model, reader, name, device)
        results.append((name, acc))
        print(f"{name}: {acc:.4f}")
        
    # Summary
    avg_acc = np.mean([acc for _, acc in results])
    print(f"\nAverage Accuracy: {avg_acc:.4f}")
    
    # Save results
    with open("evaluation_results/mantis_tabicl_eval.txt", "w") as f:
        for name, acc in results:
            f.write(f"{name}: {acc:.4f}\n")
        f.write(f"\nAverage: {avg_acc:.4f}\n")

if __name__ == "__main__":
    main()
