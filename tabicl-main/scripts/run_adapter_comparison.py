import subprocess
import sys
import os
import json
import numpy as np
import time
from datetime import datetime

def run_comparison():
    # Configuration
    script_path = "src/tabicl/train/train_adapter.py"
    
    # GPU IDs - Change these if needed
    adapter_gpu = "cuda:0"
    baseline_gpu = "cuda:1"
    
    # Output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    adapter_output = os.path.join(output_dir, f"results_adapter_{timestamp}.json")
    baseline_output = os.path.join(output_dir, f"results_baseline_{timestamp}.json")
    report_output = os.path.join(output_dir, f"comparison_report_{timestamp}.txt")
    
    # Training arguments
    # Using mantis_batch_size=8 to be safe against OOM as per previous context
    common_args = [
        "--epochs", "10",
        "--lr", "1e-3",
        "--div_weight", "0.1",
        "--mantis_batch_size", "8"
    ]
    
    # Command 1: Adapter (Train)
    cmd_adapter = [
        sys.executable, script_path,
        "--device", adapter_gpu,
        "--output_file", adapter_output
    ] + common_args
    
    # Command 2: Baseline (No Adapter)
    cmd_baseline = [
        sys.executable, script_path,
        "--device", baseline_gpu,
        "--no_adapter",
        "--output_file", baseline_output
    ] + common_args
    
    print(f"[{datetime.now()}] Starting Adapter training on {adapter_gpu}...")
    print(f"Command: {' '.join(cmd_adapter)}")
    p1 = subprocess.Popen(cmd_adapter)
    
    print(f"[{datetime.now()}] Starting Baseline (No Adapter) on {baseline_gpu}...")
    print(f"Command: {' '.join(cmd_baseline)}")
    p2 = subprocess.Popen(cmd_baseline)
    
    # Wait for completion
    exit_code_1 = p1.wait()
    exit_code_2 = p2.wait()
    
    print(f"[{datetime.now()}] Both runs finished.")
    
    if exit_code_1 != 0 or exit_code_2 != 0:
        print("Error: One or both processes failed.")
        if exit_code_1 != 0: print(f"Adapter run failed with code {exit_code_1}")
        if exit_code_2 != 0: print(f"Baseline run failed with code {exit_code_2}")
        # We try to proceed if files exist, but likely they don't or are incomplete
    
    # Analyze results
    if not os.path.exists(adapter_output):
        print(f"Error: Adapter output file {adapter_output} not found.")
        return
    if not os.path.exists(baseline_output):
        print(f"Error: Baseline output file {baseline_output} not found.")
        return

    with open(adapter_output, 'r') as f:
        res_adapter = json.load(f)
    with open(baseline_output, 'r') as f:
        res_baseline = json.load(f)
        
    # Comparison Logic
    datasets = sorted(list(set(res_adapter.keys()) | set(res_baseline.keys())))
    
    better = []
    worse = []
    same = []
    
    lines = []
    lines.append("="*50)
    lines.append(f"Adapter vs Baseline Comparison Report ({timestamp})")
    lines.append("="*50)
    lines.append(f"{'Dataset':<40} | {'Adapter':<10} | {'Baseline':<10} | {'Diff':<10}")
    lines.append("-" * 80)
    
    adapter_vals = []
    baseline_vals = []
    
    for ds in datasets:
        acc_a = res_adapter.get(ds, 0.0)
        acc_b = res_baseline.get(ds, 0.0)
        diff = acc_a - acc_b
        
        adapter_vals.append(acc_a)
        baseline_vals.append(acc_b)
        
        lines.append(f"{ds:<40} | {acc_a:.4f}     | {acc_b:.4f}     | {diff:+.4f}")
        
        if diff > 0.0001:
            better.append((ds, diff))
        elif diff < -0.0001:
            worse.append((ds, diff))
        else:
            same.append(ds)
            
    lines.append("-" * 80)
    lines.append(f"Overall Average Adapter:  {np.mean(adapter_vals):.4f}")
    lines.append(f"Overall Average Baseline: {np.mean(baseline_vals):.4f}")
    lines.append(f"Overall Improvement:      {np.mean(adapter_vals) - np.mean(baseline_vals):+.4f}")
    lines.append("=" * 50)
    
    lines.append(f"\nSummary:")
    lines.append(f"Adapter Better: {len(better)} datasets")
    lines.append(f"Adapter Worse:  {len(worse)} datasets")
    lines.append(f"Same:           {len(same)} datasets")
    
    lines.append("\nTop Improvements (Adapter > Baseline):")
    for ds, diff in sorted(better, key=lambda x: x[1], reverse=True)[:10]:
        lines.append(f"  {ds}: {diff:+.4f}")
        
    lines.append("\nTop Regressions (Adapter < Baseline):")
    for ds, diff in sorted(worse, key=lambda x: x[1])[:10]:
        lines.append(f"  {ds}: {diff:+.4f}")
        
    report_text = "\n".join(lines)
    print(report_text)
    
    with open(report_output, 'w') as f:
        f.write(report_text)
    print(f"\nReport saved to {report_output}")

if __name__ == "__main__":
    run_comparison()
