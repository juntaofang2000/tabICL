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
        "--mantis_batch_size", "16",
        "--seed", "42"
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
        
    lines = []
    lines.append("="*80)
    lines.append(f"Adapter vs Baseline Comparison Report ({timestamp})")
    lines.append("="*80)

    def process_benchmark(name, adapter_dict, baseline_dict):
        if not adapter_dict and not baseline_dict:
            return None
            
        datasets = sorted(list(set(adapter_dict.keys()) | set(baseline_dict.keys())))
        
        lines.append(f"\n--- {name} Benchmark ({len(datasets)} datasets) ---")
        lines.append(f"{'Dataset':<40} | {'Adapter':<10} | {'Baseline':<10} | {'Diff':<10}")
        lines.append("-" * 80)
        
        vals_a = []
        vals_b = []
        better = []
        worse = []
        same = []
        
        for ds in datasets:
            acc_a = adapter_dict.get(ds, 0.0)
            acc_b = baseline_dict.get(ds, 0.0)
            diff = acc_a - acc_b
            
            vals_a.append(acc_a)
            vals_b.append(acc_b)
            
            lines.append(f"{ds:<40} | {acc_a:.4f}     | {acc_b:.4f}     | {diff:+.4f}")
            
            if diff > 0.0001:
                better.append((ds, diff))
            elif diff < -0.0001:
                worse.append((ds, diff))
            else:
                same.append(ds)
        
        avg_a = np.mean(vals_a) if vals_a else 0.0
        avg_b = np.mean(vals_b) if vals_b else 0.0
        
        lines.append("-" * 80)
        lines.append(f"Average {name} Adapter:  {avg_a:.4f}")
        lines.append(f"Average {name} Baseline: {avg_b:.4f}")
        lines.append(f"Average {name} Diff:     {avg_a - avg_b:+.4f}")
        
        return {
            "vals_a": vals_a, "vals_b": vals_b,
            "better": better, "worse": worse, "same": same
        }

    # Handle nested structure from train_adapter.py
    # Structure is expected to be {"UEA": {...}, "UCR": {...}}
    uea_stats = process_benchmark("UEA", res_adapter.get("UEA", {}), res_baseline.get("UEA", {}))
    ucr_stats = process_benchmark("UCR", res_adapter.get("UCR", {}), res_baseline.get("UCR", {}))
    
    # Overall Summary
    lines.append("\n" + "="*80)
    lines.append("FINAL SUMMARY")
    lines.append("="*80)
    
    all_vals_a = []
    all_vals_b = []
    all_better = []
    all_worse = []
    
    if uea_stats:
        all_vals_a.extend(uea_stats["vals_a"])
        all_vals_b.extend(uea_stats["vals_b"])
        all_better.extend(uea_stats["better"])
        all_worse.extend(uea_stats["worse"])
        
    if ucr_stats:
        all_vals_a.extend(ucr_stats["vals_a"])
        all_vals_b.extend(ucr_stats["vals_b"])
        all_better.extend(ucr_stats["better"])
        all_worse.extend(ucr_stats["worse"])
        
    if all_vals_a:
        avg_a = np.mean(all_vals_a)
        avg_b = np.mean(all_vals_b)
        lines.append(f"Overall Average Adapter:  {avg_a:.4f}")
        lines.append(f"Overall Average Baseline: {avg_b:.4f}")
        lines.append(f"Overall Improvement:      {avg_a - avg_b:+.4f}")
        lines.append(f"Total Better: {len(all_better)}")
        lines.append(f"Total Worse:  {len(all_worse)}")
        
    lines.append("\nTop Improvements (Adapter > Baseline):")
    for ds, diff in sorted(all_better, key=lambda x: x[1], reverse=True)[:10]:
        lines.append(f"  {ds}: {diff:+.4f}")
        
    lines.append("\nTop Regressions (Adapter < Baseline):")
    for ds, diff in sorted(all_worse, key=lambda x: x[1])[:10]:
        lines.append(f"  {ds}: {diff:+.4f}")

    report_text = "\n".join(lines)
    print(report_text)
    
    with open(report_output, 'w') as f:
        f.write(report_text)
    print(f"\nReport saved to {report_output}")

if __name__ == "__main__":
    run_comparison()
