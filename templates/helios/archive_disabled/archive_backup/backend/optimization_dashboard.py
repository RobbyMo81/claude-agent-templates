"""
Optimization Results Dashboard
Monitor and compare optimization experiment results in real-time.
"""

import json
import os
from datetime import datetime
import glob

def load_checkpoint_results():
    """Load results from checkpoint directories."""
    experiments = {}
    
    checkpoint_dirs = [
        "checkpoints/dropout_experiment",
        "checkpoints/reduced_capacity_experiment", 
        "checkpoints/short_sequence_experiment",
        "checkpoints/temporal_validation_experiment"
    ]
    
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            # Find best model
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            if os.path.exists(best_model_path):
                # Get modification time
                mod_time = os.path.getmtime(best_model_path)
                mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
                
                # Count total checkpoints
                checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "powerball_model_epoch_*.pth"))
                total_checkpoints = len(checkpoint_files)
                
                exp_name = checkpoint_dir.split('/')[-1].replace('_experiment', '').replace('_', ' ').title()
                experiments[exp_name] = {
                    "directory": checkpoint_dir,
                    "has_best_model": True,
                    "last_updated": mod_time_str,
                    "total_checkpoints": total_checkpoints,
                    "estimated_epochs": total_checkpoints * 10 if total_checkpoints > 0 else 0
                }
    
    return experiments

def check_training_logs():
    """Check for active training logs."""
    log_files = glob.glob("logs/*.txt")
    active_logs = []
    
    for log_file in log_files:
        if os.path.exists(log_file):
            mod_time = os.path.getmtime(log_file)
            # If modified in last 5 minutes, consider active
            if (datetime.now().timestamp() - mod_time) < 300:
                active_logs.append({
                    "file": log_file,
                    "last_modified": datetime.fromtimestamp(mod_time).strftime("%H:%M:%S")
                })
    
    return active_logs

def print_dashboard():
    """Print the optimization dashboard."""
    print("\n" + "="*60)
    print("🧪 HELIOS OPTIMIZATION EXPERIMENTS DASHBOARD")
    print("="*60)
    print(f"⏰ Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load experiment results
    experiments = load_checkpoint_results()
    active_logs = check_training_logs()
    
    print(f"\n📊 EXPERIMENT STATUS:")
    print("-" * 40)
    
    if experiments:
        for exp_name, details in experiments.items():
            status = "🟢 COMPLETE" if details["total_checkpoints"] > 0 else "🟡 STARTING"
            print(f"{status} {exp_name}")
            print(f"   Last Updated: {details['last_updated']}")
            print(f"   Checkpoints: {details['total_checkpoints']}")
            print(f"   Est. Epochs: {details['estimated_epochs']}")
            print()
    else:
        print("❌ No experiments found")
    
    print(f"🔥 ACTIVE TRAINING LOGS:")
    print("-" * 40)
    if active_logs:
        for log in active_logs:
            print(f"📝 {log['file']} (Updated: {log['last_modified']})")
    else:
        print("💤 No active training sessions")
    
    print(f"\n💡 BASELINE COMPARISON:")
    print("-" * 40)
    baseline_best = os.path.join("checkpoints", "best_model.pth")
    if os.path.exists(baseline_best):
        mod_time = os.path.getmtime(baseline_best)
        print(f"🏆 Baseline Best Model: {datetime.fromtimestamp(mod_time).strftime('%H:%M:%S')}")
        print("   Original training with 30% dropout, 128 hidden dims")
    else:
        print("❌ No baseline model found")

def main():
    print_dashboard()

if __name__ == "__main__":
    main()
