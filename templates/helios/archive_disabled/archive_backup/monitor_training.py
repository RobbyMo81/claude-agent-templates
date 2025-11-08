#!/usr/bin/env python3
"""
Monitor training progress for GPU-optimized PowerBall model
"""
import os
import time
import json
from pathlib import Path

def check_model_files():
    """Check for completed model files"""
    models_dir = Path("backend/models")
    if models_dir.exists():
        pth_files = list(models_dir.glob("job_helios_gpu_optimized_powerball_2025_*.pth"))
        json_files = list(models_dir.glob("job_helios_gpu_optimized_powerball_2025_*.json"))
        
        print(f"Found {len(pth_files)} model files and {len(json_files)} training reports")
        
        if pth_files:
            latest_model = max(pth_files, key=os.path.getmtime)
            latest_report = None
            
            # Find corresponding JSON file
            base_name = latest_model.stem
            json_path = models_dir / f"{base_name}.json"
            if json_path.exists():
                latest_report = json_path
            
            print(f"\nLatest model: {latest_model}")
            if latest_report:
                print(f"Training report: {latest_report}")
                try:
                    with open(latest_report, 'r') as f:
                        report = json.load(f)
                    print(f"Training completed: {report.get('training_completed', False)}")
                    print(f"Final loss: {report.get('final_loss', 'N/A')}")
                    print(f"Total epochs: {report.get('epochs_trained', 'N/A')}")
                except Exception as e:
                    print(f"Could not read training report: {e}")
        return len(pth_files) > 0
    return False

def check_server_status():
    """Check if server is still running training"""
    try:
        import requests
        response = requests.get('http://localhost:5001/api/training/status', timeout=5)
        if response.status_code == 200:
            status = response.json()
            if status.get('is_training'):
                print(f"Training active: {status.get('current_job', 'Unknown job')}")
                return True
            else:
                print("No active training sessions")
                return False
    except Exception:
        print("Cannot connect to server (training may be in progress)")
        return True  # Assume training is still running if we can't connect

if __name__ == "__main__":
    print("🔍 Monitoring GPU-optimized PowerBall training...")
    print("=" * 60)
    
    while True:
        print(f"\nCheck at {time.strftime('%H:%M:%S')}:")
        
        # Check for completed models
        has_models = check_model_files()
        
        # Check server status
        server_training = check_server_status()
        
        if has_models and not server_training:
            print("\n✅ Training appears to be complete!")
            break
        elif not server_training and not has_models:
            print("\n❌ No training detected and no models found")
            break
        else:
            print("\n⏳ Training still in progress...")
            
        print("-" * 40)
        time.sleep(30)  # Check every 30 seconds
