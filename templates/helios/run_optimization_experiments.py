"""
Optimization Experiment Runner for Helios PowerballNet
Systematically test different model configurations and optimizations.
Includes comprehensive data verification as the first step.
"""

from full_training_suite import PowerballTrainer
from logger import setup_logger, log_epoch, log_error
from data_verification import verify_powerball_data
import json
import os
import time
from datetime import datetime
import pandas as pd

class OptimizationExperiment:
    """Run systematic optimization experiments."""
    
    def __init__(self):
        self.results = []
        self.errors = []
        self.experiment_log = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
    def run_experiment(self, config_path):
        """Run a single optimization experiment with data verification."""
        print(f"\n🧪 Starting optimization experiment: {config_path}")
        
        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)
            
        experiment_name = config.get("experiment_name", "Unknown")
        print(f"📊 Experiment: {experiment_name}")
        
        # CRITICAL: Data verification first
        print(f"\n🔍 Data Verification for {experiment_name}")
        print("-" * 50)
        
        data_verification_passed = verify_powerball_data(config['data_source'])
        
        if not data_verification_passed:
            error_msg = f"Data verification failed for {experiment_name}"
            self.errors.append(error_msg)
            print(f"❌ {error_msg}")
            
            # Record failed experiment
            error_result = {
                "experiment_name": experiment_name,
                "config": config,
                "error": error_msg,
                "verification_failed": True,
                "timestamp": datetime.now().isoformat()
            }
            self.results.append(error_result)
            return
        
        print(f"✅ Data verification passed for {experiment_name}")
        
        # Setup experiment-specific logging
        log_file = f"logs/optimization_{experiment_name.lower().replace(' ', '_')}_{datetime.now().strftime('%H%M%S')}.txt"
        setup_logger(log_file)
        
        start_time = time.time()
        
        try:
            # Handle temporal validation if specified
            if config.get("temporal_validation", False):
                trainer = TemporalValidationTrainer(config)
            else:
                trainer = PowerballTrainer(config)
            
            # Run training
            train_losses, val_losses = trainer.train()
            
            experiment_time = time.time() - start_time
            
            # Collect results
            result = {
                "experiment_name": experiment_name,
                "config": config,
                "final_train_loss": train_losses[-1] if train_losses else None,
                "final_val_loss": val_losses[-1] if val_losses else None,
                "best_val_loss": trainer.best_val_loss,
                "best_epoch": trainer.best_epoch,
                "total_epochs": len(train_losses),
                "early_stopped": len(train_losses) < config["num_epochs"],
                "experiment_time_minutes": experiment_time / 60,
                "verification_passed": True,
                "timestamp": datetime.now().isoformat()
            }
            
            self.results.append(result)
            
            print(f"✅ {experiment_name} completed!")
            print(f"   Best Val Loss: {trainer.best_val_loss:.4f}")
            print(f"   Total Epochs: {len(train_losses)}")
            print(f"   Time: {experiment_time/60:.1f} minutes")
            
        except Exception as e:
            error_result = {
                "experiment_name": experiment_name,
                "config": config,
                "error": str(e),
                "verification_passed": True,
                "training_failed": True,
                "timestamp": datetime.now().isoformat()
            }
            self.results.append(error_result)
            print(f"❌ {experiment_name} failed: {e}")
            log_error(f"Experiment {experiment_name} failed: {e}")
            
    def save_results(self):
        """Save all experiment results."""
        with open(self.experiment_log, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"📊 Results saved to: {self.experiment_log}")
        
    def print_summary(self):
        """Print experiment summary."""
        print(f"\n📈 OPTIMIZATION EXPERIMENT SUMMARY")
        print(f"=" * 50)
        
        successful_experiments = [r for r in self.results if "error" not in r]
        
        if not successful_experiments:
            print("❌ No successful experiments to compare")
            return
            
        # Sort by best validation loss
        successful_experiments.sort(key=lambda x: x["best_val_loss"])
        
        print(f"🏆 BEST PERFORMING CONFIGURATIONS:")
        for i, result in enumerate(successful_experiments[:3]):
            print(f"{i+1}. {result['experiment_name']}")
            print(f"   Best Val Loss: {result['best_val_loss']:.4f}")
            print(f"   Epochs: {result['total_epochs']}")
            print(f"   Early Stop: {result['early_stopped']}")
            print()

class TemporalValidationTrainer(PowerballTrainer):
    """Enhanced trainer with temporal validation split."""
    
    def prepare_data(self):
        """Prepare data with temporal validation split."""
        # Load full dataset
        df = pd.read_csv(self.config['data_source'])
        
        # Get temporal validation size
        val_size = self.config.get('temporal_validation_size', 247)
        
        # Use last val_size draws for validation
        train_df = df[:-val_size].copy()
        val_df = df[-val_size:].copy()
        
        print(f"📊 Temporal split: Train={len(train_df)}, Val={len(val_df)}")
        
        # Create datasets with temporal split
        from full_training_suite import PowerballDataset
        from torch.utils.data import DataLoader
        
        train_dataset = PowerballDataset(train_df, self.config.get('sequence_length', 30))
        val_dataset = PowerballDataset(val_df, self.config.get('sequence_length', 30))
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.get('batch_size', 16),
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=False
        )
        
        print(f"✅ Temporal validation setup complete")

def main():
    """Run all optimization experiments with comprehensive data verification."""
    print("🔍 HELIOS OPTIMIZATION SUITE WITH DATA VERIFICATION")
    print("=" * 60)
    
    # STEP 1: Pre-flight data verification
    print("🔍 STEP 1: PRE-FLIGHT DATA VERIFICATION")
    print("-" * 40)
    
    # Use default data path from first config
    default_data_path = "C:/Users/RobMo/OneDrive/Documents/PowerBall/downloads/powerball2015_Aug2025.csv"
    
    if not verify_powerball_data(default_data_path):
        print("❌ CRITICAL: Data verification failed!")
        print("🛑 Cannot proceed with optimization experiments.")
        print("📋 Please resolve data issues before running experiments.")
        return False
    
    print("✅ Pre-flight data verification passed!")
    print("🚀 Proceeding with optimization experiments...")
    
    # STEP 2: Run experiments
    print(f"\n🧪 STEP 2: OPTIMIZATION EXPERIMENTS")
    print("-" * 40)
    
    experiment_runner = OptimizationExperiment()
    
    # List of experiments to run
    experiments = [
        "optimization_configs/config_dropout_increase.json",
        "optimization_configs/config_reduced_capacity.json", 
        "optimization_configs/config_shorter_sequence.json",
        "optimization_configs/config_temporal_validation.json"
    ]
    
    print(f"Total experiments queued: {len(experiments)}")
    
    for experiment_config in experiments:
        config_path = os.path.join(os.path.dirname(__file__), experiment_config)
        if os.path.exists(config_path):
            experiment_runner.run_experiment(config_path)
        else:
            print(f"⚠️  Config not found: {experiment_config}")
    
    # STEP 3: Results and summary
    print(f"\n📊 STEP 3: EXPERIMENT RESULTS")
    print("-" * 35)
    
    # Save and summarize results
    experiment_runner.save_results()
    experiment_runner.print_summary()
    
    return len(experiment_runner.errors) == 0

if __name__ == "__main__":
    main()
