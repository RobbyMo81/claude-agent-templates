"""
Full Training Suite for Helios PowerballNet
Comprehensive training with real neural network, validation, and all supervision modules.
"""

from logger import setup_logger, log_epoch, log_error
from data_verification import verify_powerball_data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from agent import PowerballNet

class PowerballDataset(Dataset):
    """Dataset class for Powerball lottery data."""
    
    def __init__(self, data_source, sequence_length=50):
        self.sequence_length = sequence_length
        
        # Load and preprocess data
        if isinstance(data_source, str):
            # File path provided
            df = pd.read_csv(data_source)
            print(f"✅ Loaded {len(df)} draws from {data_source}")
        else:
            # DataFrame provided
            df = data_source.copy()
            print(f"✅ Using provided DataFrame with {len(df)} draws")
        
        # Convert date and sort
        df['draw_date'] = pd.to_datetime(df['draw_date'])
        df = df.sort_values('draw_date').reset_index(drop=True)
        
        # Extract ball numbers
        white_balls = df[['wb1', 'wb2', 'wb3', 'wb4', 'wb5']].values
        powerball = df['pb'].values
        
        # Create sequences
        self.sequences = []
        self.targets_white = []
        self.targets_powerball = []
        
        for i in range(len(df) - sequence_length):
            # Input sequence
            seq_white = white_balls[i:i+sequence_length]
            seq_powerball = powerball[i:i+sequence_length]
            
            # Target (next draw)
            target_white = white_balls[i+sequence_length]
            target_powerball = powerball[i+sequence_length]
            
            self.sequences.append({
                'white_balls': torch.tensor(seq_white, dtype=torch.long),
                'powerball': torch.tensor(seq_powerball, dtype=torch.long)
            })
            self.targets_white.append(torch.tensor(target_white, dtype=torch.long))
            self.targets_powerball.append(torch.tensor(target_powerball, dtype=torch.long))
        
        print(f"✅ Created {len(self.sequences)} training sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'target_white': self.targets_white[idx],
            'target_powerball': self.targets_powerball[idx]
        }

class PowerballTrainer:
    """Full training suite with validation and monitoring."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Enhanced device information
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 Using GPU: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
            print(f"🔥 Device: {self.device}")
        else:
            print(f"🔥 Using device: {self.device} (CUDA not available)")
            print(f"💡 To enable GPU training, install CUDA-enabled PyTorch")
        
        # Initialize model
        self.model = PowerballNet(
            sequence_length=config.get('sequence_length', 50),
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.2),
            attention_heads=config.get('attention_heads', 8)
        ).to(self.device)
        
        # Initialize optimizer with weight decay
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=config.get("lr_patience", 3)
        )
        
        # Loss functions
        self.white_criterion = nn.CrossEntropyLoss()
        self.powerball_criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Early stopping setup
        self.early_stop_patience = config.get("early_stop_patience", 5)
        self.early_stop_counter = 0
        self.best_epoch = -1
        
    def prepare_data(self):
        """Prepare training and validation datasets."""
        dataset = PowerballDataset(
            self.config['data_source'], 
            sequence_length=self.config.get('sequence_length', 50)
        )
        
        # Train/validation split
        val_size = int(len(dataset) * self.config.get('validation_split', 0.2))
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.get('batch_size', 32),
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False
        )
        
        print(f"✅ Training samples: {train_size}, Validation samples: {val_size}")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            white_balls = batch['sequence']['white_balls'].to(self.device)  # (batch, seq, 5)
            powerball = batch['sequence']['powerball'].to(self.device).unsqueeze(-1)  # (batch, seq, 1)
            target_white = batch['target_white'].to(self.device)  # (batch, 5)
            target_powerball = batch['target_powerball'].to(self.device)  # (batch,)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(white_balls, powerball)
            
            # Calculate losses for each white ball position
            white_loss = 0
            for i in range(5):
                white_loss += self.white_criterion(outputs['white_balls'], target_white[:, i] - 1)
            white_loss /= 5
            
            # Calculate powerball loss
            powerball_loss = self.powerball_criterion(outputs['powerball'], target_powerball - 1)
            
            total_loss_batch = white_loss + powerball_loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.get("clip_grad_norm", 1.0)
            )
            
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
        
        return total_loss / len(self.train_loader)
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                white_balls = batch['sequence']['white_balls'].to(self.device)  # (batch, seq, 5)
                powerball = batch['sequence']['powerball'].to(self.device).unsqueeze(-1)  # (batch, seq, 1)
                target_white = batch['target_white'].to(self.device)  # (batch, 5)
                target_powerball = batch['target_powerball'].to(self.device)  # (batch,)
                
                # Forward pass
                outputs = self.model(white_balls, powerball)
                
                # Calculate losses for each white ball position
                white_loss = 0
                for i in range(5):
                    white_loss += self.white_criterion(outputs['white_balls'], target_white[:, i] - 1)
                white_loss /= 5
                
                # Calculate powerball loss
                powerball_loss = self.powerball_criterion(outputs['powerball'], target_powerball - 1)
                
                total_loss += (white_loss + powerball_loss).item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, checkpoint_dir):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f"powerball_model_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    def train(self):
        """Full training loop with monitoring."""
        print(f"🚀 Starting PowerballNet training for {self.config['num_epochs']} epochs...")
        
        checkpoint_dir = os.path.join(os.path.dirname(__file__), self.config['checkpoint_dir'])
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Prepare data
        self.prepare_data()
        
        # Training loop
        for epoch in range(self.config['num_epochs']):
            start_time = time.time()
            
            # Train and validate
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Log progress
            epoch_time = time.time() - start_time
            log_epoch(epoch, train_loss, val_loss)
            print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Time = {epoch_time:.2f}s")
            
            # Save checkpoint
            if epoch % self.config['checkpoint_interval'] == 0:
                checkpoint_path = self.save_checkpoint(epoch, checkpoint_dir)
                print(f"💾 Saved checkpoint: {checkpoint_path}")
            
            # Early stopping and best model logic
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.early_stop_counter = 0
                
                # Save best model
                best_path = self.save_checkpoint(epoch, checkpoint_dir)
                best_path_renamed = os.path.join(checkpoint_dir, "best_model.pth")
                
                # Remove existing best model if it exists
                if os.path.exists(best_path_renamed):
                    os.remove(best_path_renamed)
                
                os.rename(best_path, best_path_renamed)
                print(f"🏆 New best model saved: {best_path_renamed}")
            else:
                self.early_stop_counter += 1
                print(f"⏳ No improvement for {self.early_stop_counter}/{self.early_stop_patience} epochs")
                
                if self.early_stop_counter >= self.early_stop_patience:
                    print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                    print(f"📊 Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
                    log_error(f"Early stopping triggered at epoch {epoch+1}, best val loss: {self.best_val_loss:.4f}")
                    break
        
        print("✅ Training completed successfully!")
        return self.train_losses, self.val_losses

def main():
    """Main training function with integrated data verification."""
    # Load config
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "training_config.json")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    
    # Enhanced config for full training
    config.update({
        'sequence_length': 30,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'attention_heads': 4,
        'learning_rate': 0.001,
        'batch_size': 16,
        'validation_split': 0.2
    })
    
    # STEP 1: DATA VERIFICATION (CRITICAL FIRST STEP)
    print("🔍 STEP 1: DATA VERIFICATION")
    print("=" * 50)
    
    data_verification_passed = verify_powerball_data(config['data_source'])
    
    if not data_verification_passed:
        print("❌ DATA VERIFICATION FAILED!")
        print("🛑 Training cannot proceed with invalid data.")
        print("📋 Please check the verification report and resolve issues.")
        return False
    
    print("✅ Data verification passed! Proceeding with training...")
    
    # STEP 2: SETUP LOGGING
    print("\n🔧 STEP 2: TRAINING SETUP")
    print("-" * 30)
    setup_logger("training_log.txt")
    
    try:
        # STEP 3: INITIALIZE AND TRAIN
        print("🚀 STEP 3: TRAINING EXECUTION")
        print("-" * 35)
        
        # Initialize trainer
        trainer = PowerballTrainer(config)
        
        # Run training
        train_losses, val_losses = trainer.train()
        
        # STEP 4: RESULTS SUMMARY
        print(f"\n🎯 STEP 4: TRAINING RESULTS")
        print("=" * 40)
        print(f"✅ Training completed successfully!")
        print(f"🏆 Best Validation Loss: {trainer.best_val_loss:.4f}")
        print(f"📊 Final Training Loss: {train_losses[-1]:.4f}")
        print(f"📊 Final Validation Loss: {val_losses[-1]:.4f}")
        print(f"⏱️  Best Epoch: {trainer.best_epoch}")
        print(f"📈 Total Epochs: {len(train_losses)}")
        
        return True
        
    except Exception as e:
        log_error(str(e))
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
