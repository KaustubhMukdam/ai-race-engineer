"""
LSTM Trainer - Tire Degradation Prediction
Matches existing multi-race architecture with Race_Encoded
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt

from ml.training.pipeline.base_trainer import BaseTrainer
from ml.models.tire_degradation_lstm import TireDegradationLSTM, TireDegradationDataset
from ml.training.evaluation.metrics import RegressionMetrics
from config.app_config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

class LSTMTrainer(BaseTrainer):
    """LSTM model trainer - matches existing multi-race architecture"""
    
    def __init__(self, model_name: str = "TireDegradationLSTM", model_version: str = "2.0.0"):
        super().__init__(model_name, model_version)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.race_encoder = None
        self.compound_encoder = None
        self.driver_encoder = None
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(self, data_path: Path) -> Dict[str, Any]:
        """Load and prepare training data - MATCHES YOUR EXISTING ARCHITECTURE"""
        logger.info(f"Loading data from {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Check required columns - YOUR EXISTING FORMAT
        required_cols = ['Driver', 'LapNumber', 'LapTime_Seconds', 'Compound', 
                        'TyreLife', 'TrackTemp', 'AirTemp', 'Race']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # FILTER OUT WET TIRES (your approach)
        df = df[df['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])].copy()
        logger.info(f"Filtered to dry compounds: {len(df)} laps")
        
        # PER-RACE IQR FILTERING (your approach)
        original_count = len(df)
        filtered_dfs = []
        
        for race_name, race_group in df.groupby('Race'):
            Q1 = race_group['LapTime_Seconds'].quantile(0.25)
            Q3 = race_group['LapTime_Seconds'].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            race_filtered = race_group[
                (race_group['LapTime_Seconds'] >= lower_bound) &
                (race_group['LapTime_Seconds'] <= upper_bound)
            ]
            
            filtered_dfs.append(race_filtered)
            logger.info(f"  {race_name}: {len(race_filtered)}/{len(race_group)} laps (mean: {race_filtered['LapTime_Seconds'].mean():.1f}s)")
        
        df = pd.concat(filtered_dfs, ignore_index=True)
        removed = original_count - len(df)
        logger.info(f"Removed {removed} outliers using IQR per-race filtering")
        
        # ENCODE (your approach)
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        self.race_encoder = LabelEncoder()
        self.compound_encoder = LabelEncoder()
        self.driver_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        df['Race_Encoded'] = self.race_encoder.fit_transform(df['Race'])
        df['Compound_Encoded'] = self.compound_encoder.fit_transform(df['Compound'])
        df['Driver_Encoded'] = self.driver_encoder.fit_transform(df['Driver'])
        
        # Previous lap time
        df['PrevLapTime'] = df.groupby(['Driver', 'Race'])['LapTime_Seconds'].shift(1)
        df['PrevLapTime'] = df['PrevLapTime'].fillna(df['LapTime_Seconds'])
        
        # 8 FEATURES (your architecture)
        feature_cols = [
            'LapNumber', 'TyreLife', 'TrackTemp', 'AirTemp',
            'Compound_Encoded', 'Driver_Encoded', 'Race_Encoded', 'PrevLapTime'
        ]
        target_col = 'LapTime_Seconds'
        
        df = df[feature_cols + [target_col, 'Race', 'Driver']].dropna()
        logger.info(f"Training on {df['Race'].nunique()} tracks: {df['Race'].unique().tolist()}")
        
        # FIT scalers
        scaled_features = self.feature_scaler.fit_transform(df[feature_cols])
        scaled_target = self.target_scaler.fit_transform(df[[target_col]]).flatten()
        
        # Create sequences (10-lap lookback)
        sequence_length = 10
        sequences, targets = [], []
        
        for _, group in df.groupby(['Driver', 'Race']):
            if len(group) < sequence_length + 1:
                continue
            
            for i in range(len(group) - sequence_length):
                seq_indices = group.iloc[i:i+sequence_length].index
                target_idx = group.iloc[i+sequence_length].name
                
                seq = scaled_features[df.index.get_indexer(seq_indices)]
                target = scaled_target[df.index.get_loc(target_idx)]
                
                sequences.append(seq)
                targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        logger.info(f"Created {len(sequences)} sequences of length {sequence_length}")
        
        # Split: 70% train, 15% val, 15% test
        train_size = int(0.7 * len(sequences))
        val_size = int(0.15 * len(sequences))
        
        X_train = sequences[:train_size]
        y_train = targets[:train_size]
        
        X_val = sequences[train_size:train_size+val_size]
        y_val = targets[train_size:train_size+val_size]
        
        X_test = sequences[train_size+val_size:]
        y_test = targets[train_size+val_size:]
        
        logger.info(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Convert to PyTorch datasets
        train_dataset = TireDegradationDataset(X_train, y_train)
        val_dataset = TireDegradationDataset(X_val, y_val)
        test_dataset = TireDegradationDataset(X_test, y_test)
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset,
            'input_size': len(feature_cols)
        }
    
    def build_model(self, config: Dict[str, Any]) -> TireDegradationLSTM:
        """Build LSTM model"""
        logger.info("Building LSTM model")
        logger.info(f"Config: {config}")
        
        self.model = TireDegradationLSTM(
            input_size=config.get('input_size', 8),
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.3)
        )
        
        self.model = self.model.to(self.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model
    
    def train(self, train_data: Any, val_data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train LSTM model"""
        batch_size = config.get('batch_size', 32)
        epochs = config.get('epochs', 100)
        learning_rate = config.get('learning_rate', 0.001)
        patience = config.get('patience', 15)
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info("="*80)
        logger.info("TRAINING STARTED")
        logger.info("="*80)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    outputs = self.model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.experiment_dir / 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load(self.experiment_dir / 'best_model.pth'))
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        
        # Plot training history
        self._plot_training_history(train_losses, val_losses)
        
        return {
            'train_loss': float(train_losses[-1]),
            'val_loss': float(best_val_loss),
            'epochs_trained': len(train_losses)
        }
    
    def evaluate(self, test_data: Any) -> Dict[str, Any]:
        """Evaluate model on test set"""
        logger.info("Evaluating model on test set...")
        
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
        
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch).squeeze()
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(y_batch.numpy())
        
        predictions = np.array(predictions).reshape(-1, 1)
        actuals = np.array(actuals).reshape(-1, 1)
        
        # Inverse transform
        predictions_original = self.target_scaler.inverse_transform(predictions).flatten()
        actuals_original = self.target_scaler.inverse_transform(actuals).flatten()
        
        # Calculate metrics
        metrics = RegressionMetrics.calculate_all(actuals_original, predictions_original)
        RegressionMetrics.print_metrics(metrics)
        
        # Plot predictions
        self._plot_predictions(actuals_original, predictions_original)
        
        return metrics
    
    def save_model(self, model_path: Path):
        """Save model and scalers"""
        import joblib
        
        # Save model
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save scalers
        scalers = {
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'race_encoder': self.race_encoder,
            'compound_encoder': self.compound_encoder,
            'driver_encoder': self.driver_encoder
        }
        scaler_path = self.experiment_dir / 'scalers.pkl'
        joblib.dump(scalers, scaler_path)
        logger.info(f"Scalers saved to {scaler_path}")
    
    def _plot_training_history(self, train_losses, val_losses):
        """Plot training history"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('LSTM Training History - Multi-Race')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = self.experiment_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history saved to {plot_path}")
        plt.close()
    
    def _plot_predictions(self, actuals, predictions):
        """Plot predictions vs actuals"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Scatter plot
        axes[0].scatter(actuals, predictions, alpha=0.5)
        axes[0].plot([actuals.min(), actuals.max()], 
                     [actuals.min(), actuals.max()], 'r--', linewidth=2)
        axes[0].set_xlabel('Actual Lap Time (s)')
        axes[0].set_ylabel('Predicted Lap Time (s)')
        axes[0].set_title('Predictions vs Actuals')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = predictions - actuals
        axes[1].scatter(actuals, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Actual Lap Time (s)')
        axes[1].set_ylabel('Residual (s)')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.experiment_dir / 'predictions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Predictions plot saved to {plot_path}")
        plt.close()
