"""
LSTM Model for Tire Degradation Prediction
Predicts lap-by-lap tire degradation based on historical patterns
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from typing import Tuple, Dict, List
import json

from config.app_config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class TireDegradationDataset(Dataset):
    """PyTorch dataset for tire degradation sequences"""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class TireDegradationLSTM(nn.Module):
    """
    LSTM network for predicting tire degradation
    
    Input features per timestep:
    - Lap number
    - Tire age
    - Track temperature
    - Air temperature
    - Compound (encoded)
    - Driver (encoded)
    - Previous lap time
    
    Output: Predicted lap time (degradation implicit)
    """
    
    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(TireDegradationLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        prediction = self.fc(last_output)
        
        return prediction


class TireDegradationPredictor:
    """Wrapper class for training and inference"""
    
    def __init__(self, model_path: Path = None):
        self.model = None
        self.scaler = StandardScaler()
        self.compound_encoder = LabelEncoder()
        self.driver_encoder = LabelEncoder()
        self.model_path = model_path or (settings.base_dir / 'ml' / 'saved_models' / 'tire_degradation_lstm.pth')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(
        self,
        laps_df: pd.DataFrame,
        sequence_length: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Prepare sequential data for LSTM training
        
        Args:
            laps_df: DataFrame with lap data
            sequence_length: Number of past laps to use as context
        
        Returns:
            sequences, targets, processed_df
        """
        # Feature engineering
        df = laps_df.copy()
        
        # Encode categorical features
        df['Compound_Encoded'] = self.compound_encoder.fit_transform(df['Compound'])
        df['Driver_Encoded'] = self.driver_encoder.fit_transform(df['Driver'])
        
        # Create previous lap time feature
        df['PrevLapTime'] = df.groupby(['Driver', 'Stint'])['LapTime_Seconds'].shift(1)
        df['PrevLapTime'] = df['PrevLapTime'].fillna(df['LapTime_Seconds'])
        
        # Select features
        feature_cols = [
            'LapNumber',
            'TyreLife',
            'TrackTemp',
            'AirTemp',
            'Compound_Encoded',
            'Driver_Encoded',
            'PrevLapTime'
        ]
        
        target_col = 'LapTime_Seconds'
        
        # Remove rows with missing values
        df = df[feature_cols + [target_col]].dropna()
        
        # Scale features
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        
        # Create sequences
        sequences = []
        targets = []
        
        # Group by driver and stint to maintain continuity
        for (driver, stint), group in df.groupby(['Driver_Encoded', 'Compound_Encoded']):
            if len(group) < sequence_length + 1:
                continue
            
            group_features = group[feature_cols].values
            group_targets = group[target_col].values
            
            for i in range(len(group) - sequence_length):
                seq = group_features[i:i+sequence_length]
                target = group_targets[i+sequence_length]
                
                sequences.append(seq)
                targets.append(target)
        
        return np.array(sequences), np.array(targets), df
    
    def train(
        self,
        laps_df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2
    ) -> Dict[str, List[float]]:
        """
        Train the LSTM model
        
        Returns:
            Dictionary with training history
        """
        logger.info("Preparing training data...")
        sequences, targets, processed_df = self.prepare_data(laps_df)
        
        logger.info(f"Created {len(sequences)} sequences of length {sequences.shape[1]}")
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            sequences, targets, test_size=validation_split, random_state=42
        )
        
        # Create datasets
        train_dataset = TireDegradationDataset(X_train, y_train)
        val_dataset = TireDegradationDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        input_size = sequences.shape[2]
        self.model = TireDegradationLSTM(input_size=input_size).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        current_lr = learning_rate

        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        
        logger.info("Starting training...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for batch_sequences, batch_targets in train_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_sequences).squeeze()
                loss = criterion(predictions, batch_targets)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_sequences, batch_targets in val_loader:
                    batch_sequences = batch_sequences.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    predictions = self.model(batch_sequences).squeeze()
                    loss = criterion(predictions, batch_targets)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)

            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                logger.info(f"Learning rate reduced: {current_lr} -> {new_lr}")
                current_lr = new_lr
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model()
                logger.info(f"✅ Epoch {epoch+1}: New best model saved (val_loss: {avg_val_loss:.4f})")
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        logger.info(f"Training complete! Best validation loss: {best_val_loss:.4f}")
        
        return history
    
    def predict(self, sequence: np.ndarray) -> float:
        """
        Predict next lap time given a sequence of previous laps
        
        Args:
            sequence: Array of shape (sequence_length, input_size)
        
        Returns:
            Predicted lap time
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        self.model.eval()
        
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            prediction = self.model(sequence_tensor).item()
        
        return prediction
    
    def save_model(self):
        """Save model and preprocessing objects"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch model
        torch.save(self.model.state_dict(), self.model_path)
        
        # Save preprocessing objects
        scaler_path = self.model_path.parent / 'scaler.pkl'
        compound_encoder_path = self.model_path.parent / 'compound_encoder.pkl'
        driver_encoder_path = self.model_path.parent / 'driver_encoder.pkl'
        
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.compound_encoder, compound_encoder_path)
        joblib.dump(self.driver_encoder, driver_encoder_path)
        
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self, input_size: int = 7):
        """Load trained model and preprocessing objects"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = TireDegradationLSTM(input_size=input_size).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        
        # Load preprocessing objects
        scaler_path = self.model_path.parent / 'scaler.pkl'
        compound_encoder_path = self.model_path.parent / 'compound_encoder.pkl'
        driver_encoder_path = self.model_path.parent / 'driver_encoder.pkl'
        
        self.scaler = joblib.load(scaler_path)
        self.compound_encoder = joblib.load(compound_encoder_path)
        self.driver_encoder = joblib.load(driver_encoder_path)
        
        logger.info(f"Model loaded from {self.model_path}")


def main():
    """Train LSTM model on Abu Dhabi 2024 data"""
    from config.app_config import settings
    
    # Load training data
    session_key = "2024_Abu_Dhabi_Grand_Prix_Race"
    processed_path = settings.processed_data_dir / f"{session_key}_processed"
    laps_file = processed_path / "processed_laps.csv"
    
    if not laps_file.exists():
        logger.error(f"Training data not found: {laps_file}")
        return
    
    laps_df = pd.read_csv(laps_file)
    logger.info(f"Loaded {len(laps_df)} laps from {session_key}")
    
    # Initialize predictor
    predictor = TireDegradationPredictor()
    
    # Train model
    history = predictor.train(
        laps_df=laps_df,
        epochs=50,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('LSTM Training History - Tire Degradation Prediction')
    plt.legend()
    plt.grid(True)
    
    plot_path = settings.base_dir / 'ml' / 'saved_models' / 'training_history.png'
    plt.savefig(plot_path)
    logger.info(f"Training history saved to {plot_path}")
    
    print("\n" + "="*80)
    print("✅ LSTM Model Training Complete!")
    print(f"Model saved to: {predictor.model_path}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
