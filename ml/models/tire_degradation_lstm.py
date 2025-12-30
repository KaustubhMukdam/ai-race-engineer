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
    """Wrapper class for training and inference - FIXED VERSION"""
    
    def __init__(self, model_path: Path = None):
        self.model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.compound_encoder = LabelEncoder()
        self.driver_encoder = LabelEncoder()
        self.race_encoder = LabelEncoder()  # 🔥 NEW: Track-specific encoding
        self.model_path = model_path or (settings.base_dir / 'ml' / 'saved_models' / 'tire_degradation_lstm.pth')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def prepare_training_data(
        self,
        laps_df: pd.DataFrame,
        sequence_length: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data - FIT scalers"""
        df = laps_df.copy()
        
        # FILTER OUT WET TIRES
        df = df[df['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])].copy()
        
        # 🔥 NEW: Per-race IQR filtering (removes safety car, pit laps, etc.)
        original_count = len(df)
        filtered_dfs = []
        
        for race_name, race_group in df.groupby('Race'):
            Q1 = race_group['LapTime_Seconds'].quantile(0.25)
            Q3 = race_group['LapTime_Seconds'].quantile(0.75)
            IQR = Q3 - Q1
            
            # Keep laps within 1.5 * IQR (standard outlier detection)
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            race_filtered = race_group[
                (race_group['LapTime_Seconds'] >= lower_bound) & 
                (race_group['LapTime_Seconds'] <= upper_bound)
            ]
            
            filtered_dfs.append(race_filtered)
            logger.info(f"  {race_name}: {len(race_filtered)}/{len(race_group)} laps kept (mean: {race_filtered['LapTime_Seconds'].mean():.1f}s)")
        
        df = pd.concat(filtered_dfs, ignore_index=True)
        removed = original_count - len(df)
        logger.info(f"Filtered to {len(df)} laps (removed {removed} outliers using IQR method)")
        
        # 🔥 NEW: Encode Race/Track (so model learns per-circuit baselines)
        df['Race_Encoded'] = self.race_encoder.fit_transform(df['Race'])
        
        # Encode other categoricals
        df['Compound_Encoded'] = self.compound_encoder.fit_transform(df['Compound'])
        df['Driver_Encoded'] = self.driver_encoder.fit_transform(df['Driver'])
        
        # Previous lap time
        df['PrevLapTime'] = df.groupby(['Driver', 'Stint', 'Race'])['LapTime_Seconds'].shift(1)
        df['PrevLapTime'] = df['PrevLapTime'].fillna(df['LapTime_Seconds'])
        
        # 🔥 UPDATED: Now 8 features instead of 7 (added Race_Encoded)
        feature_cols = [
            'LapNumber',
            'TyreLife',
            'TrackTemp',
            'AirTemp',
            'Compound_Encoded',
            'Driver_Encoded',
            'Race_Encoded',      # 🔥 NEW
            'PrevLapTime'
        ]
        target_col = 'LapTime_Seconds'
        
        # Keep Race for grouping
        df = df[feature_cols + [target_col, 'Race', 'Driver', 'Stint']].dropna()
        
        logger.info(f"Training on {df['Race'].nunique()} unique tracks: {df['Race'].unique().tolist()}")
        
        # FIT scalers
        scaled_features = self.feature_scaler.fit_transform(df[feature_cols])
        scaled_target = self.target_scaler.fit_transform(df[[target_col]]).flatten()
        
        # Create sequences
        sequences, targets = [], []
        
        for _, group in df.groupby(['Driver', 'Stint', 'Race']):
            if len(group) < sequence_length + 1:
                continue
            
            for i in range(len(group) - sequence_length):
                seq_indices = group.iloc[i:i+sequence_length].index
                target_idx = group.iloc[i+sequence_length].name
                
                seq = scaled_features[df.index.get_indexer(seq_indices)]
                target = scaled_target[df.index.get_loc(target_idx)]
                
                sequences.append(seq)
                targets.append(target)
        
        return np.array(sequences), np.array(targets)


    def prepare_test_data(
        self,
        laps_df: pd.DataFrame,
        sequence_length: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare test data - TRANSFORM ONLY"""
        df = laps_df.copy()
        
        # FILTER OUT WET TIRES
        df = df[df['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])].copy()
        
        # FIXED RANGE: 70-110s
        original_count = len(df)
        df = df[(df['LapTime_Seconds'] >= 70.0) & (df['LapTime_Seconds'] <= 110.0)].copy()
        removed = original_count - len(df)
        logger.info(f"Test data: {len(df)} laps (removed {removed} outliers)")
        
        # 🔥 TRANSFORM ONLY (use fitted encoder from training)
        df['Race_Encoded'] = self.race_encoder.transform(df['Race'])
        
        # TRANSFORM ONLY
        df['Compound_Encoded'] = self.compound_encoder.transform(df['Compound'])
        df['Driver_Encoded'] = self.driver_encoder.transform(df['Driver'])
        
        df['PrevLapTime'] = df.groupby(['Driver', 'Stint', 'Race'])['LapTime_Seconds'].shift(1)
        df['PrevLapTime'] = df['PrevLapTime'].fillna(df['LapTime_Seconds'])
        
        # 🔥 UPDATED: 8 features
        feature_cols = [
            'LapNumber',
            'TyreLife',
            'TrackTemp',
            'AirTemp',
            'Compound_Encoded',
            'Driver_Encoded',
            'Race_Encoded',      # 🔥 NEW
            'PrevLapTime'
        ]
        target_col = 'LapTime_Seconds'
        
        df = df[feature_cols + [target_col, 'Race', 'Driver', 'Stint']].dropna()
        
        logger.info(f"Test tracks: {df['Race'].unique().tolist()}")
        
        # TRANSFORM ONLY
        scaled_features = self.feature_scaler.transform(df[feature_cols])
        scaled_target = self.target_scaler.transform(df[[target_col]]).flatten()
        
        # Create sequences
        sequences, targets = [], []
        
        for _, group in df.groupby(['Driver', 'Stint', 'Race']):
            if len(group) < sequence_length + 1:
                continue
            
            for i in range(len(group) - sequence_length):
                seq_indices = group.iloc[i:i+sequence_length].index
                target_idx = group.iloc[i+sequence_length].name
                
                seq = scaled_features[df.index.get_indexer(seq_indices)]
                target = scaled_target[df.index.get_loc(target_idx)]
                
                sequences.append(seq)
                targets.append(target)
        
        return np.array(sequences), np.array(targets)


    def _create_sequences(
        self,
        scaled_features: np.ndarray,
        scaled_target: np.ndarray,
        df: pd.DataFrame,
        sequence_length: int
    ) -> Tuple[List, List]:
        """Helper to create sequences from scaled data"""
        sequences = []
        targets = []
        
        # Group by driver, compound, and race
        for (driver, compound, race), group in df.groupby(['Driver_Encoded', 'Compound_Encoded', 'Race']):
            if len(group) < sequence_length + 1:
                continue
            
            group_indices = group.index.tolist()
            
            for i in range(len(group_indices) - sequence_length):
                seq_start = df.index.get_loc(group_indices[i])
                seq_end = df.index.get_loc(group_indices[i + sequence_length - 1]) + 1
                target_idx = df.index.get_loc(group_indices[i + sequence_length])
                
                seq = scaled_features[seq_start:seq_end]
                target = scaled_target[target_idx]
                
                if len(seq) == sequence_length:
                    sequences.append(seq)
                    targets.append(target)
        
        return sequences, targets
    
    def train(
        self,
        laps_df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2
    ) -> Dict[str, List[float]]:
        """Train the LSTM model"""
        
        logger.info("Preparing training data with separate feature/target scaling...")
        sequences, targets = self.prepare_training_data(laps_df)  # Changed here
        
        logger.info(f"Created {len(sequences)} sequences of length {sequences.shape[1]}")
        
        # Rest stays the same...
        X_train, X_val, y_train, y_val = train_test_split(
            sequences, targets, test_size=validation_split, random_state=42
        )
        
        train_dataset = TireDegradationDataset(X_train, y_train)
        val_dataset = TireDegradationDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        input_size = sequences.shape[2]
        self.model = TireDegradationLSTM(input_size=input_size).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        
        logger.info("Starting training...")
        
        for epoch in range(epochs):
            # Training
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
            
            # Validation
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
            
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model()
                logger.info(f"Epoch {epoch+1}: New best model saved (val_loss: {avg_val_loss:.4f})")
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        logger.info(f"Training complete! Best validation loss: {best_val_loss:.4f}")
        
        return history

    
    def predict(self, sequence: np.ndarray) -> float:
        """
        Predict next lap time with proper inverse scaling
        
        Args:
            sequence: Scaled feature sequence (sequence_length, input_size)
        
        Returns:
            Predicted lap time in original seconds scale
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        self.model.eval()
        
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            prediction_scaled = self.model(sequence_tensor).item()
        
        # CRITICAL FIX: Inverse transform using target scaler
        prediction_original = self.target_scaler.inverse_transform([[prediction_scaled]])[0][0]
        
        return prediction_original
    
    def save_model(self):
        """Save model and ALL preprocessing objects"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch model
        torch.save(self.model.state_dict(), self.model_path)
        
        # Save preprocessing objects
        feature_scaler_path = self.model_path.parent / 'feature_scaler.pkl'
        target_scaler_path = self.model_path.parent / 'target_scaler.pkl'
        compound_encoder_path = self.model_path.parent / 'compound_encoder.pkl'
        driver_encoder_path = self.model_path.parent / 'driver_encoder.pkl'
        race_encoder_path = self.model_path.parent / 'race_encoder.pkl'  # 🔥 NEW
        
        joblib.dump(self.feature_scaler, feature_scaler_path)
        joblib.dump(self.target_scaler, target_scaler_path)
        joblib.dump(self.compound_encoder, compound_encoder_path)
        joblib.dump(self.driver_encoder, driver_encoder_path)
        joblib.dump(self.race_encoder, race_encoder_path)  # 🔥 NEW
        
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self, input_size: int = 8):  # 🔥 CHANGED: 7 → 8
        """Load trained model and ALL preprocessing objects"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = TireDegradationLSTM(input_size=input_size).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        
        # Load preprocessing objects
        feature_scaler_path = self.model_path.parent / 'feature_scaler.pkl'
        target_scaler_path = self.model_path.parent / 'target_scaler.pkl'
        compound_encoder_path = self.model_path.parent / 'compound_encoder.pkl'
        driver_encoder_path = self.model_path.parent / 'driver_encoder.pkl'
        race_encoder_path = self.model_path.parent / 'race_encoder.pkl'  # 🔥 NEW
        
        self.feature_scaler = joblib.load(feature_scaler_path)
        self.target_scaler = joblib.load(target_scaler_path)
        self.compound_encoder = joblib.load(compound_encoder_path)
        self.driver_encoder = joblib.load(driver_encoder_path)
        self.race_encoder = joblib.load(race_encoder_path)  # 🔥 NEW
        
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
    print("LSTM Model Training Complete!")
    print(f"Model saved to: {predictor.model_path}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
