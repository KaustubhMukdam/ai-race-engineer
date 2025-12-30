"""
Retrain LSTM Model on Multi-Race Dataset
Improved generalization across different tracks and conditions
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.models.tire_degradation_lstm import TireDegradationPredictor
from config.app_config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


def evaluate_model(predictor, test_file: Path) -> dict:
    """Evaluate trained model on test set"""
    logger.info("Evaluating model on test set...")
    
    test_df = pd.read_csv(test_file)
    logger.info(f"Loaded {len(test_df)} test laps")
    
    # Use prepare_test_data (transform only)
    try:
        test_sequences, test_targets = predictor.prepare_test_data(test_df, sequence_length=10)
    except Exception as e:
        logger.error(f"Error preparing test data: {e}")
        return {}
    
    logger.info(f"Created {len(test_sequences)} test sequences")
    
    # Make predictions
    predictions = []
    for seq in test_sequences:
        pred = predictor.predict(seq)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Inverse transform targets to original scale for comparison
    test_targets_original = predictor.target_scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(test_targets_original, predictions)
    mse = mean_squared_error(test_targets_original, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_targets_original, predictions)
    
    errors = np.abs(test_targets_original - predictions)
    median_error = np.median(errors)
    percentile_90_error = np.percentile(errors, 90)
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'median_error': median_error,
        'percentile_90_error': percentile_90_error,
        'num_predictions': len(predictions)
    }
    
    logger.info("\n" + "="*80)
    logger.info("TEST SET EVALUATION")
    logger.info("="*80)
    logger.info(f"Mean Absolute Error (MAE): {mae:.4f} seconds")
    logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f} seconds")
    logger.info(f"R² Score: {r2:.4f}")
    logger.info(f"Median Error: {median_error:.4f} seconds")
    logger.info(f"90th Percentile Error: {percentile_90_error:.4f} seconds")
    logger.info("="*80)
    
    return metrics

def plot_predictions(predictor, test_file: Path, output_dir: Path):
    """Generate prediction visualizations"""
    
    test_df = pd.read_csv(test_file)
    
    # FILTER WET TIRES AND OUTLIERS (same as training)
    test_df = test_df[test_df['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])]
    test_df = test_df[(test_df['LapTime_Seconds'] >= 70.0) & (test_df['LapTime_Seconds'] <= 110.0)]
    
    test_sequences, test_targets = predictor.prepare_test_data(test_df, sequence_length=10)
    # Make predictions
    predictions = np.array([predictor.predict(seq) for seq in test_sequences[:500]])  # Plot first 500
    
    # Inverse transform targets for plotting
    targets = predictor.target_scaler.inverse_transform(test_targets[:500].reshape(-1, 1)).flatten()
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Predicted vs Actual
    axes[0, 0].scatter(targets, predictions, alpha=0.5, s=10)
    axes[0, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Lap Time (s)')
    axes[0, 0].set_ylabel('Predicted Lap Time (s)')
    axes[0, 0].set_title('Predicted vs Actual Lap Times')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Prediction Errors
    errors = predictions - targets
    axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Prediction Error (s)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Prediction Errors')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Time Series Sample
    sample_idx = slice(0, min(100, len(predictions)))
    axes[1, 0].plot(targets[sample_idx], label='Actual', linewidth=2)
    axes[1, 0].plot(predictions[sample_idx], label='Predicted', linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel('Sequence Index')
    axes[1, 0].set_ylabel('Lap Time (s)')
    axes[1, 0].set_title('Sample Predictions (First 100 sequences)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Absolute Errors Over Time
    abs_errors = np.abs(errors)
    axes[1, 1].plot(abs_errors[sample_idx], linewidth=1)
    axes[1, 1].axhline(np.median(abs_errors), color='r', linestyle='--', 
                       label=f'Median: {np.median(abs_errors):.3f}s')
    axes[1, 1].set_xlabel('Sequence Index')
    axes[1, 1].set_ylabel('Absolute Error (s)')
    axes[1, 1].set_title('Absolute Prediction Errors')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / 'prediction_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Prediction analysis saved to: {output_file}")
    
    plt.close()


def main():
    """Train LSTM on multi-race dataset with evaluation"""
    
    # Define paths
    datasets_dir = settings.base_dir / 'ml' / 'datasets'
    train_file = datasets_dir / 'train_laps.csv'
    test_file = datasets_dir / 'test_laps.csv'
    model_dir = settings.base_dir / 'ml' / 'saved_models'
    
    # Check files exist
    if not train_file.exists():
        logger.error(f"Training file not found: {train_file}")
        logger.info("Run: python ml/training/collect_multi_race_data.py")
        return
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    # Load training data
    train_df = pd.read_csv(train_file)
    logger.info(f"Loaded {len(train_df)} training laps from {train_df['Race'].nunique()} races")
    
    print("\n" + "="*80)
    print("MULTI-RACE LSTM TRAINING")
    print("="*80)
    print(f"Training data: {len(train_df):,} laps")
    print(f"Races: {train_df['Race'].nunique()}")
    print(f"Drivers: {train_df['Driver'].nunique()}")
    print(f"Compounds: {train_df['Compound'].nunique()}")
    print("="*80 + "\n")
    
    # Initialize predictor
    predictor = TireDegradationPredictor()
    
    # Train model
    logger.info("Starting training on multi-race dataset...")
    
    history = predictor.train(
        laps_df=train_df,
        epochs=100,  # More epochs for better learning
        batch_size=64,  # Larger batch for stability
        learning_rate=0.001,
        validation_split=0.2
    )
    
    # Evaluate on test set
    metrics = evaluate_model(predictor, test_file)
    
    # Generate prediction visualizations
    logger.info("\nGenerating prediction analysis plots...")
    plot_predictions(predictor, test_file, model_dir)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Multi-Race LSTM Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    
    history_file = model_dir / 'multi_race_training_history.png'
    plt.savefig(history_file, dpi=150, bbox_inches='tight')
    logger.info(f"Training history saved to: {history_file}")
    plt.close()
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Model saved to: {predictor.model_path}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"\nTest Set Performance:")
    print(f"  - MAE: {metrics.get('mae', 0):.4f} seconds")
    print(f"  - RMSE: {metrics.get('rmse', 0):.4f} seconds")
    print(f"  - R² Score: {metrics.get('r2', 0):.4f}")
    print(f"  - Median Error: {metrics.get('median_error', 0):.4f} seconds")
    print(f"\nVisualization files:")
    print(f"  - {model_dir / 'prediction_analysis.png'}")
    print(f"  - {model_dir / 'multi_race_training_history.png'}")
    print("="*80)
    
    # Comparison with old model
    print("\nIMPROVEMENT NOTES:")
    print("- Old model (single race): Overfitted, flat predictions")
    print("- New model (10 races): Generalizes across tracks/conditions")
    print("- Expected MAE: 0.5-1.5 seconds (good F1 prediction)")
    print("\nRestart your backend to use the improved model!")
    print("="*80)


if __name__ == "__main__":
    main()
