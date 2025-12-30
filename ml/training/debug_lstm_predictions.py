"""
Debug LSTM Predictions - Analyze what's going wrong
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ml.models.tire_degradation_lstm import TireDegradationPredictor
from config.app_config import settings


def analyze_data_distribution():
    """Compare train vs test data distributions"""
    
    datasets_dir = settings.base_dir / 'ml' / 'datasets'
    train_df = pd.read_csv(datasets_dir / 'train_laps.csv')
    test_df = pd.read_csv(datasets_dir / 'test_laps.csv')
    
    # Filter wet tires (same as training)
    train_df = train_df[train_df['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])]
    test_df = test_df[test_df['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])]
    
    print("\n" + "="*80)
    print("DATA DISTRIBUTION ANALYSIS")
    print("="*80)
    
    print("\n📊 LAP TIME STATISTICS:")
    print(f"\nTraining Set:")
    print(f"  Mean: {train_df['LapTime_Seconds'].mean():.3f}s")
    print(f"  Std:  {train_df['LapTime_Seconds'].std():.3f}s")
    print(f"  Min:  {train_df['LapTime_Seconds'].min():.3f}s")
    print(f"  Max:  {train_df['LapTime_Seconds'].max():.3f}s")
    
    print(f"\nTest Set:")
    print(f"  Mean: {test_df['LapTime_Seconds'].mean():.3f}s")
    print(f"  Std:  {test_df['LapTime_Seconds'].std():.3f}s")
    print(f"  Min:  {test_df['LapTime_Seconds'].min():.3f}s")
    print(f"  Max:  {test_df['LapTime_Seconds'].max():.3f}s")
    
    print("\n🌡️ TEMPERATURE STATISTICS:")
    print(f"\nTraining Set:")
    print(f"  Track Temp: {train_df['TrackTemp'].mean():.1f}°C (±{train_df['TrackTemp'].std():.1f})")
    print(f"  Air Temp:   {train_df['AirTemp'].mean():.1f}°C (±{train_df['AirTemp'].std():.1f})")
    
    print(f"\nTest Set:")
    print(f"  Track Temp: {test_df['TrackTemp'].mean():.1f}°C (±{test_df['TrackTemp'].std():.1f})")
    print(f"  Air Temp:   {test_df['AirTemp'].mean():.1f}°C (±{test_df['AirTemp'].std():.1f})")
    
    print("\n🛞 COMPOUND DISTRIBUTION:")
    print("\nTraining Set:")
    print(train_df['Compound'].value_counts(normalize=True).to_string())
    print("\nTest Set:")
    print(test_df['Compound'].value_counts(normalize=True).to_string())
    
    print("\n🏁 RACES:")
    print(f"\nTraining: {train_df['Race'].unique().tolist()}")
    print(f"Test: {test_df['Race'].unique().tolist()}")
    
    print("\n🔧 TIRE AGE DISTRIBUTION:")
    print(f"\nTraining TyreLife: {train_df['TyreLife'].mean():.1f} laps (max: {train_df['TyreLife'].max()})")
    print(f"Test TyreLife:     {test_df['TyreLife'].mean():.1f} laps (max: {test_df['TyreLife'].max()})")
    
    # Visualize distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Lap Times
    axes[0, 0].hist(train_df['LapTime_Seconds'], bins=50, alpha=0.7, label='Train', edgecolor='black')
    axes[0, 0].hist(test_df['LapTime_Seconds'], bins=50, alpha=0.7, label='Test', edgecolor='black')
    axes[0, 0].set_xlabel('Lap Time (s)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Lap Time Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Track Temperature
    axes[0, 1].hist(train_df['TrackTemp'], bins=30, alpha=0.7, label='Train', edgecolor='black')
    axes[0, 1].hist(test_df['TrackTemp'], bins=30, alpha=0.7, label='Test', edgecolor='black')
    axes[0, 1].set_xlabel('Track Temp (°C)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Track Temperature Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Tire Life
    axes[0, 2].hist(train_df['TyreLife'], bins=40, alpha=0.7, label='Train', edgecolor='black')
    axes[0, 2].hist(test_df['TyreLife'], bins=40, alpha=0.7, label='Test', edgecolor='black')
    axes[0, 2].set_xlabel('Tire Life (laps)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Tire Life Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Lap Number
    axes[1, 0].hist(train_df['LapNumber'], bins=58, alpha=0.7, label='Train', edgecolor='black')
    axes[1, 0].hist(test_df['LapNumber'], bins=58, alpha=0.7, label='Test', edgecolor='black')
    axes[1, 0].set_xlabel('Lap Number')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Lap Number Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Compound breakdown
    train_compounds = train_df['Compound'].value_counts()
    test_compounds = test_df['Compound'].value_counts()

    compound_names = ['SOFT', 'MEDIUM', 'HARD']
    x = np.arange(len(compound_names))
    width = 0.35

    train_vals = [train_compounds.get(c, 0) for c in compound_names]
    test_vals = [test_compounds.get(c, 0) for c in compound_names]

    axes[1, 1].bar(x - width/2, train_vals, width, label='Train', alpha=0.7)
    axes[1, 1].bar(x + width/2, test_vals, width, label='Test', alpha=0.7)
    axes[1, 1].set_xlabel('Compound')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Compound Distribution')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(compound_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    
    # Avg lap time per race
    train_race_times = train_df.groupby('Race')['LapTime_Seconds'].mean().sort_values()
    test_race_times = test_df.groupby('Race')['LapTime_Seconds'].mean().sort_values()
    
    axes[1, 2].barh(range(len(train_race_times)), train_race_times.values, alpha=0.7, label='Train Races')
    axes[1, 2].barh(range(len(test_race_times)), test_race_times.values, alpha=0.7, label='Test Races', 
                    height=0.4, left=train_race_times.values.max() - test_race_times.values.max())
    axes[1, 2].set_yticks(range(len(train_race_times)))
    axes[1, 2].set_yticklabels([r.replace('2024_', '') for r in train_race_times.index], fontsize=8)
    axes[1, 2].set_xlabel('Avg Lap Time (s)')
    axes[1, 2].set_title('Average Lap Time by Race')
    axes[1, 2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    output_dir = settings.base_dir / 'ml' / 'saved_models'
    output_file = output_dir / 'data_distribution_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Distribution analysis saved to: {output_file}")
    plt.close()
    
    print("\n" + "="*80)


def test_sample_predictions():
    """Test model on specific samples to see actual vs predicted"""
    
    print("\n" + "="*80)
    print("SAMPLE PREDICTION ANALYSIS")
    print("="*80)
    
    # Load model
    predictor = TireDegradationPredictor()
    try:
        predictor.load_model()
    except:
        print("❌ Model not found. Train first.")
        return
    
    # Load test data
    datasets_dir = settings.base_dir / 'ml' / 'datasets'
    test_df = pd.read_csv(datasets_dir / 'test_laps.csv')
    test_df = test_df[test_df['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])]
    
    # Get sequences
    test_sequences, test_targets = predictor.prepare_test_data(test_df, sequence_length=10)
    
    # Test first 10 predictions
    print("\n📝 SAMPLE PREDICTIONS (first 10):\n")
    print(f"{'#':<5} {'Actual':<10} {'Predicted':<12} {'Error':<10} {'Error %':<10}")
    print("-" * 60)
    
    for i in range(min(10, len(test_sequences))):
        seq = test_sequences[i]
        actual_scaled = test_targets[i]
        actual = predictor.target_scaler.inverse_transform([[actual_scaled]])[0][0]
        
        predicted = predictor.predict(seq)
        error = predicted - actual
        error_pct = (error / actual) * 100
        
        print(f"{i+1:<5} {actual:<10.3f} {predicted:<12.3f} {error:<10.3f} {error_pct:<10.2f}%")
    
    print("\n" + "="*80)


def main():
    """Run all diagnostics"""
    print("\n🔍 LSTM DEBUGGING - SYSTEMATIC ANALYSIS")
    print("="*80)
    
    # Step 1: Data distribution
    analyze_data_distribution()
    
    # Step 2: Sample predictions
    test_sample_predictions()
    
    print("\n✅ Diagnostic analysis complete!")
    print("\nNext steps based on findings:")
    print("1. If lap times differ significantly: Add more diverse training data")
    print("2. If temperature ranges differ: Normalize temperature features")
    print("3. If test races are outliers: Include similar tracks in training")
    print("="*80)


if __name__ == "__main__":
    main()
