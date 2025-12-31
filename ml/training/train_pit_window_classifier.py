"""
Train XGBoost Pit Window Classifier
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from ml.models.pit_window_classifier import PitWindowClassifier
from config.app_config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    """Train pit window classifier"""
    
    # Load prepared features
    features_path = settings.base_dir / 'ml/datasets/pit_window_features.csv'
    
    if not features_path.exists():
        logger.error(f"Features not found at {features_path}")
        logger.info("Please run prepare_pit_window_data.py first")
        return
    
    logger.info(f"Loading features from {features_path}")
    features_df = pd.read_csv(features_path)
    
    logger.info(f"Loaded {len(features_df)} samples")
    logger.info(f"Features shape: {features_df.shape}")
    
    # Train model
    classifier = PitWindowClassifier()
    results = classifier.train(features_df)
    
    # Save model
    classifier.save_model()
    
    print("\n" + "="*80)
    print("PIT WINDOW CLASSIFIER TRAINING COMPLETE!")
    print("="*80)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"PR-AUC: {results['pr_auc']:.4f}")
    print("="*80)
    print("\nTop 3 Most Important Features:")
    for i, row in results['feature_importance'].head(3).iterrows():
        print(f"   {i+1}. {row['feature']:20s}: {row['importance']:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
