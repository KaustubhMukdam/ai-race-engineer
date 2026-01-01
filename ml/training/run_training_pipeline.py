"""
Main Training Pipeline Orchestrator
Runs comprehensive training and evaluation for all models
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.training.pipeline.lstm_trainer import LSTMTrainer
from ml.training.pipeline.xgboost_trainer import XGBoostTrainer
from config.app_config import settings
from utils.logger import setup_logger
import json

logger = setup_logger(__name__)

def run_lstm_pipeline():
    """Run LSTM training pipeline"""
    logger.info("🏎️  Starting LSTM Tire Degradation Pipeline")
    
    # Configuration
    config = {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'patience': 15
    }
    
    # Initialize trainer
    trainer = LSTMTrainer(
        model_name="TireDegradationLSTM",
        model_version="2.0.0"
    )
    
    # Run pipeline
    data_path = settings.base_dir / 'ml/datasets/multi_race_training_data.csv'
    metrics = trainer.run_pipeline(data_path, config)
    
    return metrics

def run_xgboost_pipeline():
    """Run XGBoost pit window classification pipeline"""
    logger.info("🏎️  Starting XGBoost Pit Window Pipeline")
    
    # Configuration
    config = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    # Initialize trainer
    trainer = XGBoostTrainer(
        model_name="PitWindowClassifier",
        model_version="2.0.0"
    )
    
    # Run pipeline
    data_path = settings.base_dir / 'ml/datasets/pit_window_features.csv'
    metrics = trainer.run_pipeline(data_path, config)
    
    return metrics

def compare_models():
    """Compare all trained models"""
    registry_path = project_root / 'ml/model_registry/models.json'
    
    if not registry_path.exists():
        logger.warning("No models found in registry")
        return
    
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    for model in registry['models'][-10:]:  # Last 10 models
        print(f"\n{model['model_name']} v{model['version']}")
        print(f"  Experiment: {model['experiment_name']}")
        print(f"  Timestamp: {model['timestamp']}")
        print(f"  Metrics:")
        for key, value in model['metrics'].items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
        print("-"*80)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("F1 AI RACE ENGINEER - TRAINING PIPELINE")
    print("="*80)
    
    # Run LSTM training
    print("\n\n")
    lstm_metrics = run_lstm_pipeline()
    
    # Run XGBoost training
    print("\n\n")
    xgb_metrics = run_xgboost_pipeline()
    
    # Compare all models
    print("\n\n")
    compare_models()
    
    print("\n" + "="*80)
    print("ALL PIPELINES COMPLETE!")
    print("="*80)
