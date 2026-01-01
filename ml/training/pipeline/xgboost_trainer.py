"""
XGBoost Trainer - Pit Window Classification
Predicts optimal pit windows using multi-race data
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score
)
import matplotlib.pyplot as plt
from typing import Dict, Any
import joblib

from ml.training.pipeline.base_trainer import BaseTrainer
from config.app_config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

class XGBoostTrainer(BaseTrainer):
    """XGBoost trainer for pit window classification"""
    
    def __init__(self, model_name: str = "PitWindowClassifier", model_version: str = "2.0.0"):
        super().__init__(model_name, model_version)
        self.model = None
        self.compound_encoder = LabelEncoder()
        self.driver_encoder = LabelEncoder()
        self.feature_names = None
    
    def prepare_data(self, data_path: Path) -> Dict[str, Any]:
        """Load and prepare pit window training data"""
        logger.info(f"Loading data from {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Check required columns - YOUR EXACT FORMAT
        required_cols = [
            'tire_age', 'tire_compound', 'lap_number', 'race_progress',
            'position', 'track_temp', 'air_temp', 'degradation_rate', 
            'lap_time', 'should_pit'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Optional: Add driver encoding if 'driver' column exists
        if 'driver' in df.columns:
            df['driver_encoded'] = self.driver_encoder.fit_transform(df['driver'])
        
        # Encode tire compound
        df['tire_compound_encoded'] = self.compound_encoder.fit_transform(df['tire_compound'])
        
        # Define feature columns
        feature_cols = [
            'tire_age', 'tire_compound_encoded', 'lap_number', 'race_progress',
            'position', 'track_temp', 'air_temp', 'degradation_rate', 'lap_time'
        ]
        
        if 'driver_encoded' in df.columns:
            feature_cols.append('driver_encoded')
        
        self.feature_names = feature_cols
        
        # Features and target
        X = df[feature_cols]
        y = df['should_pit']
        
        # Split data: 70% train, 15% val, 15% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        logger.info(f"Training class distribution:")
        logger.info(f"  Stay out (0): {(y_train == 0).sum()} samples")
        logger.info(f"  Pit (1): {(y_train == 1).sum()} samples")
        
        # ✅ FIXED: Return in format expected by base_trainer
        return {
            'train': {'X': X_train, 'y': y_train},
            'val': {'X': X_val, 'y': y_val},
            'test': {'X': X_test, 'y': y_test},
            'input_size': len(feature_cols)
        }
    
    def build_model(self, config: Dict[str, Any]) -> xgb.XGBClassifier:
        """Build XGBoost classifier"""
        logger.info("Building XGBoost model")
        
        # Calculate scale_pos_weight from config if provided
        scale_pos_weight = config.get('scale_pos_weight', 3.0)
        
        self.model = xgb.XGBClassifier(
            n_estimators=config.get('n_estimators', 200),
            max_depth=config.get('max_depth', 6),
            learning_rate=config.get('learning_rate', 0.1),
            subsample=config.get('subsample', 0.8),
            colsample_bytree=config.get('colsample_bytree', 0.8),
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=10
        )
        
        logger.info(f"Model config: {config}")
        
        return self.model
    
    def train(self, train_data: Any, val_data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train XGBoost model"""
        # ✅ FIXED: Extract X and y from dicts
        X_train = train_data['X']
        y_train = train_data['y']
        X_val = val_data['X']
        y_val = val_data['y']
        
        # Calculate scale_pos_weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        config['scale_pos_weight'] = scale_pos_weight
        logger.info(f"Class imbalance ratio: {scale_pos_weight:.2f}")
        
        # Build model with config
        self.build_model(config)
        
        logger.info("="*80)
        logger.info("TRAINING XGBOOST PIT WINDOW CLASSIFIER")
        logger.info("="*80)
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate on validation set
        y_val_pred = self.model.predict(X_val)
        y_val_proba = self.model.predict_proba(X_val)[:, 1]
        
        val_f1 = f1_score(y_val, y_val_pred)
        val_roc_auc = roc_auc_score(y_val, y_val_proba)
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Validation F1 Score: {val_f1:.4f}")
        logger.info(f"Validation ROC-AUC: {val_roc_auc:.4f}")
        
        # Plot feature importance
        self._plot_feature_importance()
        
        return {
            'val_f1': float(val_f1),
            'val_roc_auc': float(val_roc_auc),
            'n_estimators': self.model.n_estimators
        }
    
    def evaluate(self, test_data: Any) -> Dict[str, Any]:
        """Evaluate model on test set"""
        logger.info("Evaluating model on test set...")
        
        # ✅ FIXED: Extract X and y from dict
        X_test = test_data['X']
        y_test = test_data['y']
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        logger.info("\n" + "="*80)
        logger.info("MODEL PERFORMANCE")
        logger.info("="*80)
        logger.info("\n" + classification_report(
            y_test, y_pred,
            target_names=['Stay Out', 'Pit Now'],
            digits=4
        ))
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
        
        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        logger.info(f"PR-AUC Score: {pr_auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info("\nConfusion Matrix:")
        logger.info(f"                Predicted")
        logger.info(f"                Stay    Pit")
        logger.info(f"Actual Stay  [{cm[0][0]:5d}  {cm[0][1]:4d}]")
        logger.info(f"       Pit   [{cm[1][0]:5d}  {cm[1][1]:4d}]")
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm)
        
        # Plot ROC and PR curves
        self._plot_roc_pr_curves(y_test, y_pred_proba)
        
        return {
            'accuracy': float((y_pred == y_test).mean()),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'f1_score': float(f1_score(y_test, y_pred))
        }
    
    def save_model(self, model_path: Path):
        """Save model and encoders"""
        # Save model as pickle (recommended)
        model_pkl_path = model_path.with_suffix('.pkl')
        joblib.dump(self.model, model_pkl_path)
        logger.info(f"Model saved to {model_pkl_path}")
        
        # Save encoders
        encoders = {
            'compound_encoder': self.compound_encoder,
            'driver_encoder': self.driver_encoder,
            'feature_names': self.feature_names
        }
        encoder_path = self.experiment_dir / 'encoders.pkl'
        joblib.dump(encoders, encoder_path)
        logger.info(f"Encoders saved to {encoder_path}")
    
    def _plot_feature_importance(self):
        """Plot feature importance"""
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nFeature Importance:")
        for _, row in importance_df.iterrows():
            logger.info(f"  {row['feature']:25s}: {row['importance']:.4f}")
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance')
        plt.title('XGBoost Feature Importance - Pit Window Classification')
        plt.tight_layout()
        
        plot_path = self.experiment_dir / 'feature_importance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {plot_path}")
        plt.close()
    
    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Stay Out', 'Pit Now'],
               yticklabels=['Stay Out', 'Pit Now'],
               title='Confusion Matrix - Pit Window Classification',
               ylabel='True label',
               xlabel='Predicted label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plot_path = self.experiment_dir / 'confusion_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {plot_path}")
        plt.close()
    
    def _plot_roc_pr_curves(self, y_test, y_pred_proba):
        """Plot ROC and Precision-Recall curves"""
        from sklearn.metrics import roc_curve
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc_val = auc(fpr, tpr)
        
        axes[0].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc_val:.4f})')
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2)
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc_val = auc(recall, precision)
        
        axes[1].plot(recall, precision, linewidth=2, label=f'PR (AUC = {pr_auc_val:.4f})')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.experiment_dir / 'roc_pr_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC/PR curves saved to {plot_path}")
        plt.close()
