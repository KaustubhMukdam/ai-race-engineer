"""
XGBoost Classifier for Optimal Pit Window Prediction

Predicts whether a driver should pit within the next 3 laps
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
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
    auc
)
import joblib
from config.app_config import settings
from utils.logger import setup_logger
import pickle

logger = setup_logger(__name__)

class PitWindowClassifier:
    """XGBoost classifier for pit window prediction"""
    
    def __init__(self, model_path: Path = None):
        self.model = None
        self.compound_encoder = LabelEncoder()
        self.model_path = model_path or settings.base_dir / 'ml/saved_models/pit_window_xgb.json'
        self.encoder_path = self.model_path.parent / 'pit_window_encoder.pkl'
        self.feature_names = None        
    def train(
        self,
        features_df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> dict:
        """
        Train XGBoost classifier
        
        Args:
            features_df: DataFrame with features and 'should_pit' target
            test_size: Test set fraction
            random_state: Random seed
            
        Returns:
            Dictionary with training metrics
        """
        
        logger.info("="*80)
        logger.info("TRAINING XGBOOST PIT WINDOW CLASSIFIER")
        logger.info("="*80)
        
        # Encode tire compound
        features_df['tire_compound_encoded'] = self.compound_encoder.fit_transform(
            features_df['tire_compound']
        )
        
        # Features
        feature_cols = [
            'tire_age', 'tire_compound_encoded', 'lap_number', 'race_progress',
            'position', 'track_temp', 'air_temp', 'degradation_rate', 'lap_time'
        ]
        
        X = features_df[feature_cols]
        y = features_df['should_pit']
        
        self.feature_names = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"\nDataset Split:")
        logger.info(f"  Training set: {len(X_train)} samples")
        logger.info(f"  Test set: {len(X_test)} samples")
        logger.info(f"\nTraining class distribution:")
        logger.info(f"  Stay out (0): {(y_train == 0).sum()} samples")
        logger.info(f"  Pit (1): {(y_train == 1).sum()} samples")
        
        # Calculate class weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        logger.info(f"\nClass imbalance ratio: {scale_pos_weight:.2f}")
        
        # Train XGBoost
        logger.info("\nTraining XGBoost model...")
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,  # Handle class imbalance
            random_state=random_state,
            eval_metric='logloss',
            early_stopping_rounds=10
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
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
        
        # Precision-Recall AUC (better for imbalanced data)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        logger.info(f"PR-AUC Score: {pr_auc:.4f}")
        
        logger.info("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"                Predicted")
        logger.info(f"              Stay  Pit")
        logger.info(f"Actual Stay  [{cm[0][0]:5d} {cm[0][1]:4d}]")
        logger.info(f"       Pit   [{cm[1][0]:5d} {cm[1][1]:4d}]")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nFeature Importance:")
        for _, row in importance_df.iterrows():
            logger.info(f"  {row['feature']:20s}: {row['importance']:.4f}")
        
        return {
            'accuracy': (y_pred == y_test).mean(),
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'feature_importance': importance_df
        }
    
    def predict(self, features: dict, threshold: float = 0.35) -> dict:
        """
        Predict pit decision
        
        Args:
            features: Dict with feature values
            
        Returns:
            Dict with prediction and probability
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")
        
        # Encode compound
        compound_encoded = self.compound_encoder.transform([features['tire_compound']])[0]
        
        # Create feature array
        X = np.array([[
            features['tire_age'],
            compound_encoded,
            features['lap_number'],
            features['race_progress'],
            features['position'],
            features['track_temp'],
            features['air_temp'],
            features['degradation_rate'],
            features['lap_time']
        ]])
        
        # Predict
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        # SMART THRESHOLD: Adjust based on tire age and compound
        tire_age = features['tire_age']
        compound = features['tire_compound']
        degradation = features.get('degradation_rate', 0.05)
        
        # DYNAMIC THRESHOLD LOGIC
        if compound == 'SOFT':
            if tire_age > 15:
                threshold = 0.25  # Very aggressive
            else:
                threshold = 0.35
        
        elif compound == 'MEDIUM':
            if tire_age > 17:
                # 🔥 NEW: Consider degradation rate
                if degradation > 0.07:  # High degradation
                    threshold = 0.25  # Very aggressive
                else:
                    threshold = 0.30  # Aggressive
            else:
                threshold = 0.40  # Conservative when fresh
        
        elif compound == 'HARD':
            if tire_age > 25:
                threshold = 0.35  # Normal
            else:
                threshold = 0.40  # Conservative

        pred = 1 if proba[1] >= threshold else 0
        
        return {
            'should_pit': bool(pred),
            'pit_probability': float(proba[1]),
            'stay_out_probability': float(proba[0]),
            'confidence': float(max(proba)),
            'threshold_used': threshold,
            'reason': f"Tire: {compound}, Age: {tire_age}, Deg: {degradation:.4f}"
        }
    
    def save_model(self):
        """Save model and encoders using pickle for full compatibility"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Change to .pkl extension for pickle format
        model_pkl_path = self.model_path.with_suffix('.pkl')
        
        # Save entire XGBoost classifier with pickle (includes all attributes)
        joblib.dump(self.model, model_pkl_path)
        
        # Save encoder
        joblib.dump(self.compound_encoder, self.encoder_path)
        
        logger.info(f"\nModel saved to {model_pkl_path}")
        logger.info(f"Encoder saved to {self.encoder_path}")

    def load_model(self):
        """Load trained model and encoder"""
        # Try .pkl first (new format), then .json (old format)
        model_pkl_path = self.model_path.with_suffix('.pkl')
        
        if model_pkl_path.exists():
            # Load from pickle (recommended)
            self.model = joblib.load(model_pkl_path)
            logger.info(f"Model loaded from {model_pkl_path}")
        elif self.model_path.exists():
            # Fallback: Load from JSON (legacy)
            logger.warning("Loading from JSON format. Consider retraining for better compatibility.")
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(self.model_path))
            logger.info(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {model_pkl_path} or {self.model_path}")
        
        # Load encoder
        if not self.encoder_path.exists():
            raise FileNotFoundError(f"Encoder not found at {self.encoder_path}")
        
        self.compound_encoder = joblib.load(self.encoder_path)


