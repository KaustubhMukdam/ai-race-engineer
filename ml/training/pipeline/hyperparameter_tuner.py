"""
Automated Hyperparameter Tuning using Random Search
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import RandomizedSearchCV
from typing import Dict, Any, Callable
import numpy as np
from utils.logger import setup_logger

logger = setup_logger(__name__)

class HyperparameterTuner:
    """Automated hyperparameter tuning with Random Search"""
    
    def __init__(
        self,
        model_builder: Callable,
        param_distributions: Dict[str, Any],
        n_iter: int = 20,
        cv: int = 3,
        scoring: str = 'neg_mean_squared_error',
        random_state: int = 42
    ):
        """
        Initialize tuner
        
        Args:
            model_builder: Function that returns model instance
            param_distributions: Dictionary of parameter distributions
            n_iter: Number of random samples to try
            cv: Cross-validation folds
            scoring: Scoring metric
            random_state: Random seed
        """
        self.model_builder = model_builder
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.best_params = None
        self.best_score = None
    
    def tune(self, X_train, y_train) -> Dict[str, Any]:
        """
        Run hyperparameter tuning
        
        Returns:
            Dictionary with best parameters and score
        """
        logger.info("="*80)
        logger.info("STARTING HYPERPARAMETER TUNING")
        logger.info("="*80)
        logger.info(f"Search space: {self.param_distributions}")
        logger.info(f"Iterations: {self.n_iter}, CV folds: {self.cv}")
        
        # Create base model
        model = self.model_builder()
        
        # Random search
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=self.param_distributions,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit
        random_search.fit(X_train, y_train)
        
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        
        logger.info("="*80)
        logger.info("TUNING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")
        
        return {
            'best_params': self.best_params,
            'best_score': float(self.best_score),
            'cv_results': random_search.cv_results_
        }
    
    @staticmethod
    def get_xgboost_param_grid() -> Dict[str, Any]:
        """Default hyperparameter grid for XGBoost"""
        return {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2]
        }
    
    @staticmethod
    def get_lstm_param_grid() -> Dict[str, Any]:
        """Default hyperparameter grid for LSTM"""
        return {
            'hidden_size': [32, 64, 128, 256],
            'num_layers': [1, 2, 3],
            'dropout': [0.1, 0.2, 0.3, 0.5],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [16, 32, 64],
            'sequence_length': [5, 10, 15, 20]
        }
