"""
Comprehensive evaluation metrics for regression and classification
"""
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any
import pandas as pd

class RegressionMetrics:
    """Calculate comprehensive regression metrics"""
    
    @staticmethod
    def calculate_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all regression metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Median Absolute Error
        median_ae = np.median(np.abs(y_true - y_pred))
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'mape': float(mape),
            'median_ae': float(median_ae)
        }
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float]):
        """Pretty print metrics"""
        print("\n" + "="*60)
        print("REGRESSION METRICS")
        print("="*60)
        print(f"  Mean Absolute Error (MAE):     {metrics['mae']:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
        print(f"  R² Score:                       {metrics['r2_score']:.4f}")
        print(f"  Mean Absolute % Error (MAPE):   {metrics['mape']:.2f}%")
        print(f"  Median Absolute Error:          {metrics['median_ae']:.4f}")
        print("="*60)


class ClassificationMetrics:
    """Calculate comprehensive classification metrics"""
    
    @staticmethod
    def calculate_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> Dict[str, Any]:
        """Calculate all classification metrics"""
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Add ROC-AUC if probabilities provided
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba, average='weighted')
                metrics['roc_auc'] = float(roc_auc)
            except:
                pass
        
        # Detailed classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, Any]):
        """Pretty print metrics"""
        print("\n" + "="*60)
        print("CLASSIFICATION METRICS")
        print("="*60)
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print("\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(cm)
        print("="*60)
