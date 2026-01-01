"""
Base Trainer Class - Abstract interface for all model trainers
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
from datetime import datetime
from utils.logger import setup_logger

logger = setup_logger(__name__)

class BaseTrainer(ABC):
    """Abstract base class for model trainers"""
    
    def __init__(
        self,
        model_name: str,
        model_version: str = "1.0.0",
        experiment_name: Optional[str] = None
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.experiment_name = experiment_name or f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.metrics = {}
        
        # Create experiment directory
        self.experiment_dir = project_root / 'ml/model_registry/experiments' / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized {model_name} trainer - Experiment: {self.experiment_name}")
    
    @abstractmethod
    def prepare_data(self, data_path: Path) -> Dict[str, Any]:
        """Load and prepare training data"""
        pass
    
    @abstractmethod
    def build_model(self, config: Dict[str, Any]) -> Any:
        """Build model with given configuration"""
        pass
    
    @abstractmethod
    def train(self, train_data: Any, val_data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train the model"""
        pass
    
    @abstractmethod
    def evaluate(self, test_data: Any) -> Dict[str, Any]:
        """Evaluate model performance"""
        pass
    
    def save_model(self, model_path: Path):
        """Save trained model"""
        logger.info(f"Saving model to {model_path}")
    
    def save_experiment_metadata(self):
        """Save experiment configuration and results"""
        metadata = {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics
        }
        
        metadata_path = self.experiment_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Experiment metadata saved to {metadata_path}")
    
    def register_model(self):
        """Register model in model registry"""
        registry_path = project_root / 'ml/model_registry/models.json'
        
        # Load existing registry
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {'models': []}
        
        # Add new model entry
        model_entry = {
            'model_name': self.model_name,
            'version': self.model_version,
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics,
            'artifact_path': str(self.experiment_dir)
        }
        
        registry['models'].append(model_entry)
        
        # Save registry
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        logger.info(f"Model registered: {self.model_name} v{self.model_version}")
    
    def run_pipeline(self, data_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full training pipeline"""
        logger.info("="*80)
        logger.info(f"STARTING TRAINING PIPELINE: {self.model_name}")
        logger.info("="*80)
        
        # 1. Prepare data
        logger.info("Step 1: Preparing data...")
        data = self.prepare_data(data_path)
        
        # 2. Build model
        logger.info("Step 2: Building model...")
        model = self.build_model(config)
        
        # 3. Train model
        logger.info("Step 3: Training model...")
        train_results = self.train(data['train'], data['val'], config)
        self.metrics.update(train_results)
        
        # 4. Evaluate model
        logger.info("Step 4: Evaluating model...")
        eval_results = self.evaluate(data['test'])
        self.metrics.update(eval_results)
        
        # 5. Save artifacts
        logger.info("Step 5: Saving artifacts...")
        model_path = self.experiment_dir / f"{self.model_name}.pth"
        self.save_model(model_path)
        self.save_experiment_metadata()
        
        # 6. Register model
        logger.info("Step 6: Registering model...")
        self.register_model()
        
        logger.info("="*80)
        logger.info("TRAINING PIPELINE COMPLETE!")
        logger.info("="*80)
        
        return self.metrics
