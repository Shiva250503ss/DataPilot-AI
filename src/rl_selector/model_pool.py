"""
DataPilot AI Pro - Model Pool
==============================
Registry of available ML models for the selection system.

Supported Models:
- XGBoost: Gradient boosting with regularization
- LightGBM: Fast histogram-based gradient boosting
- CatBoost: Native categorical feature handling
- Random Forest: Ensemble of decision trees
- Gradient Boosting: Scikit-learn gradient boosting
- Logistic Regression: Linear baseline
"""

from typing import Any, Dict, List, Optional, Callable
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


class ModelPool:
    """
    Registry of ML models available for selection.
    
    Provides factory functions for creating model instances
    with default hyperparameters suitable for most tasks.
    """
    
    def __init__(self):
        """Initialize the model pool with available models."""
        self._models: Dict[str, Dict[str, Any]] = {}
        self._register_models()
    
    def _register_models(self) -> None:
        """Register all available models."""
        # XGBoost
        if HAS_XGB:
            self._models["xgboost"] = {
                "name": "XGBoost",
                "factory": self._create_xgboost,
                "description": "Gradient boosting with L1/L2 regularization",
                "strengths": ["Class imbalance", "Missing values", "Feature importance"],
                "best_for": "Medium datasets, mixed features, imbalanced classes",
            }
        
        # LightGBM
        if HAS_LGB:
            self._models["lightgbm"] = {
                "name": "LightGBM",
                "factory": self._create_lightgbm,
                "description": "Fast histogram-based gradient boosting",
                "strengths": ["Speed", "Large datasets", "Many features"],
                "best_for": "Large datasets (>10K samples), fast training needed",
            }
        
        # CatBoost
        if HAS_CAT:
            self._models["catboost"] = {
                "name": "CatBoost",
                "factory": self._create_catboost,
                "description": "Gradient boosting with native categorical handling",
                "strengths": ["Categorical features", "Ordered boosting", "GPU support"],
                "best_for": "High cardinality categorical features",
            }
        
        # Random Forest (always available via sklearn)
        self._models["random_forest"] = {
            "name": "Random Forest",
            "factory": self._create_random_forest,
            "description": "Ensemble of decision trees with bagging",
            "strengths": ["Robustness", "Interpretability", "Noise tolerance"],
            "best_for": "Small datasets, high noise, need interpretability",
        }
        
        # Gradient Boosting (sklearn)
        self._models["gradient_boosting"] = {
            "name": "Gradient Boosting",
            "factory": self._create_gradient_boosting,
            "description": "Scikit-learn gradient boosting",
            "strengths": ["Solid baseline", "No extra dependencies"],
            "best_for": "General purpose, when boosting libs unavailable",
        }
        
        # Logistic Regression
        self._models["logistic_regression"] = {
            "name": "Logistic Regression",
            "factory": self._create_logistic_regression,
            "description": "Linear baseline model",
            "strengths": ["Speed", "Interpretability", "Probability calibration"],
            "best_for": "Baseline, linearly separable problems",
        }
        
        logger.info(f"Registered {len(self._models)} models in pool")
    
    def get_model_names(self) -> List[str]:
        """Get list of available model names."""
        return list(self._models.keys())
    
    def get_model(self, name: str, **kwargs) -> Any:
        """
        Get a model instance by name.
        
        Args:
            name: Model name
            **kwargs: Override default hyperparameters
            
        Returns:
            Model instance
        """
        if name not in self._models:
            raise ValueError(f"Unknown model: {name}. Available: {self.get_model_names()}")
        
        factory = self._models[name]["factory"]
        return factory(**kwargs)
    
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """Get model information and metadata."""
        if name not in self._models:
            return {}
        
        info = self._models[name].copy()
        info.pop("factory", None)
        return info
    
    # Factory methods
    def _create_xgboost(self, **kwargs) -> Any:
        """Create XGBoost classifier."""
        defaults = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }
        defaults.update(kwargs)
        return xgb.XGBClassifier(**defaults)
    
    def _create_lightgbm(self, **kwargs) -> Any:
        """Create LightGBM classifier."""
        defaults = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
        }
        defaults.update(kwargs)
        return lgb.LGBMClassifier(**defaults)
    
    def _create_catboost(self, **kwargs) -> Any:
        """Create CatBoost classifier."""
        defaults = {
            "iterations": 100,
            "depth": 6,
            "learning_rate": 0.1,
            "random_seed": 42,
            "verbose": False,
        }
        defaults.update(kwargs)
        return cb.CatBoostClassifier(**defaults)
    
    def _create_random_forest(self, **kwargs) -> Any:
        """Create Random Forest classifier."""
        defaults = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42,
            "n_jobs": -1,
        }
        defaults.update(kwargs)
        return RandomForestClassifier(**defaults)
    
    def _create_gradient_boosting(self, **kwargs) -> Any:
        """Create Gradient Boosting classifier."""
        defaults = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42,
        }
        defaults.update(kwargs)
        return GradientBoostingClassifier(**defaults)
    
    def _create_logistic_regression(self, **kwargs) -> Any:
        """Create Logistic Regression classifier."""
        defaults = {
            "max_iter": 1000,
            "random_state": 42,
            "n_jobs": -1,
        }
        defaults.update(kwargs)
        return LogisticRegression(**defaults)
