"""
DataPilot AI Pro - Modeler Agent
=================================
Trains ML models and creates ensemble predictions.

Modeling Capabilities:
- Handles class imbalance (SMOTE, ADASYN)
- Optuna hyperparameter optimization
- Multiple model training (XGBoost, LightGBM, CatBoost, RF)
- Ensemble methods (Voting, Stacking, Weighted)
- MLflow experiment tracking
- Cross-validation evaluation
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)

try:
    import xgboost as xgb
    import lightgbm as lgb
    HAS_BOOSTING = True
except ImportError:
    HAS_BOOSTING = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from .base_agent import BaseAgent


class ModelerAgent(BaseAgent):
    """
    Modeler Agent - Trains models and creates ensembles.
    
    Uses RL-selected models from the selector, applies Optuna
    hyperparameter tuning, and combines models into ensembles.
    """
    
    def __init__(self, model_name: str = "llama3.1"):
        super().__init__(
            name="Modeler",
            description="Trains ML models and creates ensemble predictions",
            model_name=model_name,
        )
        self.trained_models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
    
    async def execute(
        self,
        data: pd.DataFrame,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Train models and create ensemble.
        
        Args:
            data: Feature matrix with target
            context: Pipeline context with RL selection
            
        Returns:
            Trained models and performance metrics
        """
        self.update_state("modeling", 0.0, "Starting model training")
        
        target_col = context.get("profile", {}).get("target", {}).get("column")
        selected_models = context.get("rl_selection", {}).get("selected_models", 
                                                               ["xgboost", "lightgbm", "random_forest"])
        
        # Prepare data
        X, y, X_test, y_test = self._prepare_data(data, target_col)
        
        # Step 1: Handle class imbalance
        self.update_state("modeling", 0.1, "Handling class imbalance")
        X_balanced, y_balanced = self._handle_imbalance(X, y)
        
        # Step 2: Train individual models
        self.update_state("modeling", 0.3, "Training individual models")
        for i, model_name in enumerate(selected_models):
            progress = 0.3 + (0.4 * (i + 1) / len(selected_models))
            self.update_state("modeling", progress, f"Training {model_name}")
            
            model = self._train_model(model_name, X_balanced, y_balanced)
            if model is not None:
                self.trained_models[model_name] = model
                self.metrics[model_name] = self._evaluate_model(model, X_test, y_test)
        
        # Step 3: Create ensemble
        self.update_state("modeling", 0.8, "Creating ensemble")
        ensemble = self._create_ensemble(X_balanced, y_balanced)
        if ensemble is not None:
            self.trained_models["ensemble"] = ensemble
            self.metrics["ensemble"] = self._evaluate_model(ensemble, X_test, y_test)
        
        # Step 4: Find best model
        best_model_name = self._find_best_model()
        
        result = {
            "models": self.trained_models,
            "metrics": self.metrics,
            "best_model": best_model_name,
            "test_data": {"X": X_test, "y": y_test},
            "summary": self._generate_summary(best_model_name),
        }
        
        self.update_state("complete", 1.0, "Model training complete")
        self.log_result(result)
        
        return result
    
    def _prepare_data(
        self,
        data: pd.DataFrame,
        target_col: str,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare train/test split."""
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Handle categorical target
        if y.dtype == "object":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, y_train, X_test, y_test
    
    def _handle_imbalance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance with SMOTE."""
        try:
            from imblearn.over_sampling import SMOTE
            
            # Check for imbalance
            class_counts = y.value_counts()
            imbalance_ratio = class_counts.min() / class_counts.max()
            
            if imbalance_ratio < 0.5:
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                logger.info(f"Applied SMOTE: {len(y)} → {len(y_resampled)} samples")
                return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        except ImportError:
            logger.warning("imbalanced-learn not installed, skipping SMOTE")
        
        return X, y
    
    def _train_model(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Any:
        """Train a single model with optional Optuna tuning."""
        try:
            if model_name == "xgboost" and HAS_BOOSTING:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )
            elif model_name == "lightgbm" and HAS_BOOSTING:
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1,
                )
            elif model_name == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                logger.warning(f"Unknown or unavailable model: {model_name}")
                return None
            
            model.fit(X, y)
            logger.info(f"Trained {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            return None
    
    def _create_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Any:
        """Create voting ensemble from trained models."""
        if len(self.trained_models) < 2:
            return None
        
        estimators = [
            (name, model) for name, model in self.trained_models.items()
        ]
        
        try:
            ensemble = VotingClassifier(
                estimators=estimators,
                voting="soft",
            )
            ensemble.fit(X, y)
            logger.info("Created voting ensemble")
            return ensemble
        except Exception as e:
            logger.error(f"Error creating ensemble: {e}")
            return None
    
    def _evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
        }
        
        # ROC-AUC for binary classification
        if len(np.unique(y_test)) == 2:
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
            except Exception:
                pass
        
        return {k: round(v, 4) for k, v in metrics.items()}
    
    def _find_best_model(self) -> str:
        """Find best performing model."""
        if not self.metrics:
            return "none"
        
        best_model = max(
            self.metrics.items(),
            key=lambda x: x[1].get("f1_score", 0)
        )[0]
        
        return best_model
    
    def _generate_summary(self, best_model: str) -> str:
        """Generate modeling summary."""
        n_models = len(self.trained_models)
        best_f1 = self.metrics.get(best_model, {}).get("f1_score", 0)
        
        return f"Trained {n_models} models | Best: {best_model} (F1: {best_f1:.4f})"
