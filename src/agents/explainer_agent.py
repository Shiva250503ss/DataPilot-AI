"""
DataPilot AI Pro - Explainer Agent
====================================
Generates model explanations for interpretability.

Explainability Methods:
- SHAP (Global & Local explanations)
- LIME (Instance-level)
- Feature importance rankings
- Natural language explanations via LLM
- Counterfactual analysis
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from loguru import logger

from .base_agent import BaseAgent


class ExplainerAgent(BaseAgent):
    """
    Explainer Agent - Makes models interpretable.
    
    Uses SHAP, LIME, and LLM to generate both technical
    and natural language explanations of model predictions.
    """
    
    def __init__(self, model_name: str = "llama3.1"):
        super().__init__(
            name="Explainer",
            description="Generates model explanations for interpretability",
            model_name=model_name,
        )
    
    async def execute(
        self,
        data: pd.DataFrame,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate model explanations.
        
        Args:
            data: Test data for explanations
            context: Pipeline context with trained model
            
        Returns:
            Explanations including SHAP values and NL summaries
        """
        self.update_state("explaining", 0.0, "Starting model explanation")
        
        model = context.get("modeling", {}).get("models", {}).get("ensemble")
        if model is None:
            model = list(context.get("modeling", {}).get("models", {}).values())[0]
        
        X_test = context.get("modeling", {}).get("test_data", {}).get("X")
        
        explanations = {}
        
        # Step 1: Feature importance
        self.update_state("explaining", 0.2, "Computing feature importance")
        explanations["feature_importance"] = self._compute_feature_importance(model, X_test)
        
        # Step 2: SHAP analysis
        self.update_state("explaining", 0.5, "Computing SHAP values")
        explanations["shap"] = self._compute_shap_values(model, X_test)
        
        # Step 3: Sample explanations
        self.update_state("explaining", 0.7, "Generating sample explanations")
        explanations["sample_explanations"] = await self._explain_samples(
            model, X_test, explanations["feature_importance"]
        )
        
        # Step 4: Natural language summary
        self.update_state("explaining", 0.9, "Generating natural language summary")
        explanations["nl_summary"] = await self._generate_nl_summary(
            explanations["feature_importance"],
            context,
        )
        
        result = {
            "explanations": explanations,
            "summary": self._generate_summary(explanations),
        }
        
        self.update_state("complete", 1.0, "Explanation complete")
        self.log_result(result)
        
        return result
    
    def _compute_feature_importance(
        self,
        model: Any,
        X: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute feature importance from model."""
        try:
            # Try to get feature importance
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_).flatten()
            else:
                # For ensemble, try first estimator
                if hasattr(model, "estimators_"):
                    first_est = model.estimators_[0]
                    if hasattr(first_est, "feature_importances_"):
                        importances = first_est.feature_importances_
                    else:
                        return {}
                else:
                    return {}
            
            importance_dict = dict(zip(X.columns, importances))
            sorted_importance = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
            
            return {k: round(float(v), 4) for k, v in sorted_importance.items()}
            
        except Exception as e:
            logger.error(f"Error computing feature importance: {e}")
            return {}
    
    def _compute_shap_values(
        self,
        model: Any,
        X: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compute SHAP values for model interpretation."""
        try:
            import shap
            
            # Sample data for SHAP (faster computation)
            X_sample = X.sample(min(100, len(X)), random_state=42)
            
            # Create explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Compute mean absolute SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_importance = dict(zip(X.columns, mean_abs_shap))
            
            return {
                "mean_shap_importance": {
                    k: round(float(v), 4)
                    for k, v in sorted(
                        shap_importance.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                },
                "computed": True,
            }
            
        except ImportError:
            logger.warning("SHAP not installed")
            return {"computed": False, "error": "SHAP not installed"}
        except Exception as e:
            logger.error(f"Error computing SHAP: {e}")
            return {"computed": False, "error": str(e)}
    
    async def _explain_samples(
        self,
        model: Any,
        X: pd.DataFrame,
        feature_importance: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Generate explanations for sample predictions."""
        explanations = []
        
        # Get top features
        top_features = list(feature_importance.keys())[:5]
        
        # Sample a few rows
        samples = X.sample(min(3, len(X)), random_state=42)
        
        for idx, row in samples.iterrows():
            pred = model.predict([row.values])[0]
            
            try:
                proba = model.predict_proba([row.values])[0]
                confidence = max(proba)
            except Exception:
                confidence = None
            
            # Get feature contributions
            contributions = {
                feat: round(float(row[feat]), 3)
                for feat in top_features
                if feat in row.index
            }
            
            explanations.append({
                "sample_index": int(idx),
                "prediction": int(pred),
                "confidence": round(float(confidence), 3) if confidence else None,
                "top_features": contributions,
            })
        
        return explanations
    
    async def _generate_nl_summary(
        self,
        feature_importance: Dict[str, float],
        context: Dict[str, Any],
    ) -> str:
        """Generate natural language summary of model."""
        top_features = list(feature_importance.keys())[:5]
        target_col = context.get("profile", {}).get("target", {}).get("column", "target")
        
        best_model = context.get("modeling", {}).get("best_model", "ensemble")
        metrics = context.get("modeling", {}).get("metrics", {}).get(best_model, {})
        f1 = metrics.get("f1_score", 0)
        
        prompt = f"""Generate a concise, business-friendly summary of this ML model:

Target Variable: {target_col}
Best Model: {best_model}
F1 Score: {f1:.2%}
Top 5 Important Features: {', '.join(top_features)}

Write 2-3 sentences explaining what features drive predictions and how well the model performs.
Use simple language a business stakeholder would understand.
"""
        
        try:
            summary = await self.ask_llm(prompt)
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating NL summary: {e}")
            return (
                f"The {best_model} model achieves {f1:.1%} F1 score. "
                f"Key predictive features are: {', '.join(top_features[:3])}."
            )
    
    def _generate_summary(self, explanations: Dict[str, Any]) -> str:
        """Generate explanation summary."""
        n_features = len(explanations.get("feature_importance", {}))
        shap_computed = explanations.get("shap", {}).get("computed", False)
        n_samples = len(explanations.get("sample_explanations", []))
        
        return (
            f"Analyzed {n_features} features | "
            f"SHAP: {'✓' if shap_computed else '✗'} | "
            f"{n_samples} sample explanations"
        )
