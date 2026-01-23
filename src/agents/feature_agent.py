"""
DataPilot AI Pro - Feature Engineering Agent
=============================================
Creates and transforms features for optimal model performance.

Feature Operations:
- DateTime extraction (month, year, day_of_week)
- One-Hot encoding for categorical
- Target encoding for high-cardinality
- Log transform for skewed features
- Binning for continuous variables
- Feature scaling (StandardScaler)
- Feature selection
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

from .base_agent import BaseAgent


class FeatureAgent(BaseAgent):
    """
    Feature Engineering Agent - Creates optimal features for ML.
    
    Automatically engineers features based on data types and
    characteristics identified by the Profiler agent.
    """
    
    def __init__(self, model_name: str = "llama3.1"):
        super().__init__(
            name="FeatureEngineer",
            description="Engineers and selects optimal features for ML models",
            model_name=model_name,
        )
        self.feature_log: List[Dict[str, Any]] = []
        self.transformers: Dict[str, Any] = {}
    
    async def execute(
        self,
        data: pd.DataFrame,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Engineer features from cleaned data.
        
        Args:
            data: Cleaned DataFrame
            context: Pipeline context
            
        Returns:
            Feature matrix and transformation info
        """
        self.update_state("feature_engineering", 0.0, "Starting feature engineering")
        self.feature_log = []
        
        df = data.copy()
        target_col = context.get("profile", {}).get("target", {}).get("column")
        
        # Step 1: DateTime feature extraction
        self.update_state("feature_engineering", 0.2, "Extracting datetime features")
        df = self._extract_datetime_features(df)
        
        # Step 2: Encode categorical variables
        self.update_state("feature_engineering", 0.4, "Encoding categorical features")
        df = self._encode_categorical(df, target_col)
        
        # Step 3: Transform skewed features
        self.update_state("feature_engineering", 0.6, "Transforming skewed features")
        df = self._transform_skewed(df)
        
        # Step 4: Scale numeric features
        self.update_state("feature_engineering", 0.75, "Scaling features")
        df, feature_cols = self._scale_features(df, target_col)
        
        # Step 5: Feature selection
        self.update_state("feature_engineering", 0.9, "Selecting top features")
        selected_features = self._select_features(df, feature_cols, target_col)
        
        result = {
            "feature_matrix": df,
            "feature_columns": feature_cols,
            "selected_features": selected_features,
            "transformers": self.transformers,
            "feature_log": self.feature_log,
            "summary": self._generate_summary(data, df, selected_features),
        }
        
        self.update_state("complete", 1.0, "Feature engineering complete")
        self.log_result(result)
        
        return result
    
    def _extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from datetime columns."""
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        
        for col in datetime_cols:
            # Extract components
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df[f"{col}_quarter"] = df[col].dt.quarter
            
            # Drop original datetime column
            df = df.drop(columns=[col])
            
            self.feature_log.append({
                "operation": "datetime_extraction",
                "column": col,
                "new_features": [f"{col}_{x}" for x in ["year", "month", "day", "dayofweek", "quarter"]],
            })
            
            logger.info(f"Extracted datetime features from {col}")
        
        return df
    
    def _encode_categorical(
        self,
        df: pd.DataFrame,
        target_col: Optional[str],
    ) -> pd.DataFrame:
        """Encode categorical variables."""
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Exclude target from encoding if it's categorical
        if target_col and target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        for col in categorical_cols:
            n_unique = df[col].nunique()
            
            if n_unique == 2:
                # Binary encoding
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.transformers[f"{col}_label_encoder"] = le
                self.feature_log.append({
                    "operation": "label_encoding",
                    "column": col,
                    "n_classes": 2,
                })
            
            elif n_unique <= 10:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                self.feature_log.append({
                    "operation": "onehot_encoding",
                    "column": col,
                    "new_columns": list(dummies.columns),
                })
            
            else:
                # Target encoding for high cardinality (if target exists)
                if target_col and target_col in df.columns:
                    target_means = df.groupby(col)[target_col].mean()
                    df[f"{col}_encoded"] = df[col].map(target_means)
                    df = df.drop(columns=[col])
                    self.feature_log.append({
                        "operation": "target_encoding",
                        "column": col,
                        "new_column": f"{col}_encoded",
                    })
                else:
                    # Frequency encoding
                    freq = df[col].value_counts(normalize=True)
                    df[f"{col}_freq"] = df[col].map(freq)
                    df = df.drop(columns=[col])
                    self.feature_log.append({
                        "operation": "frequency_encoding",
                        "column": col,
                    })
        
        return df
    
    def _transform_skewed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log transform to highly skewed features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            skewness = df[col].skew()
            
            if abs(skewness) > 1 and df[col].min() >= 0:
                # Log1p transform for positive skewed data
                df[f"{col}_log"] = np.log1p(df[col])
                df = df.drop(columns=[col])
                
                self.feature_log.append({
                    "operation": "log_transform",
                    "column": col,
                    "original_skewness": round(skewness, 3),
                })
        
        return df
    
    def _scale_features(
        self,
        df: pd.DataFrame,
        target_col: Optional[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Scale numeric features using StandardScaler."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude target from scaling
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if numeric_cols:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.transformers["standard_scaler"] = scaler
            
            self.feature_log.append({
                "operation": "standard_scaling",
                "columns": numeric_cols,
            })
        
        return df, numeric_cols
    
    def _select_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: Optional[str],
        k: int = 20,
    ) -> List[str]:
        """Select top k features using mutual information."""
        if not target_col or target_col not in df.columns:
            return feature_cols
        
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        # Handle categorical target
        if y.dtype == "object":
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        k = min(k, len(feature_cols))
        
        try:
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            selector.fit(X, y)
            
            scores = dict(zip(feature_cols, selector.scores_))
            selected = sorted(scores, key=scores.get, reverse=True)[:k]
            
            self.feature_log.append({
                "operation": "feature_selection",
                "method": "mutual_info",
                "n_selected": len(selected),
                "top_features": selected[:5],
            })
            
            return selected
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            return feature_cols
    
    def _generate_summary(
        self,
        original: pd.DataFrame,
        transformed: pd.DataFrame,
        selected: List[str],
    ) -> str:
        """Generate feature engineering summary."""
        n_original = len(original.columns)
        n_transformed = len(transformed.columns)
        n_selected = len(selected)
        
        return (
            f"Features: {n_original} → {n_transformed} (after engineering) "
            f"→ {n_selected} selected | {len(self.feature_log)} transformations"
        )
