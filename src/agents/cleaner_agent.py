"""
DataPilot AI Pro - Cleaner Agent
=================================
Handles data cleaning operations including missing values, outliers, and duplicates.

Cleaning Operations:
- Duplicate removal
- Missing value imputation (KNN, Mean, Mode)
- Outlier handling (Capping, IQR)
- Type conversion fixes
- Validation with Great Expectations
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder

from .base_agent import BaseAgent


class CleanerAgent(BaseAgent):
    """
    Cleaner Agent - Handles all data cleaning operations.
    
    Uses intelligent strategies based on data characteristics
    to clean missing values, outliers, and duplicates.
    """
    
    def __init__(self, model_name: str = "llama3.1"):
        super().__init__(
            name="Cleaner",
            description="Cleans data by handling missing values, outliers, and duplicates",
            model_name=model_name,
        )
        self.cleaning_log: List[Dict[str, Any]] = []
    
    async def execute(
        self,
        data: pd.DataFrame,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute cleaning operations based on profile.
        
        Args:
            data: DataFrame to clean
            context: Pipeline context with profile results
            
        Returns:
            Cleaned DataFrame and cleaning report
        """
        self.update_state("cleaning", 0.0, "Starting data cleaning")
        self.cleaning_log = []
        
        df = data.copy()
        profile = context.get("profile", {})
        
        # Step 1: Remove duplicates
        self.update_state("cleaning", 0.2, "Removing duplicates")
        df = self._remove_duplicates(df)
        
        # Step 2: Fix data types
        self.update_state("cleaning", 0.4, "Fixing data types")
        df = self._fix_data_types(df)
        
        # Step 3: Handle missing values
        self.update_state("cleaning", 0.6, "Imputing missing values")
        df = self._handle_missing_values(df, profile)
        
        # Step 4: Handle outliers
        self.update_state("cleaning", 0.8, "Handling outliers")
        df = self._handle_outliers(df)
        
        # Step 5: Validate cleaned data
        self.update_state("cleaning", 0.95, "Validating cleaned data")
        validation = self._validate_data(df)
        
        result = {
            "cleaned_data": df,
            "cleaning_log": self.cleaning_log,
            "validation": validation,
            "summary": self._generate_summary(data, df),
        }
        
        self.update_state("complete", 1.0, "Cleaning complete")
        self.log_result(result)
        
        return result
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        n_before = len(df)
        df = df.drop_duplicates()
        n_removed = n_before - len(df)
        
        if n_removed > 0:
            self.cleaning_log.append({
                "operation": "remove_duplicates",
                "rows_removed": n_removed,
                "percent_removed": round(n_removed / n_before * 100, 2),
            })
            logger.info(f"Removed {n_removed} duplicate rows")
        
        return df
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix incorrect data types."""
        for col in df.columns:
            # Try to convert object columns to numeric
            if df[col].dtype == "object":
                try:
                    # Check if it's a datetime
                    if any(keyword in col.lower() for keyword in ["date", "time", "created", "updated"]):
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                        self.cleaning_log.append({
                            "operation": "convert_to_datetime",
                            "column": col,
                        })
                    else:
                        # Try numeric conversion
                        numeric_series = pd.to_numeric(df[col], errors="coerce")
                        if numeric_series.notna().sum() / len(df) > 0.8:
                            df[col] = numeric_series
                            self.cleaning_log.append({
                                "operation": "convert_to_numeric",
                                "column": col,
                            })
                except Exception:
                    pass
        
        return df
    
    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        profile: Dict[str, Any],
    ) -> pd.DataFrame:
        """Handle missing values with appropriate strategies."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Handle numeric columns
        if numeric_cols:
            null_counts = df[numeric_cols].isnull().sum()
            cols_with_nulls = [col for col in numeric_cols if null_counts[col] > 0]
            
            if cols_with_nulls:
                # Use KNN imputation if correlations exist
                correlations = profile.get("correlations", {})
                high_corr = correlations.get("high_correlations", [])
                
                if high_corr and len(cols_with_nulls) <= 5:
                    # KNN imputation for correlated features
                    imputer = KNNImputer(n_neighbors=5)
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    self.cleaning_log.append({
                        "operation": "knn_imputation",
                        "columns": cols_with_nulls,
                    })
                else:
                    # Mean imputation for independent features
                    imputer = SimpleImputer(strategy="mean")
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    self.cleaning_log.append({
                        "operation": "mean_imputation",
                        "columns": cols_with_nulls,
                    })
        
        # Handle categorical columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "Unknown"
                df[col] = df[col].fillna(mode_value)
                self.cleaning_log.append({
                    "operation": "mode_imputation",
                    "column": col,
                    "fill_value": str(mode_value),
                })
        
        return df
    
    def _handle_outliers(
        self,
        df: pd.DataFrame,
        method: str = "cap",
    ) -> pd.DataFrame:
        """Handle outliers in numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.01)
            Q99 = df[col].quantile(0.99)
            
            n_outliers = ((df[col] < Q1) | (df[col] > Q99)).sum()
            
            if n_outliers > 0:
                if method == "cap":
                    # Cap at 1st and 99th percentile
                    df[col] = df[col].clip(lower=Q1, upper=Q99)
                    self.cleaning_log.append({
                        "operation": "outlier_capping",
                        "column": col,
                        "outliers_capped": int(n_outliers),
                        "lower_bound": float(Q1),
                        "upper_bound": float(Q99),
                    })
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate cleaned data."""
        return {
            "no_missing_values": df.isnull().sum().sum() == 0,
            "no_duplicates": df.duplicated().sum() == 0,
            "rows": len(df),
            "columns": len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        }
    
    def _generate_summary(
        self,
        original: pd.DataFrame,
        cleaned: pd.DataFrame,
    ) -> str:
        """Generate cleaning summary."""
        rows_removed = len(original) - len(cleaned)
        ops = len(self.cleaning_log)
        
        return (
            f"Cleaned data: {len(cleaned):,} rows remaining "
            f"({rows_removed} removed) | {ops} cleaning operations applied"
        )
