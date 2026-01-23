"""
DataPilot AI Pro - Profiler Agent
==================================
Analyzes data characteristics to understand dataset structure, quality, and target variable.

Responsibilities:
- Basic statistics (rows, columns, types)
- Data quality assessment (nulls, duplicates)
- Target variable detection via LLM
- Class balance analysis
- Correlation detection
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from loguru import logger

from .base_agent import BaseAgent


class ProfilerAgent(BaseAgent):
    """
    Profiler Agent - First step in the DataPilot pipeline.
    
    Examines the dataset comprehensively to understand its characteristics
    and provides insights for subsequent agents.
    """
    
    def __init__(self, model_name: str = "llama3.1"):
        super().__init__(
            name="Profiler",
            description="Analyzes data characteristics and detects target variable",
            model_name=model_name,
        )
    
    async def execute(
        self,
        data: pd.DataFrame,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Profile the dataset and return comprehensive analysis.
        
        Args:
            data: Input DataFrame
            context: Pipeline context
            
        Returns:
            Profile report with statistics and recommendations
        """
        self.update_state("profiling", 0.0, "Starting data profiling")
        
        profile = {}
        
        # Step 1: Basic Statistics
        self.update_state("profiling", 0.2, "Computing basic statistics")
        profile["basic_stats"] = self._compute_basic_stats(data)
        
        # Step 2: Data Quality Check
        self.update_state("profiling", 0.4, "Checking data quality")
        profile["quality"] = self._check_data_quality(data)
        
        # Step 3: Target Detection
        self.update_state("profiling", 0.6, "Detecting target variable")
        profile["target"] = await self._detect_target(data, context)
        
        # Step 4: Class Balance (if classification)
        self.update_state("profiling", 0.8, "Analyzing class balance")
        if profile["target"]["detected"]:
            profile["class_balance"] = self._analyze_class_balance(
                data, profile["target"]["column"]
            )
        
        # Step 5: Correlations
        self.update_state("profiling", 0.9, "Computing correlations")
        profile["correlations"] = self._compute_correlations(data)
        
        # Generate summary
        profile["summary"] = self._generate_summary(profile)
        
        self.update_state("complete", 1.0, "Profiling complete")
        self.log_result(profile)
        
        return profile
    
    def _compute_basic_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute basic dataset statistics."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = data.select_dtypes(include=["datetime64"]).columns.tolist()
        
        return {
            "n_rows": len(data),
            "n_columns": len(data.columns),
            "n_numeric": len(numeric_cols),
            "n_categorical": len(categorical_cols),
            "n_datetime": len(datetime_cols),
            "memory_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
            "columns": {
                col: str(dtype) for col, dtype in data.dtypes.items()
            },
        }
    
    def _check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality issues."""
        null_counts = data.isnull().sum()
        null_pct = (null_counts / len(data) * 100).round(2)
        
        return {
            "missing_values": {
                col: {"count": int(null_counts[col]), "percent": float(null_pct[col])}
                for col in null_counts.index
                if null_counts[col] > 0
            },
            "duplicates": {
                "count": int(data.duplicated().sum()),
                "percent": round(data.duplicated().sum() / len(data) * 100, 2),
            },
            "columns_with_high_nulls": [
                col for col in null_pct.index if null_pct[col] > 5
            ],
        }
    
    async def _detect_target(
        self,
        data: pd.DataFrame,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Detect target variable using LLM."""
        # Check if target is specified in context
        if "target_column" in context:
            target = context["target_column"]
            return {
                "detected": True,
                "column": target,
                "type": self._infer_problem_type(data[target]),
                "source": "user_specified",
            }
        
        # Use LLM to detect target
        columns_info = "\n".join([
            f"- {col}: {dtype}, {data[col].nunique()} unique values"
            for col, dtype in data.dtypes.items()
        ])
        
        prompt = f"""Analyze these columns and identify the most likely target variable for ML:

{columns_info}

Return ONLY the column name that is most likely the target variable.
Consider: 'target', 'label', 'y', 'class', 'outcome', binary columns, etc.
If uncertain, return the column that looks most like a prediction target.
"""
        
        try:
            response = await self.ask_llm(prompt)
            target_col = response.strip().strip("'\"")
            
            if target_col in data.columns:
                return {
                    "detected": True,
                    "column": target_col,
                    "type": self._infer_problem_type(data[target_col]),
                    "source": "llm_detected",
                }
        except Exception as e:
            logger.warning(f"LLM target detection failed: {e}")
        
        return {"detected": False, "column": None, "type": None, "source": None}
    
    def _infer_problem_type(self, target_series: pd.Series) -> str:
        """Infer if classification or regression."""
        n_unique = target_series.nunique()
        
        if n_unique == 2:
            return "binary_classification"
        elif n_unique <= 10:
            return "multiclass_classification"
        elif target_series.dtype in [np.float64, np.float32]:
            return "regression"
        else:
            return "classification"
    
    def _analyze_class_balance(
        self,
        data: pd.DataFrame,
        target_col: str,
    ) -> Dict[str, Any]:
        """Analyze class distribution."""
        value_counts = data[target_col].value_counts()
        
        return {
            "distribution": value_counts.to_dict(),
            "min_class_ratio": float(value_counts.min() / value_counts.max()),
            "is_imbalanced": value_counts.min() / value_counts.max() < 0.5,
        }
    
    def _compute_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute correlations between numeric features."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return {"correlations": [], "high_correlations": []}
        
        corr_matrix = numeric_data.corr()
        
        # Find high correlations
        high_corr = []
        for i, col1 in enumerate(corr_matrix.columns):
            for col2 in corr_matrix.columns[i + 1:]:
                corr_val = abs(corr_matrix.loc[col1, col2])
                if corr_val > 0.7:
                    high_corr.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "correlation": round(float(corr_val), 3),
                    })
        
        return {
            "n_numeric_features": len(numeric_data.columns),
            "high_correlations": sorted(
                high_corr, key=lambda x: x["correlation"], reverse=True
            ),
        }
    
    def _generate_summary(self, profile: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        stats = profile["basic_stats"]
        quality = profile["quality"]
        target = profile["target"]
        
        summary_parts = [
            f"Dataset: {stats['n_rows']:,} rows × {stats['n_columns']} columns",
            f"Features: {stats['n_numeric']} numeric, {stats['n_categorical']} categorical",
        ]
        
        if target["detected"]:
            summary_parts.append(f"Target: '{target['column']}' ({target['type']})")
        
        if quality["columns_with_high_nulls"]:
            summary_parts.append(
                f"Quality issues: {len(quality['columns_with_high_nulls'])} columns with >5% nulls"
            )
        
        return " | ".join(summary_parts)
