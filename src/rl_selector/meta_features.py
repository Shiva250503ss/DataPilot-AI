"""
DataPilot AI Pro - Meta-Feature Extraction
============================================
Extracts 30+ meta-features from datasets for RL model selection.

Meta-feature Categories:
- Basic: n_samples, n_features, n_classes
- Statistical: skewness, kurtosis, outlier_ratio
- Complexity: class_imbalance, correlation_mean
- Landmarking: simple model scores for quick characterization
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


class MetaFeatureExtractor:
    """
    Extracts meta-features from datasets for RL model selection.
    
    These features characterize the dataset and help the PPO agent
    select the most appropriate ML models.
    """
    
    def __init__(self):
        """Initialize the meta-feature extractor."""
        self.feature_names: List[str] = []
    
    def extract(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """
        Extract all meta-features from dataset.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary of meta-features
        """
        meta_features = {}
        
        # Basic features
        meta_features.update(self._extract_basic(X, y))
        
        # Statistical features
        meta_features.update(self._extract_statistical(X))
        
        # Complexity features
        meta_features.update(self._extract_complexity(X, y))
        
        # Landmarking features
        meta_features.update(self._extract_landmarking(X, y))
        
        self.feature_names = list(meta_features.keys())
        
        logger.info(f"Extracted {len(meta_features)} meta-features")
        return meta_features
    
    def to_vector(self, meta_features: Dict[str, float]) -> np.ndarray:
        """Convert meta-features dict to vector for RL agent."""
        return np.array([
            meta_features.get(name, 0.0) for name in self.feature_names
        ], dtype=np.float32)
    
    def _extract_basic(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """Extract basic dataset characteristics."""
        n_samples, n_features = X.shape
        
        # Count feature types
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        
        # Handle target
        if y.dtype == "object":
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            n_classes = len(le.classes_)
        else:
            n_classes = len(np.unique(y))
        
        return {
            "n_samples": float(n_samples),
            "n_features": float(n_features),
            "n_classes": float(n_classes),
            "n_numeric": float(len(numeric_cols)),
            "n_categorical": float(len(categorical_cols)),
            "ratio_numeric": float(len(numeric_cols) / max(n_features, 1)),
            "log_n_samples": float(np.log1p(n_samples)),
            "log_n_features": float(np.log1p(n_features)),
            "samples_to_features_ratio": float(n_samples / max(n_features, 1)),
        }
    
    def _extract_statistical(self, X: pd.DataFrame) -> Dict[str, float]:
        """Extract statistical meta-features."""
        numeric_data = X.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {
                "mean_skewness": 0.0,
                "mean_kurtosis": 0.0,
                "mean_std": 0.0,
                "outlier_ratio": 0.0,
                "missing_ratio": 0.0,
            }
        
        # Skewness and kurtosis
        skewness_values = numeric_data.skew().fillna(0)
        kurtosis_values = numeric_data.kurtosis().fillna(0)
        
        # Outlier detection using IQR
        Q1 = numeric_data.quantile(0.25)
        Q3 = numeric_data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum().sum()
        total_values = numeric_data.shape[0] * numeric_data.shape[1]
        
        return {
            "mean_skewness": float(skewness_values.mean()),
            "mean_kurtosis": float(kurtosis_values.mean()),
            "max_skewness": float(skewness_values.abs().max()),
            "mean_std": float(numeric_data.std().mean()),
            "outlier_ratio": float(outliers / max(total_values, 1)),
            "missing_ratio": float(X.isnull().sum().sum() / (X.shape[0] * X.shape[1])),
            "zero_ratio": float((numeric_data == 0).sum().sum() / max(total_values, 1)),
        }
    
    def _extract_complexity(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """Extract complexity meta-features."""
        # Class imbalance
        if y.dtype == "object":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y))
        
        class_counts = y.value_counts()
        class_imbalance = float(class_counts.min() / max(class_counts.max(), 1))
        
        # Correlation mean
        numeric_data = X.select_dtypes(include=[np.number])
        if len(numeric_data.columns) >= 2:
            corr_matrix = numeric_data.corr().abs()
            # Get upper triangle without diagonal
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            correlation_mean = float(upper_tri.mean().mean())
            correlation_max = float(upper_tri.max().max())
        else:
            correlation_mean = 0.0
            correlation_max = 0.0
        
        # PCA variance ratio (approximate complexity)
        try:
            from sklearn.decomposition import PCA
            
            X_numeric = numeric_data.fillna(0)
            if X_numeric.shape[1] >= 2 and X_numeric.shape[0] >= 10:
                pca = PCA(n_components=min(5, X_numeric.shape[1]))
                pca.fit(X_numeric)
                pca_variance_ratio = float(sum(pca.explained_variance_ratio_))
            else:
                pca_variance_ratio = 1.0
        except Exception:
            pca_variance_ratio = 1.0
        
        return {
            "class_imbalance": class_imbalance,
            "correlation_mean": correlation_mean,
            "correlation_max": correlation_max,
            "pca_variance_ratio": pca_variance_ratio,
            "entropy": float(-sum(
                (class_counts / len(y)) * np.log2(class_counts / len(y) + 1e-10)
            )),
        }
    
    def _extract_landmarking(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """Extract landmarking features using simple models."""
        # Prepare data
        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
        
        if X_numeric.empty or len(X_numeric) < 10:
            return {
                "decision_tree_score": 0.5,
                "naive_bayes_score": 0.5,
                "linear_score": 0.5,
            }
        
        if y.dtype == "object":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y))
        
        # Sample if too large
        if len(X_numeric) > 1000:
            idx = np.random.choice(len(X_numeric), 1000, replace=False)
            X_sample = X_numeric.iloc[idx]
            y_sample = y.iloc[idx]
        else:
            X_sample = X_numeric
            y_sample = y
        
        scores = {}
        
        # Decision Tree
        try:
            dt = DecisionTreeClassifier(max_depth=5, random_state=42)
            dt_scores = cross_val_score(dt, X_sample, y_sample, cv=3, scoring="accuracy")
            scores["decision_tree_score"] = float(dt_scores.mean())
        except Exception:
            scores["decision_tree_score"] = 0.5
        
        # Naive Bayes
        try:
            nb = GaussianNB()
            nb_scores = cross_val_score(nb, X_sample, y_sample, cv=3, scoring="accuracy")
            scores["naive_bayes_score"] = float(nb_scores.mean())
        except Exception:
            scores["naive_bayes_score"] = 0.5
        
        # Logistic Regression
        try:
            lr = LogisticRegression(max_iter=100, random_state=42)
            lr_scores = cross_val_score(lr, X_sample, y_sample, cv=3, scoring="accuracy")
            scores["linear_score"] = float(lr_scores.mean())
        except Exception:
            scores["linear_score"] = 0.5
        
        return scores
