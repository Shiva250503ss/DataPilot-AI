"""
DataPilot AI Pro - Visualization Agent
=======================================
Automatically generates optimal visualizations based on data characteristics.

Visualization Capabilities:
- Histograms with KDE for numeric
- Bar/Pie charts for categorical
- Correlation heatmaps
- Box plots by category
- Time series plots
- SHAP summary plots
- Confusion matrix & ROC curves
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from loguru import logger
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .base_agent import BaseAgent


class VisualizationAgent(BaseAgent):
    """
    Visualization Agent - Creates intelligent, context-aware visualizations.
    
    Analyzes data types and distributions to automatically select
    the most appropriate chart types.
    """
    
    def __init__(self, model_name: str = "llama3.1"):
        super().__init__(
            name="Visualizer",
            description="Generates smart visualizations and EDA insights",
            model_name=model_name,
        )
        self.charts: List[Dict[str, Any]] = []
    
    async def execute(
        self,
        data: pd.DataFrame,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate visualizations for EDA.
        
        Args:
            data: DataFrame to visualize
            context: Pipeline context
            
        Returns:
            Collection of charts and insights
        """
        self.update_state("visualizing", 0.0, "Starting visualization generation")
        self.charts = []
        
        target_col = context.get("profile", {}).get("target", {}).get("column")
        
        # Step 1: Target distribution
        self.update_state("visualizing", 0.2, "Plotting target distribution")
        if target_col:
            self._plot_target_distribution(data, target_col)
        
        # Step 2: Numeric distributions
        self.update_state("visualizing", 0.4, "Plotting numeric distributions")
        self._plot_numeric_distributions(data, target_col)
        
        # Step 3: Categorical distributions
        self.update_state("visualizing", 0.6, "Plotting categorical distributions")
        self._plot_categorical_distributions(data, target_col)
        
        # Step 4: Correlation heatmap
        self.update_state("visualizing", 0.8, "Creating correlation heatmap")
        self._plot_correlation_heatmap(data)
        
        # Step 5: Missing values visualization
        self.update_state("visualizing", 0.9, "Visualizing missing values")
        self._plot_missing_values(data)
        
        # Generate insights
        insights = await self._generate_insights(data, context)
        
        result = {
            "charts": self.charts,
            "insights": insights,
            "summary": f"Generated {len(self.charts)} visualizations with {len(insights)} insights",
        }
        
        self.update_state("complete", 1.0, "Visualization complete")
        self.log_result(result)
        
        return result
    
    def _plot_target_distribution(
        self,
        data: pd.DataFrame,
        target_col: str,
    ) -> None:
        """Plot target variable distribution."""
        if target_col not in data.columns:
            return
        
        if data[target_col].dtype == "object" or data[target_col].nunique() <= 10:
            # Categorical target - bar chart
            fig = px.bar(
                data[target_col].value_counts().reset_index(),
                x="index",
                y=target_col,
                title=f"Target Distribution: {target_col}",
                color="index",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
        else:
            # Numeric target - histogram
            fig = px.histogram(
                data,
                x=target_col,
                title=f"Target Distribution: {target_col}",
                marginal="box",
                color_discrete_sequence=["#3498db"],
            )
        
        fig.update_layout(template="plotly_white", showlegend=False)
        
        self.charts.append({
            "name": "target_distribution",
            "title": f"Target: {target_col}",
            "figure": fig,
            "type": "distribution",
        })
    
    def _plot_numeric_distributions(
        self,
        data: pd.DataFrame,
        target_col: Optional[str],
    ) -> None:
        """Plot distributions of numeric features."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Limit to top 9 features
        for col in numeric_cols[:9]:
            fig = px.histogram(
                data,
                x=col,
                title=f"Distribution: {col}",
                marginal="box",
                color_discrete_sequence=["#2ecc71"],
            )
            fig.update_layout(template="plotly_white")
            
            self.charts.append({
                "name": f"dist_{col}",
                "title": col,
                "figure": fig,
                "type": "histogram",
            })
    
    def _plot_categorical_distributions(
        self,
        data: pd.DataFrame,
        target_col: Optional[str],
    ) -> None:
        """Plot distributions of categorical features."""
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
        
        if target_col and target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        for col in categorical_cols[:6]:
            value_counts = data[col].value_counts().head(10)
            
            if len(value_counts) <= 6:
                # Pie chart for few categories
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Distribution: {col}",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
            else:
                # Bar chart for many categories
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Distribution: {col}",
                    color=value_counts.values,
                    color_continuous_scale="Viridis",
                )
            
            fig.update_layout(template="plotly_white")
            
            self.charts.append({
                "name": f"cat_{col}",
                "title": col,
                "figure": fig,
                "type": "categorical",
            })
    
    def _plot_correlation_heatmap(self, data: pd.DataFrame) -> None:
        """Create correlation heatmap for numeric features."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return
        
        # Limit to top 15 features
        cols = numeric_data.columns[:15].tolist()
        corr_matrix = numeric_data[cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Heatmap",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            text_auto=".2f",
        )
        
        fig.update_layout(template="plotly_white")
        
        self.charts.append({
            "name": "correlation_heatmap",
            "title": "Correlations",
            "figure": fig,
            "type": "heatmap",
        })
    
    def _plot_missing_values(self, data: pd.DataFrame) -> None:
        """Visualize missing values pattern."""
        missing = data.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=True)
        
        if len(missing) == 0:
            return
        
        fig = px.bar(
            x=missing.values,
            y=missing.index,
            orientation="h",
            title="Missing Values by Column",
            color=missing.values,
            color_continuous_scale="Reds",
        )
        
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Number of Missing Values",
            yaxis_title="Column",
        )
        
        self.charts.append({
            "name": "missing_values",
            "title": "Missing Values",
            "figure": fig,
            "type": "missing",
        })
    
    async def _generate_insights(
        self,
        data: pd.DataFrame,
        context: Dict[str, Any],
    ) -> List[str]:
        """Generate natural language insights from data."""
        insights = []
        
        # Basic statistics insight
        n_rows, n_cols = data.shape
        insights.append(f"Dataset contains {n_rows:,} records with {n_cols} features")
        
        # Missing values insight
        missing_pct = data.isnull().sum().sum() / (n_rows * n_cols) * 100
        if missing_pct > 0:
            insights.append(f"Data completeness: {100 - missing_pct:.1f}% ({missing_pct:.1f}% missing)")
        else:
            insights.append("Dataset is complete with no missing values")
        
        # Correlation insights
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) >= 2:
            corr = numeric_data.corr()
            high_corr = (corr.abs() > 0.7) & (corr != 1.0)
            n_high_corr = high_corr.sum().sum() // 2
            if n_high_corr > 0:
                insights.append(f"Found {n_high_corr} pairs of highly correlated features (>0.7)")
        
        # Target insight
        target_info = context.get("profile", {}).get("target", {})
        if target_info.get("detected"):
            target_col = target_info["column"]
            problem_type = target_info["type"]
            insights.append(f"Detected {problem_type.replace('_', ' ')} problem with target '{target_col}'")
        
        return insights
