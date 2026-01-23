"""
DataPilot AI Pro - Guided Mode Pipeline
========================================
Interactive execution mode with user approval at each step.

In Guided Mode:
- Plan is shown to user for approval/modification
- Each major stage can be reviewed before proceeding
- User has full control over the pipeline
- Best for production use cases requiring oversight
"""

from typing import Any, Dict, Optional, Callable
import pandas as pd
from loguru import logger

from .state_machine import DataPilotPipeline, PipelineState, PipelineStage


class GuidedModePipeline(DataPilotPipeline):
    """
    Guided mode - interactive pipeline with user approval.
    
    Pauses at key stages to allow user review and
    modification before proceeding.
    """
    
    def __init__(
        self,
        approval_callback: Optional[Callable[[str, Dict], bool]] = None,
    ):
        """
        Initialize guided mode pipeline.
        
        Args:
            approval_callback: Function called for user approval.
                              Takes (stage_name, stage_data) and returns bool.
        """
        super().__init__()
        self.approval_callback = approval_callback or self._default_approval
        self.pending_approval: Optional[Dict] = None
    
    def _default_approval(self, stage: str, data: Dict) -> bool:
        """Default approval - always approve (for testing)."""
        logger.info(f"Auto-approving stage: {stage}")
        return True
    
    async def run_with_approval(
        self,
        data: pd.DataFrame,
        user_prompt: Optional[str] = None,
    ) -> PipelineState:
        """
        Run pipeline with approval checkpoints.
        
        Args:
            data: Input DataFrame
            user_prompt: Optional instructions
            
        Returns:
            Final pipeline state
        """
        state = PipelineState(
            raw_data=data,
            mode="guided",
            user_prompt=user_prompt,
        )
        
        logger.info("Starting Guided Mode analysis")
        
        # Stage 1: Profile
        state.profile = await self.profiler.execute(state.raw_data, state.to_context())
        state.current_stage = PipelineStage.PROFILE
        
        # Stage 2: Generate and approve plan
        plan = await self._generate_plan(state)
        state.plan = plan
        
        if not self.approval_callback("plan", plan):
            logger.info("User rejected plan")
            return state
        
        # Stage 3: Clean
        clean_result = await self.cleaner.execute(state.raw_data, state.to_context())
        state.cleaned_data = clean_result["cleaned_data"]
        state.cleaning_report = clean_result
        
        if not self.approval_callback("cleaning", clean_result):
            logger.info("User rejected cleaning")
            return state
        
        # Stage 4: Feature Engineering
        feature_result = await self.feature_engineer.execute(
            state.cleaned_data, state.to_context()
        )
        state.feature_matrix = feature_result["feature_matrix"]
        state.feature_report = feature_result
        
        # Stage 5: Visualization
        viz_result = await self.visualizer.execute(
            state.feature_matrix, state.to_context()
        )
        state.visualizations = viz_result["charts"]
        
        # Stage 6: RL Model Selection
        target_col = state.profile.get("target", {}).get("column")
        if target_col:
            X = state.feature_matrix.drop(columns=[target_col], errors="ignore")
            y = state.feature_matrix[target_col]
            meta_features = self.meta_extractor.extract(X, y)
            selected = self.rl_selector.select_models(meta_features)
            state.rl_selection = {
                "meta_features": meta_features,
                "selected_models": selected,
            }
        
        if not self.approval_callback("model_selection", state.rl_selection):
            logger.info("User rejected model selection")
            return state
        
        # Stage 7: Model Training
        model_result = await self.modeler.execute(
            state.feature_matrix, state.to_context()
        )
        state.models = model_result["models"]
        state.metrics = model_result["metrics"]
        
        # Stage 8: Explanation
        explain_result = await self.explainer.execute(
            state.feature_matrix, state.to_context()
        )
        state.explanations = explain_result["explanations"]
        
        state.current_stage = PipelineStage.COMPLETE
        logger.info("Guided mode pipeline completed")
        
        return state
    
    async def _generate_plan(self, state: PipelineState) -> Dict[str, Any]:
        """Generate execution plan for user approval."""
        profile = state.profile
        
        plan = {
            "data_summary": {
                "rows": profile.get("basic_stats", {}).get("n_rows", 0),
                "columns": profile.get("basic_stats", {}).get("n_columns", 0),
                "target": profile.get("target", {}).get("column"),
                "problem_type": profile.get("target", {}).get("type"),
            },
            "issues_found": [],
            "cleaning_plan": [],
            "feature_plan": [],
            "model_plan": {
                "method": "RL-based selection",
                "ensemble": "Voting + Stacking",
            },
            "estimated_time": "5-10 minutes",
        }
        
        # Identify issues
        quality = profile.get("quality", {})
        if quality.get("columns_with_high_nulls"):
            plan["issues_found"].append(
                f"{len(quality['columns_with_high_nulls'])} columns with >5% missing values"
            )
            plan["cleaning_plan"].append("KNN/Mean imputation for missing values")
        
        if quality.get("duplicates", {}).get("count", 0) > 0:
            plan["issues_found"].append(
                f"{quality['duplicates']['count']} duplicate rows"
            )
            plan["cleaning_plan"].append("Remove duplicate rows")
        
        # Feature engineering plan
        n_categorical = profile.get("basic_stats", {}).get("n_categorical", 0)
        if n_categorical > 0:
            plan["feature_plan"].append(f"Encode {n_categorical} categorical features")
        
        plan["feature_plan"].append("Apply StandardScaler to numeric features")
        plan["feature_plan"].append("Select top features using mutual information")
        
        return plan
    
    async def modify_plan(
        self,
        modifications: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Modify the pending plan based on user feedback.
        
        Args:
            modifications: Dictionary of changes to apply
            
        Returns:
            Updated plan
        """
        if self.pending_approval is None:
            return {"error": "No pending plan to modify"}
        
        # Apply modifications
        for key, value in modifications.items():
            if key in self.pending_approval:
                self.pending_approval[key] = value
        
        logger.info(f"Plan modified: {list(modifications.keys())}")
        return self.pending_approval
