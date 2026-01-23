"""
DataPilot AI Pro - Pipeline State Machine
==========================================
LangGraph-based orchestration of the data science workflow.

Pipeline Stages:
START → PROFILE → PLAN → CLEAN → FEATURE → VISUALIZE → RL_SELECT → MODEL → EXPLAIN → COMPLETE

Each stage is handled by a specialized agent, with state passed
between stages via the PipelineState dataclass.
"""

from typing import Any, Dict, List, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from loguru import logger

try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False


class PipelineStage(str, Enum):
    """Enumeration of pipeline stages."""
    START = "start"
    PROFILE = "profile"
    PLAN = "plan"
    CLEAN = "clean"
    FEATURE = "feature"
    VISUALIZE = "visualize"
    RL_SELECT = "rl_select"
    MODEL = "model"
    EXPLAIN = "explain"
    COMPLETE = "complete"


@dataclass
class PipelineState:
    """
    State object passed through the LangGraph pipeline.
    
    Contains all data and results from each stage,
    enabling agents to access previous outputs.
    """
    # Input
    raw_data: Optional[pd.DataFrame] = None
    mode: str = "chat"  # "chat" or "guided"
    user_prompt: Optional[str] = None
    
    # Stage outputs
    profile: Dict[str, Any] = field(default_factory=dict)
    plan: Dict[str, Any] = field(default_factory=dict)
    cleaned_data: Optional[pd.DataFrame] = None
    cleaning_report: Dict[str, Any] = field(default_factory=dict)
    feature_matrix: Optional[pd.DataFrame] = None
    feature_report: Dict[str, Any] = field(default_factory=dict)
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    rl_selection: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    explanations: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    current_stage: PipelineStage = PipelineStage.START
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: Dict[str, float] = field(default_factory=dict)
    
    def to_context(self) -> Dict[str, Any]:
        """Convert state to context dict for agents."""
        return {
            "mode": self.mode,
            "user_prompt": self.user_prompt,
            "profile": self.profile,
            "plan": self.plan,
            "cleaning_report": self.cleaning_report,
            "feature_report": self.feature_report,
            "rl_selection": self.rl_selection,
            "modeling": {
                "models": self.models,
                "metrics": self.metrics,
                "best_model": max(
                    self.metrics.items(),
                    key=lambda x: x[1].get("f1_score", 0),
                    default=("none", {})
                )[0] if self.metrics else None,
            },
        }


class DataPilotPipeline:
    """
    Main pipeline orchestrator using LangGraph state machine.
    
    Coordinates the execution of all agents in sequence,
    passing state between stages and handling errors.
    """
    
    def __init__(self):
        """Initialize the pipeline with all agents."""
        from ..agents import (
            ProfilerAgent,
            CleanerAgent,
            FeatureAgent,
            VisualizationAgent,
            ModelerAgent,
            ExplainerAgent,
        )
        from ..rl_selector import PPOModelSelector, MetaFeatureExtractor
        
        # Initialize agents
        self.profiler = ProfilerAgent()
        self.cleaner = CleanerAgent()
        self.feature_engineer = FeatureAgent()
        self.visualizer = VisualizationAgent()
        self.modeler = ModelerAgent()
        self.explainer = ExplainerAgent()
        
        # Initialize RL selector
        self.rl_selector = PPOModelSelector()
        self.meta_extractor = MetaFeatureExtractor()
        
        # Build graph if LangGraph is available
        self.graph = self._build_graph() if HAS_LANGGRAPH else None
        
        logger.info("DataPilot pipeline initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        graph = StateGraph(dict)
        
        # Add nodes for each stage
        graph.add_node("profile", self._profile_node)
        graph.add_node("plan", self._plan_node)
        graph.add_node("clean", self._clean_node)
        graph.add_node("feature", self._feature_node)
        graph.add_node("visualize", self._visualize_node)
        graph.add_node("rl_select", self._rl_select_node)
        graph.add_node("model", self._model_node)
        graph.add_node("explain", self._explain_node)
        graph.add_node("complete", self._complete_node)
        
        # Define edges
        graph.set_entry_point("profile")
        graph.add_edge("profile", "plan")
        graph.add_edge("plan", "clean")
        graph.add_edge("clean", "feature")
        graph.add_edge("feature", "visualize")
        graph.add_edge("visualize", "rl_select")
        graph.add_edge("rl_select", "model")
        graph.add_edge("model", "explain")
        graph.add_edge("explain", "complete")
        graph.add_edge("complete", END)
        
        return graph.compile()
    
    async def run(
        self,
        data: pd.DataFrame,
        mode: str = "chat",
        user_prompt: Optional[str] = None,
    ) -> PipelineState:
        """
        Execute the full pipeline.
        
        Args:
            data: Input DataFrame
            mode: "chat" (autonomous) or "guided" (interactive)
            user_prompt: Optional natural language instructions
            
        Returns:
            Final pipeline state with all results
        """
        state = PipelineState(
            raw_data=data,
            mode=mode,
            user_prompt=user_prompt,
        )
        
        logger.info(f"Starting pipeline in {mode} mode")
        
        if self.graph is not None:
            # Use LangGraph for execution
            result = await self.graph.ainvoke({"state": state})
            return result["state"]
        else:
            # Fallback to manual execution
            return await self._run_manual(state)
    
    async def _run_manual(self, state: PipelineState) -> PipelineState:
        """Manual pipeline execution when LangGraph is unavailable."""
        import time
        
        # Profile
        start = time.time()
        state.profile = await self.profiler.execute(state.raw_data, state.to_context())
        state.execution_time["profile"] = time.time() - start
        
        # Clean
        start = time.time()
        clean_result = await self.cleaner.execute(state.raw_data, state.to_context())
        state.cleaned_data = clean_result["cleaned_data"]
        state.cleaning_report = clean_result
        state.execution_time["clean"] = time.time() - start
        
        # Feature Engineering
        start = time.time()
        feature_result = await self.feature_engineer.execute(
            state.cleaned_data, state.to_context()
        )
        state.feature_matrix = feature_result["feature_matrix"]
        state.feature_report = feature_result
        state.execution_time["feature"] = time.time() - start
        
        # Visualization
        start = time.time()
        viz_result = await self.visualizer.execute(
            state.feature_matrix, state.to_context()
        )
        state.visualizations = viz_result["charts"]
        state.execution_time["visualize"] = time.time() - start
        
        # RL Model Selection
        start = time.time()
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
        state.execution_time["rl_select"] = time.time() - start
        
        # Model Training
        start = time.time()
        model_result = await self.modeler.execute(
            state.feature_matrix, state.to_context()
        )
        state.models = model_result["models"]
        state.metrics = model_result["metrics"]
        state.execution_time["model"] = time.time() - start
        
        # Explanation
        start = time.time()
        explain_result = await self.explainer.execute(
            state.feature_matrix, state.to_context()
        )
        state.explanations = explain_result["explanations"]
        state.execution_time["explain"] = time.time() - start
        
        state.current_stage = PipelineStage.COMPLETE
        logger.info("Pipeline completed successfully")
        
        return state
    
    # Node functions for LangGraph
    async def _profile_node(self, state_dict: Dict) -> Dict:
        """Profile stage node."""
        state = state_dict["state"]
        state.profile = await self.profiler.execute(state.raw_data, state.to_context())
        state.current_stage = PipelineStage.PROFILE
        return {"state": state}
    
    async def _plan_node(self, state_dict: Dict) -> Dict:
        """Plan stage node."""
        state = state_dict["state"]
        # Plan generation would happen here
        state.current_stage = PipelineStage.PLAN
        return {"state": state}
    
    async def _clean_node(self, state_dict: Dict) -> Dict:
        """Clean stage node."""
        state = state_dict["state"]
        result = await self.cleaner.execute(state.raw_data, state.to_context())
        state.cleaned_data = result["cleaned_data"]
        state.cleaning_report = result
        state.current_stage = PipelineStage.CLEAN
        return {"state": state}
    
    async def _feature_node(self, state_dict: Dict) -> Dict:
        """Feature engineering stage node."""
        state = state_dict["state"]
        result = await self.feature_engineer.execute(state.cleaned_data, state.to_context())
        state.feature_matrix = result["feature_matrix"]
        state.feature_report = result
        state.current_stage = PipelineStage.FEATURE
        return {"state": state}
    
    async def _visualize_node(self, state_dict: Dict) -> Dict:
        """Visualization stage node."""
        state = state_dict["state"]
        result = await self.visualizer.execute(state.feature_matrix, state.to_context())
        state.visualizations = result["charts"]
        state.current_stage = PipelineStage.VISUALIZE
        return {"state": state}
    
    async def _rl_select_node(self, state_dict: Dict) -> Dict:
        """RL model selection stage node."""
        state = state_dict["state"]
        target_col = state.profile.get("target", {}).get("column")
        if target_col:
            X = state.feature_matrix.drop(columns=[target_col], errors="ignore")
            y = state.feature_matrix[target_col]
            meta_features = self.meta_extractor.extract(X, y)
            selected = self.rl_selector.select_models(meta_features)
            state.rl_selection = {"meta_features": meta_features, "selected_models": selected}
        state.current_stage = PipelineStage.RL_SELECT
        return {"state": state}
    
    async def _model_node(self, state_dict: Dict) -> Dict:
        """Model training stage node."""
        state = state_dict["state"]
        result = await self.modeler.execute(state.feature_matrix, state.to_context())
        state.models = result["models"]
        state.metrics = result["metrics"]
        state.current_stage = PipelineStage.MODEL
        return {"state": state}
    
    async def _explain_node(self, state_dict: Dict) -> Dict:
        """Explanation stage node."""
        state = state_dict["state"]
        result = await self.explainer.execute(state.feature_matrix, state.to_context())
        state.explanations = result["explanations"]
        state.current_stage = PipelineStage.EXPLAIN
        return {"state": state}
    
    async def _complete_node(self, state_dict: Dict) -> Dict:
        """Completion stage node."""
        state = state_dict["state"]
        state.current_stage = PipelineStage.COMPLETE
        logger.info("Pipeline completed")
        return {"state": state}
