"""
DataPilot AI Pro - Chat Mode Pipeline
======================================
Fully autonomous execution mode - no user intervention required.

In Chat Mode:
- Pipeline executes automatically from start to finish
- Plan is generated and executed without approval
- User can ask questions about results via natural language
- Best for quick analysis and exploration
"""

from typing import Any, Dict, Optional
import pandas as pd
from loguru import logger

from .state_machine import DataPilotPipeline, PipelineState


class ChatModePipeline(DataPilotPipeline):
    """
    Chat mode - autonomous pipeline execution.
    
    Runs the complete data science workflow without
    requiring user intervention at each step.
    """
    
    def __init__(self):
        super().__init__()
        self.conversation_history: list = []
    
    async def analyze(
        self,
        data: pd.DataFrame,
        prompt: Optional[str] = None,
    ) -> PipelineState:
        """
        Run autonomous analysis on data.
        
        Args:
            data: Input DataFrame
            prompt: Optional instructions (e.g., "focus on churn prediction")
            
        Returns:
            Complete pipeline state with all results
        """
        logger.info("Starting Chat Mode analysis")
        
        state = await self.run(
            data=data,
            mode="chat",
            user_prompt=prompt,
        )
        
        # Store in conversation
        self.conversation_history.append({
            "type": "analysis",
            "prompt": prompt,
            "state": state,
        })
        
        return state
    
    async def ask(self, question: str) -> str:
        """
        Ask a question about the most recent analysis.
        
        Args:
            question: Natural language question
            
        Returns:
            Natural language answer
        """
        if not self.conversation_history:
            return "No analysis has been run yet. Please upload data first."
        
        last_analysis = self.conversation_history[-1]
        state = last_analysis["state"]
        
        # Build context for LLM
        context = self._build_context(state)
        
        # Ask LLM
        prompt = f"""Based on this data analysis:

{context}

Answer this question: {question}

Provide a clear, concise answer based on the analysis results."""
        
        try:
            answer = await self.profiler.ask_llm(prompt)
            
            self.conversation_history.append({
                "type": "question",
                "question": question,
                "answer": answer,
            })
            
            return answer
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"Error: {e}"
    
    def _build_context(self, state: PipelineState) -> str:
        """Build context string from state for LLM."""
        parts = []
        
        # Profile summary
        if state.profile:
            stats = state.profile.get("basic_stats", {})
            parts.append(f"Dataset: {stats.get('n_rows', 0):,} rows × {stats.get('n_columns', 0)} columns")
            
            target = state.profile.get("target", {})
            if target.get("detected"):
                parts.append(f"Target: {target['column']} ({target['type']})")
        
        # Best model
        if state.metrics:
            best = max(state.metrics.items(), key=lambda x: x[1].get("f1_score", 0))
            parts.append(f"Best model: {best[0]} (F1: {best[1].get('f1_score', 0):.3f})")
        
        # Feature importance
        if state.explanations:
            importance = state.explanations.get("feature_importance", {})
            top_features = list(importance.keys())[:5]
            if top_features:
                parts.append(f"Top features: {', '.join(top_features)}")
        
        return "\n".join(parts)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of the last analysis."""
        if not self.conversation_history:
            return {"status": "No analysis run"}
        
        last = self.conversation_history[-1]
        if last["type"] != "analysis":
            # Find last analysis
            for item in reversed(self.conversation_history):
                if item["type"] == "analysis":
                    last = item
                    break
        
        state = last["state"]
        
        return {
            "rows": state.profile.get("basic_stats", {}).get("n_rows", 0),
            "columns": state.profile.get("basic_stats", {}).get("n_columns", 0),
            "target": state.profile.get("target", {}).get("column"),
            "best_model": max(
                state.metrics.items(),
                key=lambda x: x[1].get("f1_score", 0),
                default=("none", {})
            )[0] if state.metrics else None,
            "n_visualizations": len(state.visualizations),
            "execution_time": state.execution_time,
        }
