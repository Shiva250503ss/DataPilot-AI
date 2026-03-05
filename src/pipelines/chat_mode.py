"""
DataPilot AI Pro - Chat Mode Pipeline
======================================
Fully autonomous execution mode with RAG-based conversational AI.

In Chat Mode:
- Pipeline executes automatically from start to finish
- Plan is generated and executed without approval
- User can ask questions about results via RAG (Retrieve-Augment-Generate)
- Analysis context is stored in ChromaDB for semantic retrieval
- Best for quick analysis and exploration
"""

import os
import json
import uuid
from typing import Any, Dict, Optional
import pandas as pd
from loguru import logger

from .state_machine import DataPilotPipeline, PipelineState

try:
    import chromadb
    from chromadb.config import Settings
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.schema import Document
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


class ChatModePipeline(DataPilotPipeline):
    """
    Chat mode - autonomous pipeline execution with RAG conversational AI.

    Uses ChromaDB as a vector store to persist analysis context.
    When a user asks a question, relevant context is retrieved via
    semantic similarity search and injected into the GPT-4 prompt
    (Retrieve → Augment → Generate).
    """

    def __init__(self):
        super().__init__()
        self.conversation_history: list = []
        self.vector_store = None
        self.session_id = str(uuid.uuid4())[:8]
        self._init_vector_store()

    def _init_vector_store(self):
        """Initialize ChromaDB vector store for RAG retrieval."""
        if not HAS_CHROMADB:
            logger.warning("ChromaDB not available - RAG disabled, falling back to direct context injection")
            return
        try:
            embeddings = OpenAIEmbeddings(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="text-embedding-3-small",
            )
            chroma_client = chromadb.PersistentClient(
                path=os.getenv("CHROMADB_PATH", "./chroma_store"),
                settings=Settings(anonymized_telemetry=False),
            )
            self.vector_store = Chroma(
                client=chroma_client,
                collection_name=f"datapilot_{self.session_id}",
                embedding_function=embeddings,
            )
            logger.info("ChromaDB vector store initialized for RAG")
        except Exception as e:
            logger.warning(f"ChromaDB init failed: {e} - falling back to direct context injection")
            self.vector_store = None

    def _index_analysis_context(self, state: PipelineState):
        """
        Embed and store analysis results in ChromaDB.

        Each major result section (profile, metrics, feature importance,
        SHAP explanations) is stored as a separate document so the
        retriever can fetch only the most relevant chunk per query.
        """
        if self.vector_store is None:
            return

        docs = []

        # Profile document
        if state.profile:
            stats = state.profile.get("basic_stats", {})
            target = state.profile.get("target", {})
            profile_text = (
                f"Dataset profile: {stats.get('n_rows', 0):,} rows, "
                f"{stats.get('n_columns', 0)} columns. "
                f"Target variable: {target.get('column', 'unknown')} "
                f"(type: {target.get('type', 'unknown')}). "
                f"Missing values: {state.profile.get('quality', {}).get('missing_pct', 0):.1f}%."
            )
            docs.append(Document(page_content=profile_text, metadata={"section": "profile"}))

        # Model metrics document
        if state.metrics:
            best = max(state.metrics.items(), key=lambda x: x[1].get("f1_score", 0))
            metrics_text = (
                f"Model performance: Best model is {best[0]} with F1={best[1].get('f1_score', 0):.3f}, "
                f"Accuracy={best[1].get('accuracy', 0):.3f}, "
                f"ROC-AUC={best[1].get('roc_auc', 0):.3f}. "
                + " | ".join(
                    f"{k}: F1={v.get('f1_score', 0):.3f}"
                    for k, v in state.metrics.items()
                )
            )
            docs.append(Document(page_content=metrics_text, metadata={"section": "metrics"}))

        # Feature importance document
        if state.explanations:
            importance = state.explanations.get("feature_importance", {})
            top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            importance_text = (
                "Feature importance (SHAP): "
                + ", ".join(f"{feat} ({score:.3f})" for feat, score in top)
            )
            docs.append(Document(page_content=importance_text, metadata={"section": "feature_importance"}))

            nl_summary = state.explanations.get("nl_summary", "")
            if nl_summary:
                docs.append(Document(page_content=nl_summary, metadata={"section": "explanation"}))

        if docs:
            self.vector_store.add_documents(docs)
            logger.info(f"Indexed {len(docs)} analysis context documents in ChromaDB")

    def _retrieve_relevant_context(self, question: str, k: int = 3) -> str:
        """
        Retrieve the most semantically relevant context chunks for a question.
        This is the 'R' (Retrieve) step in RAG.
        """
        if self.vector_store is None:
            return ""
        try:
            results = self.vector_store.similarity_search(question, k=k)
            return "\n\n".join(doc.page_content for doc in results)
        except Exception as e:
            logger.warning(f"ChromaDB retrieval failed: {e}")
            return ""

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

        # Index analysis results in ChromaDB for RAG
        self._index_analysis_context(state)

        self.conversation_history.append({
            "type": "analysis",
            "prompt": prompt,
            "state": state,
        })

        return state

    async def ask(self, question: str) -> str:
        """
        Answer a natural language question about the analysis using RAG.

        Flow:
          1. Retrieve relevant context chunks from ChromaDB (semantic search)
          2. Augment the prompt with retrieved context
          3. Generate answer via GPT-4

        Args:
            question: Natural language question

        Returns:
            Natural language answer grounded in analysis results
        """
        if not self.conversation_history:
            return "No analysis has been run yet. Please upload data first."

        # Step 1: Retrieve - semantic search over ChromaDB
        retrieved_context = self._retrieve_relevant_context(question)

        # Fallback: build context directly from state if ChromaDB unavailable
        if not retrieved_context:
            last_state = next(
                (item["state"] for item in reversed(self.conversation_history) if item["type"] == "analysis"),
                None,
            )
            retrieved_context = self._build_context(last_state) if last_state else ""

        # Step 2: Augment - inject retrieved context into prompt
        augmented_prompt = f"""You are a data science assistant. Use ONLY the following analysis results to answer the question. Do not fabricate numbers.

--- ANALYSIS CONTEXT ---
{retrieved_context}
--- END CONTEXT ---

Question: {question}

Provide a clear, concise, data-grounded answer."""

        # Step 3: Generate - GPT-4 response
        try:
            answer = await self.profiler.ask_llm(augmented_prompt)

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
        """Build context string from pipeline state (fallback for when ChromaDB is unavailable)."""
        parts = []

        if state.profile:
            stats = state.profile.get("basic_stats", {})
            parts.append(f"Dataset: {stats.get('n_rows', 0):,} rows × {stats.get('n_columns', 0)} columns")
            target = state.profile.get("target", {})
            if target.get("detected"):
                parts.append(f"Target: {target['column']} ({target['type']})")

        if state.metrics:
            best = max(state.metrics.items(), key=lambda x: x[1].get("f1_score", 0))
            parts.append(f"Best model: {best[0]} (F1: {best[1].get('f1_score', 0):.3f})")

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

        last = next(
            (item for item in reversed(self.conversation_history) if item["type"] == "analysis"),
            None,
        )
        if last is None:
            return {"status": "No analysis run"}

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
            "rag_enabled": self.vector_store is not None,
        }
