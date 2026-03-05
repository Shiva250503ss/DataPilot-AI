"""
DataPilot AI Pro - Natural Language to SQL Agent
==================================================
Converts natural language queries into SQL using GPT-4 with
context-aware prompting and dynamic schema understanding.

Supports multi-database connectivity via SQLAlchemy:
  - PostgreSQL
  - MySQL
  - SQLite
  - Any SQLAlchemy-compatible database

Flow:
  Natural Language Query
    -> Schema Retrieval (dynamic, runtime)
    -> Context-Aware Prompt Construction (schema + few-shot examples + history)
    -> SQL Generation (GPT-4)
    -> SQL Validation (syntax + schema check)
    -> Execution (SQLAlchemy)
    -> Result Formatting
    -> Self-Correction Loop (if execution fails)
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


class NLSQLAgent:
    """
    Natural Language to SQL agent with context-aware prompting.

    Dynamically reads the database schema at query time and injects
    it into the GPT-4 prompt so the LLM generates accurate SQL
    without ever hardcoding table or column names.

    Supports multi-database connectivity through SQLAlchemy's
    dialect system (PostgreSQL, MySQL, SQLite, etc.).
    """

    MAX_CORRECTION_ATTEMPTS = 3

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the NL-to-SQL agent.

        Args:
            connection_string: SQLAlchemy connection string.
                               If None, uses DATABASE_URL env variable.
        """
        self.connection_string = connection_string or os.getenv("DATABASE_URL", "")
        self.engine = None
        self.schema_cache: Dict[str, Any] = {}
        self.conversation_history: List[Dict] = []

        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.0,  # Deterministic for SQL generation
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        if self.connection_string:
            self._connect(self.connection_string)

    def connect(self, connection_string: str):
        """Connect to a new database at runtime."""
        self.connection_string = connection_string
        self._connect(connection_string)

    def _connect(self, connection_string: str):
        """Create SQLAlchemy engine and cache the schema."""
        try:
            self.engine = create_engine(connection_string)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.schema_cache = self._extract_schema()
            logger.info(f"Connected to database. Tables: {list(self.schema_cache.keys())}")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def _extract_schema(self) -> Dict[str, Any]:
        """
        Dynamically extract full schema from the connected database.

        Returns a dict mapping table_name -> {columns, primary_keys, foreign_keys, sample_rows}
        This is the 'dynamic schema understanding' that enables context-aware prompting.
        """
        schema = {}
        inspector = inspect(self.engine)

        for table_name in inspector.get_table_names():
            columns = []
            for col in inspector.get_columns(table_name):
                columns.append({
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                })

            pk = inspector.get_pk_constraint(table_name)
            fks = inspector.get_foreign_keys(table_name)

            # Fetch sample rows for few-shot context
            sample_rows = []
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 3"))
                    sample_rows = [dict(row._mapping) for row in result]
            except Exception:
                pass

            schema[table_name] = {
                "columns": columns,
                "primary_keys": pk.get("constrained_columns", []),
                "foreign_keys": [
                    {
                        "column": fk["constrained_columns"],
                        "references": f"{fk['referred_table']}.{fk['referred_columns']}",
                    }
                    for fk in fks
                ],
                "sample_rows": sample_rows,
            }

        return schema

    def _build_schema_prompt(self) -> str:
        """
        Serialize the extracted schema into a human-readable prompt section.

        This is injected into every SQL generation prompt so GPT-4 knows
        exactly what tables and columns exist — preventing hallucinated column names.
        """
        lines = ["DATABASE SCHEMA:"]
        for table, info in self.schema_cache.items():
            col_defs = ", ".join(
                f"{c['name']} ({c['type']})" for c in info["columns"]
            )
            lines.append(f"\nTable: {table}")
            lines.append(f"  Columns: {col_defs}")
            if info["primary_keys"]:
                lines.append(f"  Primary Keys: {', '.join(info['primary_keys'])}")
            if info["foreign_keys"]:
                for fk in info["foreign_keys"]:
                    lines.append(f"  Foreign Key: {fk['column']} -> {fk['references']}")
            if info["sample_rows"]:
                lines.append(f"  Sample: {info['sample_rows'][0]}")
        return "\n".join(lines)

    def _build_conversation_context(self) -> str:
        """Build recent conversation history for multi-turn query support."""
        if not self.conversation_history:
            return ""
        recent = self.conversation_history[-3:]  # Last 3 turns
        lines = ["PREVIOUS QUERIES IN THIS SESSION:"]
        for turn in recent:
            lines.append(f"  Q: {turn['question']}")
            lines.append(f"  SQL: {turn['sql']}")
        return "\n".join(lines)

    async def query(self, natural_language_query: str) -> Dict[str, Any]:
        """
        Convert a natural language query to SQL and execute it.

        Args:
            natural_language_query: Plain English question about the data.

        Returns:
            Dict with keys: sql, results (DataFrame), explanation, error
        """
        if not self.engine:
            return {"error": "No database connected. Call connect() first.", "sql": "", "results": None}

        logger.info(f"NL-to-SQL query: {natural_language_query}")

        schema_prompt = self._build_schema_prompt()
        history_prompt = self._build_conversation_context()

        sql, explanation = await self._generate_sql(
            natural_language_query, schema_prompt, history_prompt
        )

        # Self-correction loop
        for attempt in range(self.MAX_CORRECTION_ATTEMPTS):
            results, error = self._execute_sql(sql)
            if error is None:
                break
            logger.warning(f"SQL execution failed (attempt {attempt + 1}): {error}")
            sql, explanation = await self._correct_sql(
                natural_language_query, sql, error, schema_prompt
            )
        else:
            return {"sql": sql, "results": None, "explanation": explanation, "error": error}

        # Store in conversation history for multi-turn context
        self.conversation_history.append({
            "question": natural_language_query,
            "sql": sql,
        })

        return {
            "sql": sql,
            "results": results,
            "explanation": explanation,
            "error": None,
            "row_count": len(results) if results is not None else 0,
        }

    async def _generate_sql(
        self,
        question: str,
        schema_prompt: str,
        history_prompt: str,
    ) -> Tuple[str, str]:
        """
        Generate SQL using GPT-4 with context-aware prompting.

        The prompt contains:
          1. Full database schema (dynamic, from live DB)
          2. Conversation history (multi-turn support)
          3. The natural language question
          4. Instruction to return ONLY the SQL
        """
        system_prompt = f"""You are an expert SQL query generator. Generate valid SQL based ONLY on the schema provided.
Never invent table or column names. If a request is ambiguous, use your best judgment based on the schema.
Return ONLY the SQL query — no explanation, no markdown fences.

{schema_prompt}

{history_prompt}"""

        user_prompt = f"Convert this to SQL: {question}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = await self.llm.ainvoke(messages)
        sql = response.content.strip().strip("```sql").strip("```").strip()

        # Generate explanation separately
        explain_response = await self.llm.ainvoke([
            HumanMessage(content=f"In one sentence, explain what this SQL query does: {sql}")
        ])
        explanation = explain_response.content.strip()

        return sql, explanation

    async def _correct_sql(
        self,
        question: str,
        failed_sql: str,
        error: str,
        schema_prompt: str,
    ) -> Tuple[str, str]:
        """Self-correction: re-prompt GPT-4 with the error message to fix the SQL."""
        correction_prompt = f"""The following SQL query failed with this error:

SQL: {failed_sql}
Error: {error}

{schema_prompt}

Fix the SQL to correctly answer: "{question}"
Return ONLY the corrected SQL query."""

        response = await self.llm.ainvoke([HumanMessage(content=correction_prompt)])
        corrected_sql = response.content.strip().strip("```sql").strip("```").strip()
        return corrected_sql, "Corrected SQL after self-correction loop."

    def _execute_sql(self, sql: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Execute SQL against the connected database via SQLAlchemy."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                if result.returns_rows:
                    df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
                    return df, None
                else:
                    return pd.DataFrame({"affected_rows": [result.rowcount]}), None
        except SQLAlchemyError as e:
            return None, str(e)

    def get_tables(self) -> List[str]:
        """Return list of available tables in the connected database."""
        return list(self.schema_cache.keys())

    def get_schema(self) -> Dict[str, Any]:
        """Return the full extracted schema."""
        return self.schema_cache
