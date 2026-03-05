"""
DataPilot AI Pro - Base Agent
==============================
Abstract base class for all specialized agents in the multi-agent system.

This module provides the foundational structure that all agents inherit from,
including LLM integration, state management, and logging capabilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
import pandas as pd
from loguru import logger

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


@dataclass
class AgentState:
    """Represents the current state of an agent's execution."""
    
    status: str = "idle"
    progress: float = 0.0
    current_task: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(ABC):
    """
    Abstract base class for all DataPilot AI agents.
    
    Each specialized agent (Profiler, Cleaner, Feature, etc.) inherits
    from this class and implements the execute() method.
    
    Attributes:
        name: Human-readable agent name
        description: What this agent does
        llm: LangChain LLM instance for natural language processing
        state: Current execution state
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        model_name: str = "gpt-4",
        temperature: float = 0.1,
    ):
        """
        Initialize the base agent.

        Args:
            name: Agent identifier name
            description: Description of agent's purpose
            model_name: OpenAI model to use (default: gpt-4)
            temperature: LLM temperature for response variability
        """
        self.name = name
        self.description = description
        self.state = AgentState()

        # Initialize LLM - OpenAI GPT-4
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        
        # JSON output parser for structured responses
        self.json_parser = JsonOutputParser()
        
        logger.info(f"Initialized {self.name} agent")
    
    @abstractmethod
    async def execute(
        self,
        data: pd.DataFrame,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute the agent's primary task.
        
        Args:
            data: Input DataFrame to process
            context: Pipeline context with previous results
            
        Returns:
            Dictionary with execution results
        """
        pass
    
    def update_state(
        self,
        status: str,
        progress: float,
        task: Optional[str] = None,
    ) -> None:
        """Update the agent's execution state."""
        self.state.status = status
        self.state.progress = progress
        if task:
            self.state.current_task = task
        logger.debug(f"{self.name}: {status} ({progress:.0%})")
    
    def create_prompt(
        self,
        template: str,
        **kwargs,
    ) -> PromptTemplate:
        """Create a LangChain prompt template."""
        return PromptTemplate.from_template(template)
    
    async def ask_llm(
        self,
        prompt: str,
        parse_json: bool = False,
    ) -> Any:
        """
        Query the LLM with a prompt.
        
        Args:
            prompt: The prompt to send
            parse_json: Whether to parse response as JSON
            
        Returns:
            LLM response (string or parsed JSON)
        """
        try:
            response = await self.llm.ainvoke(prompt)
            # ChatOpenAI returns an AIMessage; extract text content
            text = response.content if hasattr(response, "content") else str(response)

            if parse_json:
                return self.json_parser.parse(text)

            return text
            
        except Exception as e:
            logger.error(f"{self.name} LLM error: {e}")
            raise
    
    def log_result(self, result: Dict[str, Any]) -> None:
        """Log the agent's execution result."""
        logger.info(f"{self.name} completed: {result.get('summary', 'No summary')}")
        self.state.result = result
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', status='{self.state.status}')>"
