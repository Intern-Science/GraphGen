from abc import ABC, abstractmethod
from typing import Any

from graphgen.bases.base_llm_wrapper import BaseLLMWrapper


class BaseExtractor(ABC):
    """
    Extract information from given text.

    """

    def __init__(self, llm_client: BaseLLMWrapper):
        self.llm_client = llm_client

    @abstractmethod
    async def extract(self, chunk: dict) -> Any:
        """Extract information from the given text"""

    @abstractmethod
    def build_prompt(self, text: str) -> str:
        """Build prompt for LLM based on the given text"""
