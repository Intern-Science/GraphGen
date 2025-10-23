from abc import ABC, abstractmethod
from typing import Any

from graphgen.bases.base_llm_client import BaseLLMClient


class BaseGenerator(ABC):
    """
    Generate QAs based on given prompts.
    """

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    @staticmethod
    @abstractmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        """Build prompt for LLM based on the given batch"""

    @staticmethod
    @abstractmethod
    def parse_response(response: str) -> Any:
        """Parse the LLM response and return the generated QAs"""

    async def generate(
        self,
        batch: tuple[
            list[tuple[str, dict]], list[tuple[Any, Any, dict] | tuple[Any, Any, Any]]
        ],
    ) -> dict[str, Any]:
        """
        Generate QAs based on a given batch.
        :param batch
        :return: QA pairs
        """
        result = {}
        prompt = self.build_prompt(batch)
        response = await self.llm_client.generate_answer(prompt)
        qa_pairs = self.parse_response(response)  # generate one or more QA pairs
        result.update(qa_pairs)
        return result

    @staticmethod
    def format_generation_results(
        results: list[dict], output_data_format: str
    ) -> list[dict[str, Any]]:
        if output_data_format == "Alpaca":
            results = [
                {
                    "instruction": v["question"],
                    "input": "",
                    "output": v["answer"],
                }
                for item in results
                for k, v in item.items()
            ]
        elif output_data_format == "Sharegpt":
            results = [
                {
                    "conversations": [
                        {"from": "human", "value": v["question"]},
                        {"from": "gpt", "value": v["answer"]},
                    ]
                }
                for item in results
                for k, v in item.items()
            ]
        elif output_data_format == "ChatML":
            results = [
                {
                    "messages": [
                        {"role": "user", "content": v["question"]},
                        {"role": "assistant", "content": v["answer"]},
                    ]
                }
                for item in results
                for k, v in item.items()
            ]
        else:
            raise ValueError(f"Unknown output data format: {output_data_format}")
        return results
