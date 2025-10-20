from dataclasses import dataclass
from typing import Any

from graphgen.bases import BaseGenerator


@dataclass
class VQAGenerator(BaseGenerator):
    @staticmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        raise NotImplementedError(
            "VQAGenerator.build_prompt is not implemented. "
            "Please provide an implementation for VQA prompt construction."
        )

    @staticmethod
    def parse_response(response: str) -> Any:
        raise NotImplementedError(
            "VQAGenerator.parse_response is not implemented. "
            "Please provide an implementation for VQA response parsing."
        )
