from dataclasses import dataclass

from graphgen.bases.datatypes import QAPair
from graphgen.models.evaluate.base_evaluator import BaseEvaluator
from graphgen.models.llm.tokenizer import Tokenizer
from graphgen.utils import create_event_loop


@dataclass
class LengthEvaluator(BaseEvaluator):
    tokenizer_name: str = "cl100k_base"

    def __post_init__(self):
        self.tokenizer = Tokenizer(model_name=self.tokenizer_name)

    async def evaluate_single(self, pair: QAPair) -> float:
        loop = create_event_loop()
        return await loop.run_in_executor(None, self._calculate_length, pair.answer)

    def _calculate_length(self, text: str) -> float:
        tokens = self.tokenizer.encode_string(text)
        return len(tokens)
