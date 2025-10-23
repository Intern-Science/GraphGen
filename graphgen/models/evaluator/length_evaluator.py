from graphgen.bases.datatypes import QAPair
from graphgen.models.evaluator.base_evaluator import BaseEvaluator
from graphgen.models.tokenizer import Tokenizer
from graphgen.utils import create_event_loop


class LengthEvaluator(BaseEvaluator):
    def __init__(self, tokenizer_name: str = "cl100k_base", max_concurrent: int = 100):
        super().__init__(max_concurrent)
        self.tokenizer_name = tokenizer_name
        self.tokenizer = Tokenizer(model_name=self.tokenizer_name)

    async def evaluate_single(self, pair: QAPair) -> float:
        loop = create_event_loop()
        return await loop.run_in_executor(None, self._calculate_length, pair.answer)

    def _calculate_length(self, text: str) -> float:
        tokens = self.tokenizer.encode(text)
        return len(tokens)
