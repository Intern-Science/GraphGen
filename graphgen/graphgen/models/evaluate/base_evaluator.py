import asyncio

from dataclasses import dataclass
from tqdm.asyncio import tqdm as tqdm_async
from graphgen.utils import create_event_loop
from graphgen.models.text.text_pair import TextPair

@dataclass
class BaseEvaluator:
    max_concurrent: int = 100
    results: list[float] = None

    def evaluate(self, pairs: list[TextPair]) -> list[float]:
        """
        Evaluate the text and return a score.
        """
        return create_event_loop().run_until_complete(self.async_evaluate(pairs))

    async def async_evaluate(self, pairs: list[TextPair]) -> list[float]:
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def evaluate_with_semaphore(pair):
            async with semaphore:  # 获取Semaphore
                return await self.evaluate_single(pair)

        results = []
        for result in tqdm_async(
            asyncio.as_completed([evaluate_with_semaphore(pair) for pair in pairs]),
            total=len(pairs),
        ):
            results.append(await result)
        return results

    async def evaluate_single(self, pair: TextPair) -> float:
        raise NotImplementedError()

    def get_average_score(self, pairs: list[TextPair]) -> float:
        """
        Get the average score of a batch of texts.
        """
        results = self.evaluate(pairs)
        self.results = results
        return sum(self.results) / len(pairs)

    def get_min_max_score(self, pairs: list[TextPair]) -> tuple[float, float]:
        """
        Get the min and max score of a batch of texts.
        """
        if self.results is None:
            self.get_average_score(pairs)
        return min(self.results), max(self.results)
