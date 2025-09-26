from dataclasses import dataclass, field
from typing import Set

from graphgen.bases.datatypes import QAPair
from graphgen.models.evaluate.base_evaluator import BaseEvaluator
from graphgen.utils import NLTKHelper, create_event_loop, detect_main_language

nltk_helper = NLTKHelper()


@dataclass
class MTLDEvaluator(BaseEvaluator):
    """
    衡量文本词汇多样性的指标
    """

    stopwords_en: Set[str] = field(
        default_factory=lambda: set(nltk_helper.get_stopwords("english"))
    )
    stopwords_zh: Set[str] = field(
        default_factory=lambda: set(nltk_helper.get_stopwords("chinese"))
    )

    async def evaluate_single(self, pair: QAPair) -> float:
        loop = create_event_loop()
        return await loop.run_in_executor(None, self._calculate_mtld_score, pair.answer)

    def _calculate_mtld_score(self, text: str, threshold=0.72) -> float:
        """
        计算MTLD (向前和向后的平均值)

        min is 1.0
        higher is better
        """
        if not text or not text.strip():
            return 0.0

        lang = detect_main_language(text)
        tokens = nltk_helper.word_tokenize(text, lang)

        stopwords = self.stopwords_zh if lang == "zh" else self.stopwords_en
        filtered_tokens = [word for word in tokens if word not in stopwords]
        filtered_tokens = [word for word in filtered_tokens if word.isalnum()]

        if not filtered_tokens:
            return 0

        # 计算向前的MTLD
        forward_factors = self._compute_factors(filtered_tokens, threshold)

        # 计算向后的MTLD
        backward_factors = self._compute_factors(filtered_tokens[::-1], threshold)

        # 取平均值
        return (forward_factors + backward_factors) / 2

    @staticmethod
    def _compute_factors(tokens: list, threshold: float) -> float:
        factors = 0
        current_segment = []
        unique_words = set()

        for token in tokens:
            current_segment.append(token)
            unique_words.add(token)
            ttr = len(unique_words) / len(current_segment)

            if ttr <= threshold:
                factors += 1
                current_segment = []
                unique_words = set()

        # 处理最后一个不完整片段
        if current_segment:
            ttr = len(unique_words) / len(current_segment)
            if ttr <= threshold:
                factors += 1
            else:
                factors += 1 - (ttr - threshold) / (1 - threshold)

        return len(tokens) / factors if factors > 0 else len(tokens)
