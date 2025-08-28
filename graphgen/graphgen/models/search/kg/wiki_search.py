from dataclasses import dataclass
from typing import List, Union

import wikipedia
from wikipedia import set_lang

from graphgen.utils import detect_main_language, logger


@dataclass
class WikiSearch:
    @staticmethod
    def set_language(language: str):
        assert language in ["en", "zh"], "Only support English and Chinese"
        set_lang(language)

    async def search(self, query: str, num_results: int = 1) -> Union[List[str], None]:
        self.set_language(detect_main_language(query))
        return wikipedia.search(query, results=num_results, suggestion=False)

    async def summary(self, query: str) -> Union[str, None]:
        self.set_language(detect_main_language(query))
        try:
            result = wikipedia.summary(query, auto_suggest=False, redirect=False)
        except wikipedia.exceptions.DisambiguationError as e:
            logger.error("DisambiguationError: %s", e)
            result = None
        return result

    async def page(self, query: str) -> Union[str, None]:
        self.set_language(detect_main_language(query))
        try:
            result = wikipedia.page(query, auto_suggest=False, redirect=False).content
        except wikipedia.exceptions.DisambiguationError as e:
            logger.error("DisambiguationError: %s", e)
            result = None
        return result
