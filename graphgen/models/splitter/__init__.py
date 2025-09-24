from functools import lru_cache
from typing import Union

from .recursive_character_splitter import (
    ChineseRecursiveTextSplitter,
    RecursiveCharacterSplitter,
)

_MAPPING = {
    "en": RecursiveCharacterSplitter,
    "zh": ChineseRecursiveTextSplitter,
}

SplitterT = Union[RecursiveCharacterSplitter, ChineseRecursiveTextSplitter]


@lru_cache(maxsize=None)
def _get_splitter(language: str, frozen_kwargs: frozenset) -> SplitterT:
    cls = _MAPPING[language]
    kwargs = dict(frozen_kwargs)
    return cls(**kwargs)


def split_chunks(text: str, language: str = "en", **kwargs) -> list:
    if language not in _MAPPING:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported languages are: {list(_MAPPING.keys())}"
        )
    splitter = _get_splitter(language, frozenset(kwargs.items()))
    return splitter.split_text(text)
