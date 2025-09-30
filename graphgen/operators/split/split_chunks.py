from functools import lru_cache
from typing import Union

from tqdm.asyncio import tqdm as tqdm_async

from graphgen.models import (
    ChineseRecursiveTextSplitter,
    RecursiveCharacterSplitter,
    Tokenizer,
)
from graphgen.utils import compute_content_hash, detect_main_language

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


async def chunk_documents(
    new_docs: dict,
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
    tokenizer_instance: Tokenizer = None,
    progress_bar=None,
) -> dict:
    inserting_chunks = {}
    cur_index = 1
    doc_number = len(new_docs)
    async for doc_key, doc in tqdm_async(
        new_docs.items(), desc="[1/4]Chunking documents", unit="doc"
    ):
        doc_language = detect_main_language(doc["content"])
        text_chunks = split_chunks(
            doc["content"],
            language=doc_language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks = {
            compute_content_hash(txt, prefix="chunk-"): {
                "content": txt,
                "full_doc_id": doc_key,
                "length": len(tokenizer_instance.encode(txt))
                if tokenizer_instance
                else len(txt),
                "language": doc_language,
            }
            for txt in text_chunks
        }
        inserting_chunks.update(chunks)

        if progress_bar is not None:
            progress_bar(cur_index / doc_number, f"Chunking {doc_key}")
            cur_index += 1

    return inserting_chunks
