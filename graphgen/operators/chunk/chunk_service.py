import os
from functools import lru_cache
from typing import Union

import pandas as pd

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
    frozen_kwargs = frozenset(
        (k, tuple(v) if isinstance(v, list) else v) for k, v in kwargs.items()
    )
    splitter = _get_splitter(language, frozen_kwargs)
    return splitter.split_text(text)


class ChunkService:
    def __init__(self, **chunk_kwargs):
        tokenizer_model = os.getenv("TOKENIZER_MODEL", "cl100k_base")
        self.tokenizer_instance: Tokenizer = Tokenizer(model_name=tokenizer_model)
        self.chunk_kwargs = chunk_kwargs

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        docs = batch.to_dict(orient="records")
        return pd.DataFrame(self.chunk_documents(docs))

    def chunk_documents(self, new_docs: list) -> list:
        chunks = []
        for doc in new_docs:
            doc_id = doc.get("_doc_id")
            doc_type = doc.get("type")

            if doc_type == "text":
                doc_language = detect_main_language(doc["content"])
                text_chunks = split_chunks(
                    doc["content"],
                    language=doc_language,
                    **self.chunk_kwargs,
                )

                chunks.extend(
                    [
                        {
                            "_chunk_id": compute_content_hash(
                                chunk_text, prefix="chunk-"
                            ),
                            "content": chunk_text,
                            "type": "text",
                            "_doc_id": doc_id,
                            "length": len(self.tokenizer_instance.encode(chunk_text))
                            if self.tokenizer_instance
                            else len(chunk_text),
                            "language": doc_language,
                        }
                        for chunk_text in text_chunks
                    ]
                )
            else:
                # other types of documents(images, sequences) are not chunked
                chunks.append(
                    {
                        "_chunk_id": doc_id.replace("doc-", f"{doc_type}-"),
                        **doc,
                    }
                )
        return chunks
