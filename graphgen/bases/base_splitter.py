import copy
import re
from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Literal, Optional, Union

from graphgen.bases.datatypes import Chunk
from graphgen.utils.log import logger


class BaseSplitter(ABC):
    """
    Abstract base class for splitting text into smaller chunks.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.keep_separator = keep_separator
        self.add_start_index = add_start_index
        self.strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Split the input text into smaller chunks.

        :param text: The input text to be chunk.
        :return: A list of text chunks.
        """

    def create_chunks(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Chunk]:
        """Create chunks from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        chunks = []
        for i, text in enumerate(texts):
            index = 0
            previous_chunk_len = 0
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self.add_start_index:
                    offset = index + previous_chunk_len - self.chunk_overlap
                    index = text.find(chunk, max(0, offset))
                    metadata["start_index"] = index
                    previous_chunk_len = len(chunk)
                new_chunk = Chunk(content=chunk, metadata=metadata)
                chunks.append(new_chunk)
        return chunks

    def _join_chunks(self, chunks: List[str], separator: str) -> Optional[str]:
        text = separator.join(chunks)
        if self.strip_whitespace:
            text = text.strip()
        if text == "":
            return None
        return text

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        # We now want to combine these smaller pieces into medium size chunks to send to the LLM.
        separator_len = self.length_function(separator)

        chunks = []
        current_chunk: List[str] = []
        total = 0
        for d in splits:
            _len = self.length_function(d)
            if (
                total + _len + (separator_len if len(current_chunk) > 0 else 0)
                > self.chunk_size
            ):
                if total > self.chunk_size:
                    logger.warning(
                        "Created a chunk of size %s, which is longer than the specified %s",
                        total,
                        self.chunk_size,
                    )
                if len(current_chunk) > 0:
                    chunk = self._join_chunks(current_chunk, separator)
                    if chunk is not None:
                        chunks.append(chunk)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self.chunk_overlap or (
                        total + _len + (separator_len if len(current_chunk) > 0 else 0)
                        > self.chunk_size
                        and total > 0
                    ):
                        total -= self.length_function(current_chunk[0]) + (
                            separator_len if len(current_chunk) > 1 else 0
                        )
                        current_chunk = current_chunk[1:]
            current_chunk.append(d)
            total += _len + (separator_len if len(current_chunk) > 1 else 0)
        chunk = self._join_chunks(current_chunk, separator)
        if chunk is not None:
            chunks.append(chunk)
        return chunks

    @staticmethod
    def _split_text_with_regex(
        text: str, separator: str, keep_separator: Union[bool, Literal["start", "end"]]
    ) -> List[str]:
        # Now that we have the separator, chunk the text
        if separator:
            if keep_separator:
                # The parentheses in the pattern keep the delimiters in the result.
                _splits = re.split(f"({separator})", text)
                splits = (
                    (
                        [
                            _splits[i] + _splits[i + 1]
                            for i in range(0, len(_splits) - 1, 2)
                        ]
                    )
                    if keep_separator == "end"
                    else (
                        [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
                    )
                )
                if len(_splits) % 2 == 0:
                    splits += _splits[-1:]
                splits = (
                    (splits + [_splits[-1]])
                    if keep_separator == "end"
                    else ([_splits[0]] + splits)
                )
            else:
                splits = re.split(separator, text)
        else:
            splits = list(text)
        return [s for s in splits if s != ""]
