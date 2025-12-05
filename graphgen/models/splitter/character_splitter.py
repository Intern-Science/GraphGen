import re
from typing import Any, List

from graphgen.bases.base_splitter import BaseSplitter


class CharacterSplitter(BaseSplitter):
    """Splitting text that looks at characters."""

    def __init__(
        self, separator: str = "\n\n", is_separator_regex: bool = False, **kwargs: Any
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._separator = separator
        self._is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively chunk the large input into a bunch of smaller ones.
        separator = (
            self._separator if self._is_separator_regex else re.escape(self._separator)
        )
        splits = self._split_text_with_regex(text, separator, self.keep_separator)
        _separator = "" if self.keep_separator else self._separator
        return self._merge_splits(splits, _separator)
