from typing import Any

from graphgen.models.splitter.recursive_character_splitter import (
    RecursiveCharacterSplitter,
)


class MarkdownTextRefSplitter(RecursiveCharacterSplitter):
    """Attempts to chunk the text along Markdown-formatted headings."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a MarkdownTextRefSplitter."""
        separators = [
            # First, try to chunk along Markdown headings (starting with level 2)
            "\n#{1,6} ",
            # Note the alternative syntax for headings (below) is not handled here
            # Heading level 2
            # ---------------
            # End of code block
            "```\n",
            # Horizontal lines
            "\n\\*\\*\\*+\n",
            "\n---+\n",
            "\n___+\n",
            # Note: horizontal lines defined by three or more of ***, ---, or ___
            # are handled by the regexes above, but alternative syntaxes (e.g., with spaces)
            # are not handled.
            "\n\n",
            "\n",
            " ",
            "",
        ]
        super().__init__(separators=separators, **kwargs)
