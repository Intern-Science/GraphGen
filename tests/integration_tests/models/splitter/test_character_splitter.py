import pytest

from graphgen.models.splitter.character_splitter import CharacterSplitter


@pytest.mark.parametrize(
    "text,chunk_size,chunk_overlap,expected",
    [
        (
            "This is a test.\n\nThis is only a test.\n\nIn the event of an actual emergency...",
            25,
            5,
            [
                "This is a test.",
                "This is only a test.",
                "In the event of an actual emergency...",
            ],
        ),
    ],
)
def test_character_splitter(text, chunk_size, chunk_overlap, expected):
    splitter = CharacterSplitter(
        separator="\n\n",
        is_separator_regex=False,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_separator=False,
    )
    chunks = splitter.split_text(text)
    assert chunks == expected
