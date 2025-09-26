from graphgen.models.splitter.markdown_splitter import MarkdownTextRefSplitter


def test_split_markdown_structures():
    md = (
        "# Header1\n\n"
        "Some introduction here.\n\n"
        "## Header2\n\n"
        "```python\nprint('hello')\n```\n"
        "Paragraph under code block.\n\n"
        "***\n"
        "### Header3\n\n"
        "More text after horizontal rule.\n\n"
        "#### Header4\n\n"
        "Final paragraph."
    )

    splitter = MarkdownTextRefSplitter(
        chunk_size=120,
        chunk_overlap=0,
        keep_separator=True,
        is_separator_regex=True,
    )
    chunks = splitter.split_text(md)
    assert len(chunks) > 1

    for chk in chunks:
        assert len(chk) <= 120

    assert any("## Header2" in c for c in chunks)
    assert any("***" in c for c in chunks)
    assert any("```" in c for c in chunks)


def test_split_size_less_than_single_char():
    """Edge case: chunk_size is smaller than any segment; should still return the original text."""
    short = "# A\n\nB"
    splitter = MarkdownTextRefSplitter(chunk_size=1, chunk_overlap=0)
    chunks = splitter.split_text(short)
    assert "".join(chunks) == short
