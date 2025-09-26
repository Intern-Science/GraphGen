from graphgen.models.splitter.recursive_character_splitter import (
    ChineseRecursiveTextSplitter,
    RecursiveCharacterSplitter,
)


def test_split_english_paragraph():
    text = (
        "Natural language processing (NLP) is a subfield of linguistics, computer science, "
        "and artificial intelligence. It focuses on the interaction between computers and "
        "humans through natural language. The ultimate objective of NLP is to read, decipher, "
        "understand, and make sense of human languages in a manner that is valuable.\n\n"
        "Most NLP techniques rely on machine learning."
    )

    splitter = RecursiveCharacterSplitter(
        chunk_size=150,
        chunk_overlap=0,
        keep_separator=True,
        is_separator_regex=False,
    )
    chunks = splitter.split_text(text)

    assert len(chunks) > 1
    for chk in chunks:
        assert len(chk) <= 150


def test_split_chinese_with_punctuation():
    text = (
        "自然语言处理是人工智能的重要分支。它研究能实现人与计算机之间用自然语言"
        "进行有效通信的各种理论和方法！融合语言学、计算机科学、数学于一体？"
        "近年来，深度学习极大推动了NLP的发展；Transformer、BERT、GPT等模型层出不穷，"
        "，，，甚至出现了多模态大模型。\n\n"
        "未来，NLP 将继续向通用人工智能迈进。"
    )

    splitter = ChineseRecursiveTextSplitter(
        chunk_size=60,
        chunk_overlap=0,
        keep_separator=True,
        is_separator_regex=True,
    )
    chunks = splitter.split_text(text)

    assert len(chunks) > 1
    for chk in chunks:
        assert len(chk) <= 60
        assert "\n\n\n" not in chk
