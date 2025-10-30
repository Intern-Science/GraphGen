from unittest.mock import MagicMock

import pytest

from graphgen.models.llm.local.hf_wrapper import HuggingFaceWrapper


@pytest.fixture(autouse=True)
def mock_hf(monkeypatch):
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<EOS>"
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.decode.return_value = "hello"
    mock_tokenizer.encode.return_value = [1, 2, 3]
    monkeypatch.setattr(
        "graphgen.models.llm.local.hf_wrapper.AutoTokenizer.from_pretrained",
        lambda *a, **kw: mock_tokenizer,
    )

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = MagicMock(
        __getitem__=lambda s, k: [0, 1, 2, 3], shape=(1, 4)
    )
    mock_model.eval.return_value = None
    monkeypatch.setattr(
        "graphgen.models.llm.local.hf_wrapper.AutoModelForCausalLM.from_pretrained",
        lambda *a, **kw: mock_model,
    )

    monkeypatch.setattr(
        "graphgen.models.llm.local.hf_wrapper.torch.no_grad", MagicMock()
    )

    return mock_tokenizer, mock_model


@pytest.mark.asyncio
async def test_generate_answer():
    wrapper = HuggingFaceWrapper("fake-model")
    result = await wrapper.generate_answer("hi")
    assert isinstance(result, str)
