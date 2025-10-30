# pylint: disable=redefined-outer-name
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphgen.models import OllamaClient


# ----------------- fixture -----------------
@pytest.fixture
def mock_ollama_pkg():
    """
    mock ollama
    """
    ollama_mock = MagicMock()
    ollama_mock.AsyncClient = AsyncMock
    with patch.dict("sys.modules", {"ollama": ollama_mock}):
        yield ollama_mock


@pytest.fixture
def ollama_client(mock_ollama_pkg) -> OllamaClient:
    """
    Returns a default-configured OllamaClient with client.chat mocked
    """
    cli = OllamaClient(model="gemma3", base_url="http://test:11434")
    cli.tokenizer = MagicMock()
    cli.tokenizer.encode = MagicMock(side_effect=lambda x: x.split())
    cli.client.chat = AsyncMock(
        return_value={
            "message": {"content": "hi from ollama"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
    )
    return cli


@pytest.mark.asyncio
async def test_generate_answer_basic(ollama_client: OllamaClient):
    ans = await ollama_client.generate_answer("hello")
    assert ans == "hi from ollama"
    ollama_client.client.chat.assert_awaited_once()
    call = ollama_client.client.chat.call_args
    assert call.kwargs["model"] == "gemma3"
    assert call.kwargs["messages"][-1]["content"] == "hello"
    assert call.kwargs["stream"] is False


@pytest.mark.asyncio
async def test_generate_answer_with_history(ollama_client: OllamaClient):
    hist = [{"role": "user", "content": "prev"}]
    await ollama_client.generate_answer("now", history=hist)
    msgs = ollama_client.client.chat.call_args.kwargs["messages"]
    assert msgs[-2]["content"] == "prev"
    assert msgs[-1]["content"] == "now"


@pytest.mark.asyncio
async def test_token_usage_recorded(ollama_client: OllamaClient):
    await ollama_client.generate_answer("test")
    assert len(ollama_client.token_usage) == 1
    assert ollama_client.token_usage[0]["prompt_tokens"] == 10
    assert ollama_client.token_usage[0]["completion_tokens"] == 5
    assert ollama_client.token_usage[0]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_rpm_tpm_limiter_called(ollama_client: OllamaClient):
    ollama_client.request_limit = True
    with patch.object(ollama_client.rpm, "wait", AsyncMock()) as rpm_mock, patch.object(
        ollama_client.tpm, "wait", AsyncMock()
    ) as tpm_mock:

        await ollama_client.generate_answer("limited")
        rpm_mock.assert_awaited_once_with(silent=True)
        tpm_mock.assert_awaited_once_with(
            ollama_client.max_tokens + len("limited".split()), silent=True
        )


def test_import_error_when_ollama_missing():
    with patch.dict("sys.modules", {"ollama": None}):
        with pytest.raises(ImportError, match="Ollama SDK is not installed"):
            OllamaClient()


@pytest.mark.asyncio
async def test_generate_inputs_prob_not_implemented(ollama_client: OllamaClient):
    with pytest.raises(NotImplementedError):
        await ollama_client.generate_inputs_prob("any")
