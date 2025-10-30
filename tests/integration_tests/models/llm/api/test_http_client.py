# pylint: disable=protected-access
import math

import pytest

from graphgen.models.llm.api.http_client import HTTPClient


class DummyTokenizer:
    def encode(self, text: str):
        # simple tokenization: split on spaces
        return text.split()


class _MockResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    async def json(self):
        return self._data


class _PostCtx:
    def __init__(self, data):
        self._resp = _MockResponse(data)

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, exc_type, exc, tb):
        return False


class MockSession:
    def __init__(self, data):
        self._data = data
        self.closed = False

    def post(self, *args, **kwargs):
        return _PostCtx(self._data)

    async def close(self):
        self.closed = True


class DummyLimiter:
    def __init__(self):
        self.calls = []

    async def wait(self, *args, **kwargs):
        self.calls.append((args, kwargs))


@pytest.mark.asyncio
async def test_generate_answer_records_usage_and_uses_limiters():
    # arrange
    data = {
        "choices": [{"message": {"content": "Hello <think>world</think>!"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    client = HTTPClient(model="m", base_url="http://test")
    client._session = MockSession(data)
    client.tokenizer = DummyTokenizer()
    client.system_prompt = "sys"
    client.temperature = 0.0
    client.top_p = 1.0
    client.max_tokens = 10
    client.filter_think_tags = lambda s: s.replace("<think>", "").replace(
        "</think>", ""
    )
    rpm = DummyLimiter()
    tpm = DummyLimiter()
    client.rpm = rpm
    client.tpm = tpm
    client.request_limit = True

    # act
    out = await client.generate_answer("hi", history=["u1", "a1"])

    # assert
    assert out == "Hello world!"
    assert client.token_usage[-1] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert len(rpm.calls) == 1
    assert len(tpm.calls) == 1


@pytest.mark.asyncio
async def test_generate_topk_per_token_parses_logprobs():
    # arrange
    # create two token items with top_logprobs
    data = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": "A",
                            "logprob": math.log(0.6),
                            "top_logprobs": [
                                {"token": "A", "logprob": math.log(0.6)},
                                {"token": "B", "logprob": math.log(0.4)},
                            ],
                        },
                        {
                            "token": "B",
                            "logprob": math.log(0.2),
                            "top_logprobs": [
                                {"token": "B", "logprob": math.log(0.2)},
                                {"token": "C", "logprob": math.log(0.8)},
                            ],
                        },
                    ]
                }
            }
        ]
    }
    client = HTTPClient(model="m", base_url="http://test")
    client._session = MockSession(data)
    client.tokenizer = DummyTokenizer()
    client.system_prompt = None
    client.temperature = 0.0
    client.top_p = 1.0
    client.max_tokens = 10
    client.topk_per_token = 2

    # act
    tokens = await client.generate_topk_per_token("hi", history=[])

    # assert
    assert len(tokens) == 2
    # check probabilities and top_candidates
    assert abs(tokens[0].prob - 0.6) < 1e-9
    assert abs(tokens[1].prob - 0.2) < 1e-9
    assert len(tokens[0].top_candidates) == 2
    assert tokens[0].top_candidates[0].text == "A"
    assert tokens[0].top_candidates[1].text == "B"
