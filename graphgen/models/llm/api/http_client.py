import asyncio
import math
from typing import Any, Dict, List, Optional

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from graphgen.bases.base_llm_wrapper import BaseLLMWrapper
from graphgen.bases.datatypes import Token
from graphgen.models.llm.limitter import RPM, TPM


class HTTPClient(BaseLLMWrapper):
    """
    A generic async HTTP client for LLMs compatible with OpenAI's chat/completions format.
    It uses aiohttp for making requests and includes retry logic and token usage tracking.
    Usage example:
        client = HTTPClient(
            model_name="gpt-4o-mini",
            base_url="http://localhost:8080",
            api_key="your_api_key",
            json_mode=True,
            seed=42,
            topk_per_token=5,
            request_limit=True,
        )

        answer = await client.generate_answer("Hello, world!")
        tokens = await client.generate_topk_per_token("Hello, world!")
    """

    _instance: Optional["HTTPClient"] = None
    _lock = asyncio.Lock()

    def __new__(cls, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        json_mode: bool = False,
        seed: Optional[int] = None,
        topk_per_token: int = 5,
        request_limit: bool = False,
        rpm: Optional[RPM] = None,
        tpm: Optional[TPM] = None,
        **kwargs: Any,
    ):
        # Initialize only once in the singleton pattern
        if getattr(self, "_initialized", False):
            return
        self._initialized: bool = True
        super().__init__(**kwargs)
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.json_mode = json_mode
        self.seed = seed
        self.topk_per_token = topk_per_token
        self.request_limit = request_limit
        self.rpm = rpm or RPM()
        self.tpm = tpm or TPM()

        self.token_usage: List[Dict[str, int]] = []
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = (
                {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def _build_body(self, text: str, history: List[str]) -> Dict[str, Any]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # chatml format: alternating user and assistant messages
        if history and isinstance(history[0], dict):
            messages.extend(history)

        messages.append({"role": "user", "content": text})

        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.seed:
            body["seed"] = self.seed
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def generate_answer(
        self,
        text: str,
        history: Optional[List[str]] = None,
        **extra: Any,
    ) -> str:
        body = self._build_body(text, history or [])
        prompt_tokens = sum(
            len(self.tokenizer.encode(m["content"])) for m in body["messages"]
        )
        est = prompt_tokens + body["max_tokens"]

        if self.request_limit:
            await self.rpm.wait(silent=True)
            await self.tpm.wait(est, silent=True)

        async with self.session.post(
            f"{self.base_url}/chat/completions",
            json=body,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        msg = data["choices"][0]["message"]["content"]
        if "usage" in data:
            self.token_usage.append(
                {
                    "prompt_tokens": data["usage"]["prompt_tokens"],
                    "completion_tokens": data["usage"]["completion_tokens"],
                    "total_tokens": data["usage"]["total_tokens"],
                }
            )
        return self.filter_think_tags(msg)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def generate_topk_per_token(
        self,
        text: str,
        history: Optional[List[str]] = None,
        **extra: Any,
    ) -> List[Token]:
        body = self._build_body(text, history or [])
        body["max_tokens"] = 1
        if self.topk_per_token > 0:
            body["logprobs"] = True
            body["top_logprobs"] = self.topk_per_token

        async with self.session.post(
            f"{self.base_url}/chat/completions",
            json=body,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        token_logprobs = data["choices"][0]["logprobs"]["content"]
        tokens = []
        for item in token_logprobs:
            candidates = [
                Token(t["token"], math.exp(t["logprob"])) for t in item["top_logprobs"]
            ]
            tokens.append(
                Token(
                    item["token"], math.exp(item["logprob"]), top_candidates=candidates
                )
            )
        return tokens

    async def generate_inputs_prob(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        raise NotImplementedError(
            "generate_inputs_prob is not implemented in HTTPClient"
        )
