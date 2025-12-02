import math
from typing import Any, Dict, List, Optional

import openai
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, AsyncAzureOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from graphgen.bases.base_llm_wrapper import BaseLLMWrapper
from graphgen.bases.datatypes import Token
from graphgen.models.llm.limitter import RPM, TPM


def get_top_response_tokens(response: openai.ChatCompletion) -> List[Token]:
    token_logprobs = response.choices[0].logprobs.content
    tokens = []
    for token_prob in token_logprobs:
        prob = math.exp(token_prob.logprob)
        candidate_tokens = [
            Token(t.token, math.exp(t.logprob)) for t in token_prob.top_logprobs
        ]
        token = Token(token_prob.token, prob, top_candidates=candidate_tokens)
        tokens.append(token)
    return tokens


class OpenAIClient(BaseLLMWrapper):
    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        json_mode: bool = False,
        seed: Optional[int] = None,
        topk_per_token: int = 5,  # number of topk tokens to generate for each token
        request_limit: bool = False,
        rpm: Optional[RPM] = None,
        tpm: Optional[TPM] = None,
        backend: str = "openai_api",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.api_key = api_key
        self.api_version = api_version # required for Azure OpenAI
        self.base_url = base_url
        self.json_mode = json_mode
        self.seed = seed
        self.topk_per_token = topk_per_token

        self.token_usage: list = []
        self.request_limit = request_limit
        self.rpm = rpm or RPM()
        self.tpm = tpm or TPM()

        assert (
            backend in ("openai_api", "azure_openai_api")
        ), f"Unsupported backend '{backend}'. Use 'openai_api' or 'azure_openai_api'."
        self.backend = backend

        self.__post_init__()

    def __post_init__(self):

        api_name = self.backend.replace("_", " ")
        assert self.api_key is not None, f"Please provide api key to access {api_name}."
        if self.backend == "openai_api":
            self.client = AsyncOpenAI(
                api_key=self.api_key or "dummy", base_url=self.base_url
            )
        elif self.backend == "azure_openai_api":
            assert self.api_version is not None, f"Please provide api_version for {api_name}."
            assert self.base_url is not None, f"Please provide base_url for {api_name}."
            self.client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.base_url,
                api_version=self.api_version,
                azure_deployment=self.model,
            )
        else:
            raise ValueError(f"Unsupported backend {self.backend}. Use 'openai_api' or 'azure_openai_api'.")

    def _pre_generate(self, text: str, history: List[str]) -> Dict:
        kwargs = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.seed:
            kwargs["seed"] = self.seed
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": text})

        if history:
            assert len(history) % 2 == 0, "History should have even number of elements."
            messages = history + messages

        kwargs["messages"] = messages
        return kwargs

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, APITimeoutError)
        ),
    )
    async def generate_topk_per_token(
        self,
        text: str,
        history: Optional[List[str]] = None,
        **extra: Any,
    ) -> List[Token]:
        kwargs = self._pre_generate(text, history)
        if self.topk_per_token > 0:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = self.topk_per_token

        # Limit max_tokens to 1 to avoid long completions
        kwargs["max_tokens"] = 1

        completion = await self.client.chat.completions.create(  # pylint: disable=E1125
            model=self.model, **kwargs
        )

        tokens = get_top_response_tokens(completion)

        return tokens

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, APITimeoutError)
        ),
    )
    async def generate_answer(
        self,
        text: str,
        history: Optional[List[str]] = None,
        **extra: Any,
    ) -> str:
        kwargs = self._pre_generate(text, history)

        prompt_tokens = 0
        for message in kwargs["messages"]:
            prompt_tokens += len(self.tokenizer.encode(message["content"]))
        estimated_tokens = prompt_tokens + kwargs["max_tokens"]

        if self.request_limit:
            await self.rpm.wait(silent=True)
            await self.tpm.wait(estimated_tokens, silent=True)

        completion = await self.client.chat.completions.create(  # pylint: disable=E1125
            model=self.model, **kwargs
        )
        if hasattr(completion, "usage"):
            self.token_usage.append(
                {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }
            )
        return self.filter_think_tags(completion.choices[0].message.content)

    async def generate_inputs_prob(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        """Generate probabilities for each token in the input."""
        raise NotImplementedError
