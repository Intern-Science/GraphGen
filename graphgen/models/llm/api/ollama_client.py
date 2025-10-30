from typing import Any, Dict, List, Optional

from graphgen.bases.base_llm_wrapper import BaseLLMWrapper
from graphgen.bases.datatypes import Token
from graphgen.models.llm.limitter import RPM, TPM


class OllamaClient(BaseLLMWrapper):
    """
    Requires a local or remote Ollama server to be running (default port 11434).
    The top_logprobs field is not yet implemented by the official API.
    """

    def __init__(
        self,
        *,
        model: str = "gemma3",
        base_url: str = "http://localhost:11434",
        json_mode: bool = False,
        seed: Optional[int] = None,
        topk_per_token: int = 5,
        request_limit: bool = False,
        rpm: Optional[RPM] = None,
        tpm: Optional[TPM] = None,
        **kwargs: Any,
    ):
        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "Ollama SDK is not installed."
                "It is required to use OllamaClient."
                "Please install it with `pip install ollama`."
            ) from e
        super().__init__(**kwargs)
        self.model_name = model
        self.base_url = base_url
        self.json_mode = json_mode
        self.seed = seed
        self.topk_per_token = topk_per_token
        self.request_limit = request_limit
        self.rpm = rpm or RPM()
        self.tpm = tpm or TPM()
        self.token_usage: List[Dict[str, int]] = []

        self.client = ollama.AsyncClient(host=self.base_url)

    async def generate_answer(
        self,
        text: str,
        history: Optional[List[Dict[str, str]]] = None,
        **extra: Any,
    ) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": text})

        options = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.max_tokens,
        }
        if self.seed is not None:
            options["seed"] = self.seed

        prompt_tokens = sum(len(self.tokenizer.encode(m["content"])) for m in messages)
        est = prompt_tokens + self.max_tokens
        if self.request_limit:
            await self.rpm.wait(silent=True)
            await self.tpm.wait(est, silent=True)

        response = await self.client.chat(
            model=self.model_name,
            messages=messages,
            format="json" if self.json_mode else "",
            options=options,
            stream=False,
        )

        usage = response.get("prompt_eval_count", 0), response.get("eval_count", 0)
        self.token_usage.append(
            {
                "prompt_tokens": usage[0],
                "completion_tokens": usage[1],
                "total_tokens": sum(usage),
            }
        )
        content = response["message"]["content"]
        return self.filter_think_tags(content)

    async def generate_topk_per_token(
        self,
        text: str,
        history: Optional[List[Dict[str, str]]] = None,
        **extra: Any,
    ) -> List[Token]:
        raise NotImplementedError("Ollama API does not support per-token top-k yet.")

    async def generate_inputs_prob(
        self, text: str, history: Optional[List[Dict[str, str]]] = None, **extra: Any
    ) -> List[Token]:
        raise NotImplementedError("Ollama API does not support per-token logprobs yet.")
