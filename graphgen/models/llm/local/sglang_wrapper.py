import math
from typing import Any, Dict, List, Optional

from graphgen.bases.base_llm_wrapper import BaseLLMWrapper
from graphgen.bases.datatypes import Token


class SGLangWrapper(BaseLLMWrapper):
    """
    Async inference backend based on SGLang offline engine.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        topk: int = 5,
        **kwargs: Any,
    ):
        super().__init__(temperature=temperature, top_p=top_p, **kwargs)
        try:
            import sglang as sgl
            from sglang.utils import async_stream_and_merge, stream_and_merge
        except ImportError as exc:
            raise ImportError(
                "SGLangWrapper requires sglang. Install it with: "
                "uv pip install sglang --prerelease=allow"
            ) from exc

        self.model_path: str = model
        self.temperature = temperature
        self.top_p = top_p
        self.topk = topk

        # Initialise the offline engine
        self.engine = sgl.Engine(model_path=self.model_path)

        # Keep helpers for streaming
        self.async_stream_and_merge = async_stream_and_merge
        self.stream_and_merge = stream_and_merge

    @staticmethod
    def _build_sampling_params(
        temperature: float,
        top_p: float,
        max_tokens: int,
        topk: int,
        logprobs: bool = False,
    ) -> Dict[str, Any]:
        """Build SGLang-compatible sampling-params dict."""
        params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_tokens,
        }
        if logprobs and topk > 0:
            params["logprobs"] = topk
        return params

    def _prep_prompt(self, text: str, history: Optional[List[dict]] = None) -> str:
        """Convert raw text (+ optional history) into a single prompt string."""
        parts = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        if history:
            assert len(history) % 2 == 0, "History must have even length (u/a turns)."
            parts.extend([item["content"] for item in history])
        parts.append(text)
        return "\n".join(parts)

    def _tokens_from_output(self, output: Dict[str, Any]) -> List[Token]:
        tokens: List[Token] = []

        meta = output.get("meta_info", {})
        logprobs = meta.get("output_token_logprobs", [])
        topks = meta.get("output_top_logprobs", [])

        tokenizer = self.engine.tokenizer_manager.tokenizer

        for idx, (lp, tid, _) in enumerate(logprobs):
            prob = math.exp(lp)
            tok_str = tokenizer.decode([tid])

            top_candidates = []
            if self.topk > 0 and idx < len(topks):
                for t_lp, t_tid, _ in topks[idx][: self.topk]:
                    top_candidates.append(
                        Token(text=tokenizer.decode([t_tid]), prob=math.exp(t_lp))
                    )

            tokens.append(Token(text=tok_str, prob=prob, top_candidates=top_candidates))

        return tokens

    async def generate_answer(
        self,
        text: str,
        history: Optional[List[str]] = None,
        **extra: Any,
    ) -> str:
        prompt = self._prep_prompt(text, history)
        sampling_params = self._build_sampling_params(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            topk=0,  # no logprobs needed for simple generation
        )

        outputs = await self.engine.async_generate([prompt], sampling_params)
        return self.filter_think_tags(outputs[0]["text"])

    async def generate_topk_per_token(
        self,
        text: str,
        history: Optional[List[str]] = None,
        **extra: Any,
    ) -> List[Token]:
        prompt = self._prep_prompt(text, history)
        sampling_params = self._build_sampling_params(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=1,  # keep short for token-level analysis
            topk=self.topk,
        )

        outputs = await self.engine.async_generate(
            [prompt], sampling_params, return_logprob=True, top_logprobs_num=5
        )
        print(outputs)
        return self._tokens_from_output(outputs[0])

    async def generate_inputs_prob(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        raise NotImplementedError(
            "SGLangWrapper does not support per-token logprobs yet."
        )

    def shutdown(self) -> None:
        """Gracefully shutdown the SGLang engine."""
        if hasattr(self, "engine"):
            self.engine.shutdown()

    def restart(self) -> None:
        """Restart the SGLang engine."""
        self.shutdown()
        self.engine = self.engine.__class__(model_path=self.model_path)
