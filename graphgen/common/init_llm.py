import os
from typing import Any, Dict, Optional

from graphgen.bases import BaseLLMWrapper
from graphgen.models import Tokenizer


class LLMFactory:
    """
    A factory class to create LLM wrapper instances based on the specified backend.
    Supported backends include:
    - http_api: HTTPClient
    - openai_api: OpenAIClient
    - ollama_api: OllamaClient
    - huggingface: HuggingFaceWrapper
    - sglang: SGLangWrapper
    """

    @staticmethod
    def create_llm_wrapper(backend: str, config: Dict[str, Any]) -> BaseLLMWrapper:
        # add tokenizer
        tokenizer: Tokenizer = Tokenizer(
            os.environ.get("TOKENIZER_MODEL", "cl100k_base"),
        )
        config["tokenizer"] = tokenizer
        if backend == "http_api":
            from graphgen.models.llm.api.http_client import HTTPClient

            return HTTPClient(**config)
        if backend in ("openai_api", "azure_openai_api"):
            from graphgen.models.llm.api.openai_client import OpenAIClient

            # pass in concrete backend to the OpenAIClient so that internally we can distinguish
            # between OpenAI and Azure OpenAI
            return OpenAIClient(**config, backend=backend)
        if backend == "ollama_api":
            from graphgen.models.llm.api.ollama_client import OllamaClient

            return OllamaClient(**config)
        if backend == "huggingface":
            from graphgen.models.llm.local.hf_wrapper import HuggingFaceWrapper

            return HuggingFaceWrapper(**config)
        if backend == "sglang":
            from graphgen.models.llm.local.sglang_wrapper import SGLangWrapper

            return SGLangWrapper(**config)

        # if backend == "vllm":
        #     from graphgen.models.llm.local.vllm_wrapper import VLLMWrapper
        #
        #     return VLLMWrapper(**config)

        raise NotImplementedError(f"Backend {backend} is not implemented yet.")


def _load_env_group(prefix: str) -> Dict[str, Any]:
    """
    Collect environment variables with the given prefix into a dictionary,
    stripping the prefix from the keys.
    """
    return {
        k[len(prefix) :].lower(): v
        for k, v in os.environ.items()
        if k.startswith(prefix)
    }


def init_llm(model_type: str) -> Optional[BaseLLMWrapper]:
    if model_type == "synthesizer":
        prefix = "SYNTHESIZER_"
    elif model_type == "trainee":
        prefix = "TRAINEE_"
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented yet.")
    config = _load_env_group(prefix)
    # if config is empty, return None
    if not config:
        return None
    backend = config.pop("backend")
    llm_wrapper = LLMFactory.create_llm_wrapper(backend, config)
    return llm_wrapper


# TODO: use ray serve when loading large models to avoid re-loading in each actor
