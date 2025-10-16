from typing import Any

from graphgen.bases import BaseLLMClient
from graphgen.models import (
    AggregatedGenerator,
    AtomicGenerator,
    CoTGenerator,
    MultiHopGenerator,
)
from graphgen.utils import logger, run_concurrent


async def generate_qas(
    llm_client: BaseLLMClient,
    batches: list[
        tuple[
            list[tuple[str, dict]], list[tuple[Any, Any, dict] | tuple[Any, Any, Any]]
        ]
    ],
    generation_config: dict,
    progress_bar=None,
) -> list[dict[str, Any]]:
    """
    Generate question-answer pairs based on nodes and edges.
    :param llm_client: LLM client
    :param batches
    :param generation_config
    :param progress_bar
    :return: QA pairs
    """
    mode = generation_config["mode"]
    logger.info("[Generation] mode: %s, batches: %d", mode, len(batches))

    if mode == "atomic":
        generator = AtomicGenerator(llm_client)
    elif mode == "aggregated":
        generator = AggregatedGenerator(llm_client)
    elif mode == "multi_hop":
        generator = MultiHopGenerator(llm_client)
    elif mode == "cot":
        generator = CoTGenerator(llm_client)
    else:
        raise ValueError(f"Unsupported generation mode: {mode}")

    results = await run_concurrent(
        generator.generate,
        batches,
        desc="[4/4]Generating QAs",
        unit="batch",
        progress_bar=progress_bar,
    )

    # format
    data_format = generation_config["data_format"]
    logger.info("Output data format: %s", data_format)

    results = generator.format_generation_results(
        results, output_data_format=data_format
    )

    return results
