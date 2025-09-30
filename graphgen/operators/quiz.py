import asyncio
from collections import defaultdict

from tqdm.asyncio import tqdm as tqdm_async

from graphgen.models import JsonKVStorage, NetworkXStorage, OpenAIClient
from graphgen.templates import DESCRIPTION_REPHRASING_PROMPT
from graphgen.utils import detect_main_language, logger


async def quiz(
    synth_llm_client: OpenAIClient,
    graph_storage: NetworkXStorage,
    rephrase_storage: JsonKVStorage,
    max_samples: int = 1,
    max_concurrent: int = 1000,
) -> JsonKVStorage:
    """
    Get all edges and quiz them

    :param synth_llm_client: generate statements
    :param graph_storage: graph storage instance
    :param rephrase_storage: rephrase storage instance
    :param max_samples: max samples for each edge
    :param max_concurrent: max concurrent
    :return:
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_single_quiz(des: str, prompt: str, gt: str):
        async with semaphore:
            try:
                # 如果在rephrase_storage中已经存在，直接取出
                descriptions = await rephrase_storage.get_by_id(des)
                if descriptions:
                    return None

                new_description = await synth_llm_client.generate_answer(
                    prompt, temperature=1
                )
                return {des: [(new_description, gt)]}

            except Exception as e:  # pylint: disable=broad-except
                logger.error("Error when quizzing description %s: %s", des, e)
                return None

    edges = await graph_storage.get_all_edges()
    nodes = await graph_storage.get_all_nodes()

    results = defaultdict(list)
    tasks = []
    for edge in edges:
        edge_data = edge[2]

        description = edge_data["description"]
        language = "English" if detect_main_language(description) == "en" else "Chinese"

        results[description] = [(description, "yes")]

        for i in range(max_samples):
            if i > 0:
                tasks.append(
                    _process_single_quiz(
                        description,
                        DESCRIPTION_REPHRASING_PROMPT[language]["TEMPLATE"].format(
                            input_sentence=description
                        ),
                        "yes",
                    )
                )
            tasks.append(
                _process_single_quiz(
                    description,
                    DESCRIPTION_REPHRASING_PROMPT[language]["ANTI_TEMPLATE"].format(
                        input_sentence=description
                    ),
                    "no",
                )
            )

    for node in nodes:
        node_data = node[1]
        description = node_data["description"]
        language = "English" if detect_main_language(description) == "en" else "Chinese"

        results[description] = [(description, "yes")]

        for i in range(max_samples):
            if i > 0:
                tasks.append(
                    _process_single_quiz(
                        description,
                        DESCRIPTION_REPHRASING_PROMPT[language]["TEMPLATE"].format(
                            input_sentence=description
                        ),
                        "yes",
                    )
                )
            tasks.append(
                _process_single_quiz(
                    description,
                    DESCRIPTION_REPHRASING_PROMPT[language]["ANTI_TEMPLATE"].format(
                        input_sentence=description
                    ),
                    "no",
                )
            )

    for result in tqdm_async(
        asyncio.as_completed(tasks), total=len(tasks), desc="Quizzing descriptions"
    ):
        new_result = await result
        if new_result:
            for key, value in new_result.items():
                results[key].extend(value)

    for key, value in results.items():
        results[key] = list(set(value))
        await rephrase_storage.upsert({key: results[key]})

    return rephrase_storage
