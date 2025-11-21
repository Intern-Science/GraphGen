from collections import defaultdict

import gradio as gr

from graphgen.bases import BaseLLMWrapper
from graphgen.models import JsonKVStorage, NetworkXStorage, QuizGenerator
from graphgen.utils import logger, run_concurrent


async def quiz(
    synth_llm_client: BaseLLMWrapper,
    graph_storage: NetworkXStorage,
    rephrase_storage: JsonKVStorage,
    max_samples: int = 1,
    progress_bar: gr.Progress = None,
) -> JsonKVStorage:
    """
    Get all edges and quiz them using QuizGenerator.

    :param synth_llm_client: generate statements
    :param graph_storage: graph storage instance
    :param rephrase_storage: rephrase storage instance
    :param max_samples: max samples for each edge
    :param progress_bar
    :return:
    """

    generator = QuizGenerator(synth_llm_client)

    async def _process_single_quiz(item: tuple[str, str, str]):
        description, template_type, gt = item
        try:
            # if rephrase_storage exists already, directly get it
            descriptions = await rephrase_storage.get_by_id(description)
            if descriptions:
                return None

            prompt = generator.build_prompt_for_description(description, template_type)
            new_description = await synth_llm_client.generate_answer(
                prompt, temperature=1
            )
            rephrased_text = generator.parse_rephrased_text(new_description)
            return {description: [(rephrased_text, gt)]}

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error when quizzing description %s: %s", description, e)
            return None

    edges = await graph_storage.get_all_edges()
    nodes = await graph_storage.get_all_nodes()

    results = defaultdict(list)
    items = []
    for edge in edges:
        edge_data = edge[2]
        description = edge_data["description"]

        results[description] = [(description, "yes")]

        for i in range(max_samples):
            if i > 0:
                items.append((description, "TEMPLATE", "yes"))
            items.append((description, "ANTI_TEMPLATE", "no"))

    for node in nodes:
        node_data = node[1]
        description = node_data["description"]

        results[description] = [(description, "yes")]

        for i in range(max_samples):
            if i > 0:
                items.append((description, "TEMPLATE", "yes"))
            items.append((description, "ANTI_TEMPLATE", "no"))

    quiz_results = await run_concurrent(
        _process_single_quiz,
        items,
        desc="Quizzing descriptions",
        unit="description",
        progress_bar=progress_bar,
    )

    for new_result in quiz_results:
        if new_result:
            for key, value in new_result.items():
                results[key].extend(value)

    for key, value in results.items():
        results[key] = list(set(value))
        await rephrase_storage.upsert({key: results[key]})

    return rephrase_storage
