import asyncio
import math

from tqdm.asyncio import tqdm as tqdm_async

from graphgen.models import JsonKVStorage, NetworkXStorage, OpenAIModel
from graphgen.templates import STATEMENT_JUDGEMENT_PROMPT
from graphgen.utils import logger, yes_no_loss_entropy


async def judge_statement(  # pylint: disable=too-many-statements
    trainee_llm_client: OpenAIModel,
    graph_storage: NetworkXStorage,
    rephrase_storage: JsonKVStorage,
    re_judge: bool = False,
    max_concurrent: int = 1000,
) -> NetworkXStorage:
    """
    Get all edges and nodes and judge them

    :param trainee_llm_client: judge the statements to get comprehension loss
    :param graph_storage: graph storage instance
    :param rephrase_storage: rephrase storage instance
    :param re_judge: re-judge the relations
    :param max_concurrent: max concurrent
    :return:
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _judge_single_relation(
        edge: tuple,
    ):
        async with semaphore:
            source_id = edge[0]
            target_id = edge[1]
            edge_data = edge[2]

            if (not re_judge) and "loss" in edge_data and edge_data["loss"] is not None:
                logger.info(
                    "Edge %s -> %s already judged, loss: %s, skip",
                    source_id,
                    target_id,
                    edge_data["loss"],
                )
                return source_id, target_id, edge_data

            description = edge_data["description"]

            try:
                descriptions = await rephrase_storage.get_by_id(description)
                assert descriptions is not None

                judgements = []
                gts = [gt for _, gt in descriptions]
                for description, gt in descriptions:
                    judgement = await trainee_llm_client.generate_topk_per_token(
                        STATEMENT_JUDGEMENT_PROMPT["TEMPLATE"].format(
                            statement=description
                        )
                    )
                    judgements.append(judgement[0].top_candidates)

                loss = yes_no_loss_entropy(judgements, gts)

                logger.info(
                    "Edge %s -> %s description: %s loss: %s",
                    source_id,
                    target_id,
                    description,
                    loss,
                )

                edge_data["loss"] = loss
            except Exception as e:  # pylint: disable=broad-except
                logger.error(
                    "Error in judging relation %s -> %s: %s", source_id, target_id, e
                )
                logger.info("Use default loss 0.1")
                edge_data["loss"] = -math.log(0.1)

            await graph_storage.update_edge(source_id, target_id, edge_data)
            return source_id, target_id, edge_data

    edges = await graph_storage.get_all_edges()

    results = []
    for result in tqdm_async(
        asyncio.as_completed([_judge_single_relation(edge) for edge in edges]),
        total=len(edges),
        desc="Judging relations",
    ):
        results.append(await result)

    async def _judge_single_entity(
        node: tuple,
    ):
        async with semaphore:
            node_id = node[0]
            node_data = node[1]

            if (not re_judge) and "loss" in node_data and node_data["loss"] is not None:
                logger.info(
                    "Node %s already judged, loss: %s, skip", node_id, node_data["loss"]
                )
                return node_id, node_data

            description = node_data["description"]

            try:
                descriptions = await rephrase_storage.get_by_id(description)
                assert descriptions is not None

                judgements = []
                gts = [gt for _, gt in descriptions]
                for description, gt in descriptions:
                    judgement = await trainee_llm_client.generate_topk_per_token(
                        STATEMENT_JUDGEMENT_PROMPT["TEMPLATE"].format(
                            statement=description
                        )
                    )
                    judgements.append(judgement[0].top_candidates)

                loss = yes_no_loss_entropy(judgements, gts)

                logger.info(
                    "Node %s description: %s loss: %s", node_id, description, loss
                )

                node_data["loss"] = loss
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Error in judging entity %s: %s", node_id, e)
                logger.info("Use default loss 0.1")
                node_data["loss"] = -math.log(0.1)

            await graph_storage.update_node(node_id, node_data)
            return node_id, node_data

    nodes = await graph_storage.get_all_nodes()

    results = []
    for result in tqdm_async(
        asyncio.as_completed([_judge_single_entity(node) for node in nodes]),
        total=len(nodes),
        desc="Judging entities",
    ):
        results.append(await result)

    return graph_storage
