import math

import gradio as gr

from graphgen.bases import BaseLLMWrapper
from graphgen.models import JsonKVStorage, NetworkXStorage
from graphgen.templates import STATEMENT_JUDGEMENT_PROMPT
from graphgen.utils import logger, run_concurrent, yes_no_loss_entropy


async def judge_statement(  # pylint: disable=too-many-statements
    trainee_llm_client: BaseLLMWrapper,
    graph_storage: NetworkXStorage,
    rephrase_storage: JsonKVStorage,
    re_judge: bool = False,
    progress_bar: gr.Progress = None,
) -> NetworkXStorage:
    """
    Get all edges and nodes and judge them

    :param trainee_llm_client: judge the statements to get comprehension loss
    :param graph_storage: graph storage instance
    :param rephrase_storage: rephrase storage instance
    :param re_judge: re-judge the relations
    :param progress_bar
    :return:
    """

    async def _judge_single_relation(
        edge: tuple,
    ):
        source_id = edge[0]
        target_id = edge[1]
        edge_data = edge[2]

        if (not re_judge) and "loss" in edge_data and edge_data["loss"] is not None:
            logger.debug(
                "Edge %s -> %s already judged, loss: %s, skip",
                source_id,
                target_id,
                edge_data["loss"],
            )
            return source_id, target_id, edge_data

        description = edge_data["description"]

        try:
            descriptions = rephrase_storage.get_by_id(description)
            assert descriptions is not None

            judgements = []
            gts = [gt for _, gt in descriptions]
            for description, gt in descriptions:
                judgement = await trainee_llm_client.generate_topk_per_token(
                    STATEMENT_JUDGEMENT_PROMPT["TEMPLATE"].format(statement=description)
                )
                judgements.append(judgement[0].top_candidates)

            loss = yes_no_loss_entropy(judgements, gts)

            logger.debug(
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

        graph_storage.update_edge(source_id, target_id, edge_data)
        return source_id, target_id, edge_data

    edges = graph_storage.get_all_edges()

    await run_concurrent(
        _judge_single_relation,
        edges,
        desc="Judging relations",
        unit="relation",
        progress_bar=progress_bar,
    )

    async def _judge_single_entity(
        node: tuple,
    ):
        node_id = node[0]
        node_data = node[1]

        if (not re_judge) and "loss" in node_data and node_data["loss"] is not None:
            logger.debug(
                "Node %s already judged, loss: %s, skip", node_id, node_data["loss"]
            )
            return node_id, node_data

        description = node_data["description"]

        try:
            descriptions = rephrase_storage.get_by_id(description)
            assert descriptions is not None

            judgements = []
            gts = [gt for _, gt in descriptions]
            for description, gt in descriptions:
                judgement = await trainee_llm_client.generate_topk_per_token(
                    STATEMENT_JUDGEMENT_PROMPT["TEMPLATE"].format(statement=description)
                )
                judgements.append(judgement[0].top_candidates)

            loss = yes_no_loss_entropy(judgements, gts)

            logger.debug("Node %s description: %s loss: %s", node_id, description, loss)

            node_data["loss"] = loss
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error in judging entity %s: %s", node_id, e)
            logger.error("Use default loss 0.1")
            node_data["loss"] = -math.log(0.1)

        graph_storage.update_node(node_id, node_data)
        return node_id, node_data

    nodes = graph_storage.get_all_nodes()

    await run_concurrent(
        _judge_single_entity,
        nodes,
        desc="Judging entities",
        unit="entity",
        progress_bar=progress_bar,
    )

    return graph_storage
