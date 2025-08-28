import asyncio

import gradio as gr
from tqdm.asyncio import tqdm as tqdm_async

from graphgen.models import (
    JsonKVStorage,
    NetworkXStorage,
    OpenAIModel,
    Tokenizer,
    TraverseStrategy,
)
from graphgen.operators.kg.split_kg import get_batches_with_strategy
from graphgen.templates import (
    ANSWER_REPHRASING_PROMPT,
    MULTI_HOP_GENERATION_PROMPT,
    QUESTION_GENERATION_PROMPT,
)
from graphgen.utils import compute_content_hash, detect_main_language, logger


async def _pre_tokenize(
    graph_storage: NetworkXStorage, tokenizer: Tokenizer, edges: list, nodes: list
) -> tuple:

    sem = asyncio.Semaphore(1000)

    async def handle_edge(edge: tuple) -> tuple:
        async with sem:
            if "length" not in edge[2]:
                edge[2]["length"] = len(
                    await asyncio.get_event_loop().run_in_executor(
                        None, tokenizer.encode_string, edge[2]["description"]
                    )
                )
            return edge

    async def handle_node(node: dict) -> dict:
        async with sem:
            if "length" not in node[1]:
                node[1]["length"] = len(
                    await asyncio.get_event_loop().run_in_executor(
                        None, tokenizer.encode_string, node[1]["description"]
                    )
                )
            return node

    new_edges = []
    new_nodes = []

    for result in tqdm_async(
        asyncio.as_completed([handle_edge(edge) for edge in edges]),
        total=len(edges),
        desc="Pre-tokenizing edges",
    ):
        new_edge = await result
        await graph_storage.update_edge(new_edge[0], new_edge[1], new_edge[2])
        new_edges.append(new_edge)

    for result in tqdm_async(
        asyncio.as_completed([handle_node(node) for node in nodes]),
        total=len(nodes),
        desc="Pre-tokenizing nodes",
    ):
        new_node = await result
        await graph_storage.update_node(new_node[0], new_node[1])
        new_nodes.append(new_node)

    await graph_storage.index_done_callback()
    return new_edges, new_nodes


async def _construct_rephrasing_prompt(
    _process_nodes: list,
    _process_edges: list,
    text_chunks_storage: JsonKVStorage,
    add_context: bool = False,
) -> str:
    entities = [
        f"{_process_node['node_id']}: {_process_node['description']}"
        for _process_node in _process_nodes
    ]
    relations = [
        f"{_process_edge[0]} -- {_process_edge[1]}: {_process_edge[2]['description']}"
        for _process_edge in _process_edges
    ]

    entities_str = "\n".join(
        [f"{index + 1}. {entity}" for index, entity in enumerate(entities)]
    )
    relations_str = "\n".join(
        [f"{index + 1}. {relation}" for index, relation in enumerate(relations)]
    )
    language = (
        "Chinese"
        if detect_main_language(entities_str + relations_str) == "zh"
        else "English"
    )

    if add_context:
        original_ids = [
            node["source_id"].split("<SEP>")[0] for node in _process_nodes
        ] + [edge[2]["source_id"].split("<SEP>")[0] for edge in _process_edges]

        original_ids = list(set(original_ids))
        original_text = await text_chunks_storage.get_by_ids(original_ids)
        original_text = "\n".join(
            [
                f"{index + 1}. {text['content']}"
                for index, text in enumerate(original_text)
            ]
        )

        prompt = ANSWER_REPHRASING_PROMPT[language]["CONTEXT_TEMPLATE"].format(
            language=language,
            original_text=original_text,
            entities=entities_str,
            relationships=relations_str,
        )
        return prompt

    prompt = ANSWER_REPHRASING_PROMPT[language]["TEMPLATE"].format(
        language=language, entities=entities_str, relationships=relations_str
    )
    return prompt


def get_average_loss(batch: tuple, loss_strategy: str) -> float:
    try:
        if loss_strategy == "only_edge":
            return sum(edge[2]["loss"] for edge in batch[1]) / len(batch[1])
        if loss_strategy == "both":
            return sum(edge[2]["loss"] for edge in batch[1]) + sum(
                node["loss"] for node in batch[0]
            ) / (len(batch[0]) + len(batch[1]))
        raise ValueError("Invalid loss strategy")
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error calculating average loss: %s", e)
        return -1.0


def _post_process_synthetic_data(data):
    block = data.split("\n\n")
    qas = []
    for line in block:
        if "Question:" in line and "Answer:" in line:
            question = line.split("Question:")[1].split("Answer:")[0].strip()
            answer = line.split("Answer:")[1].strip()
            qas.append({"question": question, "answer": answer})
        elif "问题：" in line and "答案：" in line:
            question = line.split("问题：")[1].split("答案：")[0].strip()
            answer = line.split("答案：")[1].strip()
            qas.append({"question": question, "answer": answer})
        elif "问题:" in line and "回答:" in line:
            question = line.split("问题:")[1].split("回答:")[0].strip()
            answer = line.split("回答:")[1].strip()
            qas.append({"question": question, "answer": answer})
    return qas


async def traverse_graph_by_edge(
    llm_client: OpenAIModel,
    tokenizer: Tokenizer,
    graph_storage: NetworkXStorage,
    traverse_strategy: TraverseStrategy,
    text_chunks_storage: JsonKVStorage,
    progress_bar: gr.Progress = None,
    max_concurrent: int = 1000,
) -> dict:
    """
    Traverse the graph

    :param llm_client
    :param tokenizer
    :param graph_storage
    :param traverse_strategy
    :param text_chunks_storage
    :param progress_bar
    :param max_concurrent
    :return: question and answer
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_nodes_and_edges(
        _process_nodes: list,
        _process_edges: list,
    ) -> str:
        prompt = await _construct_rephrasing_prompt(
            _process_nodes, _process_edges, text_chunks_storage, add_context=False
        )
        context = await llm_client.generate_answer(prompt)

        # post-process the context
        if context.startswith("Rephrased Text:"):
            context = context[len("Rephrased Text:") :].strip()
        elif context.startswith("重述文本:"):
            context = context[len("重述文本:") :].strip()

        return context

    async def _process_single_batch(
        _process_batch: tuple, question_type: str = "single"
    ) -> dict:
        async with semaphore:
            context = await _process_nodes_and_edges(
                _process_batch[0],
                _process_batch[1],
            )

            language = "Chinese" if detect_main_language(context) == "zh" else "English"
            pre_length = sum(node["length"] for node in _process_batch[0]) + sum(
                edge[2]["length"] for edge in _process_batch[1]
            )

            if question_type == "single":
                question = await llm_client.generate_answer(
                    QUESTION_GENERATION_PROMPT[language]["SINGLE_TEMPLATE"].format(
                        answer=context
                    )
                )
                if question.startswith("Question:"):
                    question = question[len("Question:") :].strip()
                elif question.startswith("问题："):
                    question = question[len("问题：") :].strip()

                logger.info(
                    "%d nodes and %d edges processed",
                    len(_process_batch[0]),
                    len(_process_batch[1]),
                )
                logger.info("Pre-length: %s", pre_length)
                logger.info("Question: %s", question)
                logger.info("Answer: %s", context)

                return {
                    compute_content_hash(context): {
                        "question": question,
                        "answer": context,
                        "loss": get_average_loss(
                            _process_batch, traverse_strategy.loss_strategy
                        ),
                    }
                }

            content = await llm_client.generate_answer(
                QUESTION_GENERATION_PROMPT[language]["MULTI_TEMPLATE"].format(
                    doc=context
                )
            )
            qas = _post_process_synthetic_data(content)

            if len(qas) == 0:
                print(content)
                logger.error(
                    "Error occurred while processing batch, question or answer is None"
                )
                return {}

            final_results = {}
            logger.info(
                "%d nodes and %d edges processed",
                len(_process_batch[0]),
                len(_process_batch[1]),
            )
            logger.info("Pre-length: %s", pre_length)
            for qa in qas:
                logger.info("Question: %s", qa["question"])
                logger.info("Answer: %s", qa["answer"])
                final_results[compute_content_hash(qa["question"])] = {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "loss": get_average_loss(
                        _process_batch, traverse_strategy.loss_strategy
                    ),
                }
            return final_results

    results = {}
    edges = list(await graph_storage.get_all_edges())
    nodes = list(await graph_storage.get_all_nodes())

    edges, nodes = await _pre_tokenize(graph_storage, tokenizer, edges, nodes)

    processing_batches = await get_batches_with_strategy(
        nodes, edges, graph_storage, traverse_strategy
    )

    for result in tqdm_async(
        asyncio.as_completed(
            [_process_single_batch(batch) for batch in processing_batches]
        ),
        total=len(processing_batches),
        desc="[4/4]Generating QAs",
    ):
        try:
            if progress_bar is not None:
                progress_bar(
                    len(results) / len(processing_batches), desc="[4/4]Generating QAs"
                )
            results.update(await result)
            if progress_bar is not None and len(results) == len(processing_batches):
                progress_bar(1, desc="[4/4]Generating QAs")
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error occurred while generating QA: %s", e)

    return results


async def traverse_graph_atomically(
    llm_client: OpenAIModel,
    tokenizer: Tokenizer,
    graph_storage: NetworkXStorage,
    traverse_strategy: TraverseStrategy,
    text_chunks_storage: JsonKVStorage,
    progress_bar: gr.Progress = None,
    max_concurrent: int = 1000,
) -> dict:
    """
    Traverse the graph atomicly

    :param llm_client
    :param tokenizer
    :param graph_storage
    :param traverse_strategy
    :param text_chunks_storage
    :param progress_bar
    :param max_concurrent
    :return: question and answer
    """
    assert traverse_strategy.qa_form == "atomic"

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _generate_question(node_or_edge: tuple):
        if len(node_or_edge) == 2:
            des = node_or_edge[0] + ": " + node_or_edge[1]["description"]
            loss = node_or_edge[1]["loss"]
        else:
            des = node_or_edge[2]["description"]
            loss = node_or_edge[2]["loss"]

        async with semaphore:
            try:
                language = "Chinese" if detect_main_language(des) == "zh" else "English"

                qa = await llm_client.generate_answer(
                    QUESTION_GENERATION_PROMPT[language]["SINGLE_QA_TEMPLATE"].format(
                        doc=des
                    )
                )

                if "Question:" in qa and "Answer:" in qa:
                    question = qa.split("Question:")[1].split("Answer:")[0].strip()
                    answer = qa.split("Answer:")[1].strip()
                elif "问题：" in qa and "答案：" in qa:
                    question = qa.split("问题：")[1].split("答案：")[0].strip()
                    answer = qa.split("答案：")[1].strip()
                else:
                    return {}

                question = question.strip('"')
                answer = answer.strip('"')

                logger.info("Question: %s", question)
                logger.info("Answer: %s", answer)
                return {
                    compute_content_hash(question): {
                        "question": question,
                        "answer": answer,
                        "loss": loss,
                    }
                }
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Error occurred while generating question: %s", e)
                return {}

    results = {}
    edges = list(await graph_storage.get_all_edges())
    nodes = list(await graph_storage.get_all_nodes())

    edges, nodes = await _pre_tokenize(graph_storage, tokenizer, edges, nodes)

    tasks = []
    for node in nodes:
        if "<SEP>" in node[1]["description"]:
            description_list = node[1]["description"].split("<SEP>")
            for item in description_list:
                tasks.append((node[0], {"description": item, "loss": node[1]["loss"]}))
        else:
            tasks.append((node[0], node[1]))
    for edge in edges:
        if "<SEP>" in edge[2]["description"]:
            description_list = edge[2]["description"].split("<SEP>")
            for item in description_list:
                tasks.append(
                    (edge[0], edge[1], {"description": item, "loss": edge[2]["loss"]})
                )
        else:
            tasks.append((edge[0], edge[1], edge[2]))

    for result in tqdm_async(
        asyncio.as_completed([_generate_question(task) for task in tasks]),
        total=len(tasks),
        desc="[4/4]Generating QAs",
    ):
        try:
            if progress_bar is not None:
                progress_bar(len(results) / len(tasks), desc="[4/4]Generating QAs")
            results.update(await result)
            if progress_bar is not None and len(results) == len(tasks):
                progress_bar(1, desc="[4/4]Generating QAs")
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error occurred while generating QA: %s", e)
    return results


async def traverse_graph_for_multi_hop(
    llm_client: OpenAIModel,
    tokenizer: Tokenizer,
    graph_storage: NetworkXStorage,
    traverse_strategy: TraverseStrategy,
    text_chunks_storage: JsonKVStorage,
    progress_bar: gr.Progress = None,
    max_concurrent: int = 1000,
) -> dict:
    """
    Traverse the graph for multi-hop

    :param llm_client
    :param tokenizer
    :param graph_storage
    :param traverse_strategy
    :param text_chunks_storage
    :param progress_bar
    :param max_concurrent
    :return: question and answer
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    results = {}
    edges = list(await graph_storage.get_all_edges())
    nodes = list(await graph_storage.get_all_nodes())

    edges, nodes = await _pre_tokenize(graph_storage, tokenizer, edges, nodes)

    processing_batches = await get_batches_with_strategy(
        nodes, edges, graph_storage, traverse_strategy
    )

    async def _process_single_batch(_process_batch: tuple) -> dict:
        async with semaphore:
            try:
                language = (
                    "Chinese"
                    if detect_main_language(_process_batch[0][0]["description"]) == "zh"
                    else "English"
                )

                _process_nodes = _process_batch[0]
                _process_edges = _process_batch[1]

                entities = [
                    f"{_process_node['node_id']}: {_process_node['description']}"
                    for _process_node in _process_nodes
                ]

                relations = [
                    f"{_process_edge[0]} -- {_process_edge[1]}: {_process_edge[2]['description']}"
                    for _process_edge in _process_edges
                ]

                entities_str = "\n".join(
                    [f"{index + 1}. {entity}" for index, entity in enumerate(entities)]
                )
                relations_str = "\n".join(
                    [
                        f"{index + 1}. {relation}"
                        for index, relation in enumerate(relations)
                    ]
                )

                prompt = MULTI_HOP_GENERATION_PROMPT[language].format(
                    entities=entities_str, relationships=relations_str
                )

                context = await llm_client.generate_answer(prompt)

                # post-process the context
                if "Question:" in context and "Answer:" in context:
                    question = context.split("Question:")[1].split("Answer:")[0].strip()
                    answer = context.split("Answer:")[1].strip()
                elif "问题：" in context and "答案：" in context:
                    question = context.split("问题：")[1].split("答案：")[0].strip()
                    answer = context.split("答案：")[1].strip()
                else:
                    return {}

                question = question.strip('"')
                answer = answer.strip('"')

                logger.info("Question: %s", question)
                logger.info("Answer: %s", answer)

                return {
                    compute_content_hash(question): {
                        "question": question,
                        "answer": answer,
                        "loss": get_average_loss(
                            _process_batch, traverse_strategy.loss_strategy
                        ),
                    }
                }

            except Exception as e:  # pylint: disable=broad-except
                logger.error("Error occurred while processing batch: %s", e)
                return {}

    async for result in tqdm_async(
        asyncio.as_completed(
            [_process_single_batch(batch) for batch in processing_batches]
        ),
        total=len(processing_batches),
        desc="[4/4]Generating QAs",
    ):
        try:
            if progress_bar is not None:
                progress_bar(
                    len(results) / len(processing_batches), desc="[4/4]Generating QAs"
                )
            results.update(await result)
            if progress_bar is not None and len(results) == len(processing_batches):
                progress_bar(1, desc="[4/4]Generating QAs")
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error occurred while generating QA: %s", e)
    return results
