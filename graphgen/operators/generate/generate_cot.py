import asyncio
from typing import Dict, List, Tuple

from tqdm.asyncio import tqdm as tqdm_async

from graphgen.models import CommunityDetector, NetworkXStorage, OpenAIClient
from graphgen.templates import COT_GENERATION_PROMPT, COT_TEMPLATE_DESIGN_PROMPT
from graphgen.utils import compute_content_hash, detect_main_language


async def generate_cot(
    graph_storage: NetworkXStorage,
    synthesizer_llm_client: OpenAIClient,
    method_params: Dict = None,
):
    method = method_params.get("method", "leiden")
    detector = CommunityDetector(
        graph_storage=graph_storage, method=method, method_params=method_params
    )

    results = await detector.detect_communities()

    # Convert results to a format suitable for summarization
    communities = {}
    for node, community_id in results.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)

    if not communities:
        return {}

    semaphore = asyncio.Semaphore(value=1000)

    async def _generate_from_single_community(
        c_id: int, nodes: List[str]
    ) -> Tuple[int, Tuple[str, str, str]]:
        """Summarize a single community."""
        async with semaphore:
            entities: List[str] = []
            relationships: List[str] = []

            for n in nodes:
                node_data = await graph_storage.get_node(n)
                if node_data is not None:
                    entities.append(f"({n}: {node_data.get('description')})")

                edges = await graph_storage.get_node_edges(n)
                for edge in edges:
                    target = edge[1]
                    if target in nodes:
                        edge_data = await graph_storage.get_edge(n, target)
                        relationships.append(
                            f"({n}) - [{edge_data['description']}] -> ({target})"
                        )

            entities_str = "\n".join(entities)
            relationships_str = "\n".join(relationships)

            language = (
                "English"
                if detect_main_language(entities_str + relationships_str) == "en"
                else "Chinese"
            )

            prompt = COT_TEMPLATE_DESIGN_PROMPT[language]["TEMPLATE"].format(
                entities=entities_str,
                relationships=relationships_str,
            )

            cot_template = await synthesizer_llm_client.generate_answer(prompt)

            if "问题：" in cot_template and "推理路径设计：" in cot_template:
                question = cot_template.split("问题：")[1].split("推理路径设计：")[0].strip()
                reasoning_path = cot_template.split("推理路径设计：")[1].strip()
            elif (
                "Question:" in cot_template and "Reasoning-Path Design:" in cot_template
            ):
                question = (
                    cot_template.split("Question:")[1]
                    .split("Reasoning-Path Design:")[0]
                    .strip()
                )
                reasoning_path = cot_template.split("Reasoning-Path Design:")[1].strip()
            else:
                raise ValueError("COT template format is incorrect.")

            prompt = COT_GENERATION_PROMPT[language]["TEMPLATE"].format(
                entities=entities_str,
                relationships=relationships_str,
                question=question,
                reasoning_template=reasoning_path,
            )

            cot_answer = await synthesizer_llm_client.generate_answer(prompt)

            return c_id, (question, reasoning_path, cot_answer)

    cid_nodes = list(communities.items())

    results: Dict = {}
    async for coro in tqdm_async(
        asyncio.as_completed(
            [_generate_from_single_community(cid, nodes) for cid, nodes in cid_nodes]
        ),
        total=len(cid_nodes),
        desc="[Generating COT] Generating CoT data from communities",
        unit="community",
    ):
        cid, (q, r, a) = await coro
        results[compute_content_hash(q)] = {
            "question": q,
            "reasoning_path": r,
            "answer": a,
        }

    return results
