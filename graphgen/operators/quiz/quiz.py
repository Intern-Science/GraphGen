from collections.abc import Iterable

import pandas as pd

from graphgen.bases import BaseGraphStorage, BaseKVStorage, BaseLLMWrapper
from graphgen.common import init_llm, init_storage
from graphgen.models import QuizGenerator
from graphgen.utils import compute_content_hash, logger, run_concurrent


class QuizService:
    def __init__(self, working_dir: str = "cache", quiz_samples: int = 1):
        self.quiz_samples = quiz_samples
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.graph_storage: BaseGraphStorage = init_storage(
            backend="networkx", working_dir=working_dir, namespace="graph"
        )
        # { _description_id: { "description": str, "quizzes": List[Tuple[str, str]] } }
        self.quiz_storage: BaseKVStorage = init_storage(
            backend="json_kv", working_dir=working_dir, namespace="quiz"
        )
        self.generator = QuizGenerator(self.llm_client)

        self.concurrency_limit = 20

    def __call__(self, batch: pd.DataFrame) -> Iterable[pd.DataFrame]:
        # this operator does not consume any batch data
        # but for compatibility we keep the interface
        _ = batch.to_dict(orient="records")
        self.graph_storage.reload()
        yield from self.quiz()

    async def _process_single_quiz(self, item: str) -> dict | None:
        # if quiz in quiz_storage exists already, directly get it
        _description_id = compute_content_hash(item)
        if self.quiz_storage.get_by_id(_description_id):
            return None

        tasks = []
        for i in range(self.quiz_samples):
            if i > 0:
                tasks.append((item, "TEMPLATE", "yes"))
            tasks.append((item, "ANTI_TEMPLATE", "no"))
        try:
            quizzes = []
            for description, template_type, gt in tasks:
                prompt = self.generator.build_prompt_for_description(
                    description, template_type
                )
                new_description = await self.llm_client.generate_answer(
                    prompt, temperature=1
                )
                rephrased_text = self.generator.parse_rephrased_text(new_description)
                quizzes.append((rephrased_text, gt))
            return {
                "_description_id": _description_id,
                "description": item,
                "quizzes": quizzes,
            }
        except Exception as e:
            logger.error("Error when quizzing description %s: %s", item, e)
            return None

    def quiz(self) -> Iterable[pd.DataFrame]:
        """
        Get all nodes and edges and quiz their descriptions using QuizGenerator.
        """
        edges = self.graph_storage.get_all_edges()
        nodes = self.graph_storage.get_all_nodes()

        items = []

        for edge in edges:
            edge_data = edge[2]
            description = edge_data["description"]
            items.append(description)

        for node in nodes:
            node_data = node[1]
            description = node_data["description"]
            items.append(description)

        logger.info("Total descriptions to quiz: %d", len(items))

        for i in range(0, len(items), self.concurrency_limit):
            batch_items = items[i : i + self.concurrency_limit]
            batch_results = run_concurrent(
                self._process_single_quiz,
                batch_items,
                desc=f"Quizzing descriptions ({i} / {i + len(batch_items)})",
                unit="description",
            )

            final_results = []
            for new_result in batch_results:
                if new_result:
                    self.quiz_storage.upsert(
                        {
                            new_result["_description_id"]: {
                                "description": new_result["description"],
                                "quizzes": new_result["quizzes"],
                            }
                        }
                    )
                    final_results.append(new_result)
            self.quiz_storage.index_done_callback()
            yield pd.DataFrame(final_results)
