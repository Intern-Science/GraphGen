from typing import Any

from graphgen.bases import BaseGenerator
from graphgen.templates import AGGREGATED_GENERATION_PROMPT
from graphgen.utils import compute_content_hash, detect_main_language, logger


class AggregatedGenerator(BaseGenerator):
    """
    Aggregated Generator follows a TWO-STEP process:
    1. rephrase: Rephrase the input nodes and edges into a coherent text that maintains the original meaning.
                 The rephrased text is considered as answer to be used in the next step.
    2. question generation: Generate relevant questions based on the rephrased text.
    """

    @staticmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        """
        Build prompts for REPHRASE.
        :param batch
        :return:
        """
        nodes, edges = batch
        entities_str = "\n".join(
            [
                f"{index + 1}. {node[0]}: {node[1]['description']}"
                for index, node in enumerate(nodes)
            ]
        )
        relations_str = "\n".join(
            [
                f"{index + 1}. {edge[0]} -- {edge[1]}: {edge[2]['description']}"
                for index, edge in enumerate(edges)
            ]
        )
        language = detect_main_language(entities_str + relations_str)

        # TODO: configure add_context
        #     if add_context:
        #         original_ids = [
        #             node["source_id"].split("<SEP>")[0] for node in _process_nodes
        #         ] + [edge[2]["source_id"].split("<SEP>")[0] for edge in _process_edges]
        #         original_ids = list(set(original_ids))
        #         original_text = await text_chunks_storage.get_by_ids(original_ids)
        #         original_text = "\n".join(
        #             [
        #                 f"{index + 1}. {text['content']}"
        #                 for index, text in enumerate(original_text)
        #             ]
        #         )
        prompt = AGGREGATED_GENERATION_PROMPT[language]["ANSWER_REPHRASING"].format(
            entities=entities_str, relationships=relations_str
        )
        return prompt

    @staticmethod
    def parse_rephrased_text(response: str) -> str:
        """
        Parse the rephrased text from the response.
        :param response:
        :return: rephrased text
        """
        if "Rephrased Text:" in response:
            rephrased_text = response.split("Rephrased Text:")[1].strip()
        elif "重述文本:" in response:
            rephrased_text = response.split("重述文本:")[1].strip()
        else:
            rephrased_text = response.strip()
        return rephrased_text.strip('"')

    @staticmethod
    def _build_prompt_for_question_generation(answer: str) -> str:
        """
        Build prompts for QUESTION GENERATION.
        :param answer:
        :return:
        """
        language = detect_main_language(answer)
        prompt = AGGREGATED_GENERATION_PROMPT[language]["QUESTION_GENERATION"].format(
            answer=answer
        )
        return prompt

    @staticmethod
    def parse_response(response: str) -> dict:
        if response.startswith("Question:"):
            question = response[len("Question:") :].strip()
        elif response.startswith("问题："):
            question = response[len("问题：") :].strip()
        else:
            question = response.strip()
        return {
            "question": question,
        }

    async def generate(
        self,
        batch: tuple[
            list[tuple[str, dict]], list[tuple[Any, Any, dict] | tuple[Any, Any, Any]]
        ],
    ) -> dict[str, Any]:
        """
        Generate QAs based on a given batch.
        :param batch
        :return: QA pairs
        """
        result = {}
        rephrasing_prompt = self.build_prompt(batch)
        response = await self.llm_client.generate_answer(rephrasing_prompt)
        context = self.parse_rephrased_text(response)
        question_generation_prompt = self._build_prompt_for_question_generation(context)
        response = await self.llm_client.generate_answer(question_generation_prompt)
        question = self.parse_response(response)["question"]
        logger.debug("Question: %s", question)
        logger.debug("Answer: %s", context)
        qa_pairs = {
            compute_content_hash(question): {
                "question": question,
                "answer": context,
            }
        }
        result.update(qa_pairs)
        return result
