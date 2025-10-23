from typing import Any

from graphgen.bases import BaseGenerator
from graphgen.templates import COT_GENERATION_PROMPT
from graphgen.utils import compute_content_hash, detect_main_language, logger


class CoTGenerator(BaseGenerator):
    @staticmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        """
        Build prompts for COT Template Design.
        :param batch:
        :return:
        """
        nodes, edges = batch
        entities_str = "\n".join(
            [
                f"{index + 1}. {node[0]}: {node[1]['description']}"
                for index, node in enumerate(nodes)
            ]
        )
        relationships_str = "\n".join(
            [
                f"{index + 1}. {edge[0]} -- {edge[1]}: {edge[2]['description']}"
                for index, edge in enumerate(edges)
            ]
        )
        language = detect_main_language(entities_str + relationships_str)
        prompt = COT_GENERATION_PROMPT[language]["COT_TEMPLATE_DESIGN"].format(
            entities=entities_str, relationships=relationships_str
        )
        return prompt

    @staticmethod
    def build_prompt_for_cot_generation(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]],
        question: str,
        reasoning_path: str,
    ) -> str:
        """
        Build prompts for COT Generation.
        """
        nodes, edges = batch
        entities_str = "\n".join(
            [
                f"{index + 1}. {node[0]}: {node[1]['description']}"
                for index, node in enumerate(nodes)
            ]
        )
        relationships_str = "\n".join(
            [
                f"{index + 1}. {edge[0]} -- {edge[1]}: {edge[2]['description']}"
                for index, edge in enumerate(edges)
            ]
        )
        language = detect_main_language(entities_str + relationships_str)
        prompt = COT_GENERATION_PROMPT[language]["COT_GENERATION"].format(
            entities=entities_str,
            relationships=relationships_str,
            question=question,
            reasoning_template=reasoning_path,
        )
        return prompt

    @staticmethod
    def parse_response(response: str) -> dict:
        if "Question:" in response and "Reasoning-Path Design:" in response:
            question = (
                response.split("Question:")[1]
                .split("Reasoning-Path Design:")[0]
                .strip()
            )
            reasoning_path = response.split("Reasoning-Path Design:")[1].strip()
        elif "问题：" in response and "推理路径设计：" in response:
            question = response.split("问题：")[1].split("推理路径设计：")[0].strip()
            reasoning_path = response.split("推理路径设计：")[1].strip()
        else:
            logger.warning("Failed to parse CoT template: %s", response)
            return {}

        question = question.strip('"')
        reasoning_path = reasoning_path.strip('"')
        logger.debug("CoT Question: %s", question)
        logger.debug("CoT Reasoning Path: %s", reasoning_path)
        return {
            "question": question,
            "reasoning_path": reasoning_path,
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
        prompt = self.build_prompt(batch)
        response = await self.llm_client.generate_answer(prompt)
        response = self.parse_response(response)
        question, reasoning_path = response["question"], response["reasoning_path"]
        prompt = self.build_prompt_for_cot_generation(batch, question, reasoning_path)
        cot_answer = await self.llm_client.generate_answer(prompt)
        logger.debug("CoT Answer: %s", cot_answer)
        qa_pairs = {
            compute_content_hash(question): {
                "question": question,
                "answer": cot_answer,
                "reasoning_path": reasoning_path,
            }
        }
        result.update(qa_pairs)
        return result
