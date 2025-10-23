from typing import Any

from graphgen.bases import BaseGenerator
from graphgen.templates import MULTI_HOP_GENERATION_PROMPT
from graphgen.utils import compute_content_hash, detect_main_language, logger


class MultiHopGenerator(BaseGenerator):
    @staticmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
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
        prompt = MULTI_HOP_GENERATION_PROMPT[language].format(
            entities=entities_str, relationships=relationships_str
        )
        return prompt

    @staticmethod
    def parse_response(response: str) -> dict:
        if "Question:" in response and "Answer:" in response:
            question = response.split("Question:")[1].split("Answer:")[0].strip()
            answer = response.split("Answer:")[1].strip()
        elif "问题：" in response and "答案：" in response:
            question = response.split("问题：")[1].split("答案：")[0].strip()
            answer = response.split("答案：")[1].strip()
        else:
            logger.warning("Failed to parse response: %s", response)
            return {}
        question = question.strip('"')
        answer = answer.strip('"')
        logger.debug("Question: %s", question)
        logger.debug("Answer: %s", answer)
        return {
            compute_content_hash(question): {
                "question": question,
                "answer": answer,
            }
        }
