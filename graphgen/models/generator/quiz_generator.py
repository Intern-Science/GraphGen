from typing import Any

from graphgen.bases import BaseGenerator
from graphgen.templates import DESCRIPTION_REPHRASING_PROMPT
from graphgen.utils import detect_main_language, logger


class QuizGenerator(BaseGenerator):
    """
    Quiz Generator rephrases given descriptions to create quiz questions.
    """

    @staticmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        """
        Build prompt for rephrasing the description.
        :param batch: A tuple containing (nodes, edges) where nodes/edges
                      contain description information
        :return: Prompt string
        """
        # Extract description from batch
        # For quiz generator, we expect a special format where
        # the description is passed as the first node's description
        nodes, edges = batch
        if nodes:
            description = nodes[0][1].get("description", "")
            template_type = nodes[0][1].get("template_type", "TEMPLATE")
        elif edges:
            description = edges[0][2].get("description", "")
            template_type = edges[0][2].get("template_type", "TEMPLATE")
        else:
            raise ValueError("Batch must contain at least one node or edge with description")

        return QuizGenerator.build_prompt_for_description(description, template_type)

    @staticmethod
    def build_prompt_for_description(description: str, template_type: str = "TEMPLATE") -> str:
        """
        Build prompt for rephrasing a single description.
        :param description: The description to rephrase
        :param template_type: Either "TEMPLATE" (same meaning) or "ANTI_TEMPLATE" (opposite meaning)
        :return: Prompt string
        """
        language = detect_main_language(description)
        prompt = DESCRIPTION_REPHRASING_PROMPT[language][template_type].format(
            input_sentence=description
        )
        return prompt

    @staticmethod
    def parse_rephrased_text(response: str) -> str:
        """
        Parse the rephrased text from the response.
        :param response:
        :return:
        """
        rephrased_text = response.strip().strip('"')
        logger.debug("Rephrased Text: %s", rephrased_text)
        return rephrased_text

    @staticmethod
    def parse_response(response: str) -> Any:
        """
        Parse the LLM response. For quiz generator, this returns the rephrased text.
        :param response: LLM response
        :return: Rephrased text
        """
        return QuizGenerator.parse_rephrased_text(response)
