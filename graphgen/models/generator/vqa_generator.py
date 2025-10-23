from typing import Any

from graphgen.bases import BaseGenerator
from graphgen.templates import VQA_GENERATION_PROMPT
from graphgen.utils import compute_content_hash, detect_main_language, logger


class VQAGenerator(BaseGenerator):
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
        prompt = VQA_GENERATION_PROMPT[language].format(
            entities=entities_str, relationships=relationships_str
        )
        return prompt

    @staticmethod
    def parse_response(response: str) -> Any:
        """
        Parse the LLM response and return the generated QAs
        :param response
        :return: QA pairs
        """
        qa_pairs = {}
        qa_list = response.strip().split("\n\n")
        for qa in qa_list:
            if "Question:" in qa and "Answer:" in qa:
                question = qa.split("Question:")[1].split("Answer:")[0].strip()
                answer = qa.split("Answer:")[1].strip()
            elif "问题：" in qa and "答案：" in qa:
                question = qa.split("问题：")[1].split("答案：")[0].strip()
                answer = qa.split("答案：")[1].strip()
            else:
                logger.error("Failed to parse QA pair: %s", qa)
                continue
            question = question.strip('"')
            answer = answer.strip('"')
            logger.debug("Question: %s", question)
            logger.debug("Answer: %s", answer)
            qa_pairs[compute_content_hash(question)] = {
                "question": question,
                "answer": answer,
            }
        return qa_pairs

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
        qa_pairs = self.parse_response(response)  # generate one or more QA pairs
        nodes, _ = batch
        for node in nodes:
            node_data = node[1]
            if "images" in node_data and node_data["images"]:
                img_path = node_data["images"]["img_path"]
                for qa in qa_pairs.values():
                    qa["img_path"] = img_path
        result.update(qa_pairs)
        return result

    @staticmethod
    def format_generation_results(
        results: list[dict], output_data_format: str
    ) -> list[dict[str, Any]]:
        if output_data_format == "Alpaca":
            results = [
                {
                    "instruction": v["question"],
                    "input": "",
                    "output": v["answer"],
                    "image": v.get("img_path", ""),
                }
                for item in results
                for k, v in item.items()
            ]
        elif output_data_format == "Sharegpt":
            results = [
                {
                    "conversations": [
                        {
                            "from": "human",
                            "value": [
                                {"text": v["question"], "image": v.get("img_path", "")}
                            ],
                        },
                        {"from": "gpt", "value": v["answer"]},
                    ]
                }
                for item in results
                for k, v in item.items()
            ]
        elif output_data_format == "ChatML":
            results = [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"text": v["question"], "image": v.get("img_path", "")}
                            ],
                        },
                        {"role": "assistant", "content": v["answer"]},
                    ]
                }
                for item in results
                for k, v in item.items()
            ]
        else:
            raise ValueError(f"Unknown output data format: {output_data_format}")
        return results
