import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from tqdm.asyncio import tqdm as tqdm_async

from graphgen.models import OpenAIClient
from graphgen.utils import compute_content_hash, create_event_loop

INSTRUCTION_GENERATION_PROMPT = """The background knowledge is:
{doc}

Please generate ten instruction questions as diverse as possible based on the content of the above article.
These questions can be questions about facts or an understanding and evaluation of relevant content.
Please assume that there is no corresponding article to refer to when asking questions, so do not use demonstrative pronouns such as “this” or “these” in the question.

Please generate questions in the following format:
1. Question: ...
2. Question: ...
"""

READING_COMPREHENSION_PROMPT = """The background knowledge is:
{doc}
Please answer the following question based on the content of the article above:
{question}

Please answer this question as thoroughly as possible, but do not change the key information in the original text, and do not include expressions such as “based on the above article” in the answer.

Please generate the corresponding answer in the following format:
Question: ...
Answer: ...
"""


def _post_process_instructions(content: str) -> list:
    lines = content.split("\n")
    questions = []
    for line in lines:
        if "Question:" in line:
            question = line.split("Question:")[1].strip()
            questions.append(question)
    return questions


def _post_process_answers(content: str) -> tuple:
    if "Question:" in content and "Answer:" in content:
        question = content.split("Question:")[1].split("Answer:")[0].strip()
        answer = content.split("Answer:")[1].strip()
        return question, answer
    return None, None


@dataclass
class SelfQA:
    llm_client: OpenAIClient = None
    max_concurrent: int = 100

    def generate(self, docs: List[List[dict]]) -> List[dict]:
        loop = create_event_loop()
        return loop.run_until_complete(self.async_generate(docs))

    async def async_generate(self, docs: List[List[dict]]) -> dict:
        final_results = {}
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_chunk(content: str):
            async with semaphore:
                prompt = INSTRUCTION_GENERATION_PROMPT.format(doc=content)
                response = await self.llm_client.generate_answer(prompt)
                try:
                    instruction_questions = _post_process_instructions(response)

                    qas = []
                    for qa in tqdm_async(
                        asyncio.as_completed(
                            [
                                self.llm_client.generate_answer(
                                    READING_COMPREHENSION_PROMPT.format(
                                        doc=content, question=question
                                    )
                                )
                                for question in instruction_questions
                            ]
                        ),
                        total=len(instruction_questions),
                        desc="Generating QAs",
                    ):
                        try:
                            question, answer = _post_process_answers(await qa)
                            if question and answer:
                                qas.append(
                                    {
                                        compute_content_hash(question): {
                                            "question": question,
                                            "answer": answer,
                                        }
                                    }
                                )
                        except Exception as e:  # pylint: disable=broad-except
                            print(f"Error: {e}")
                            continue
                    return qas
                except Exception as e:  # pylint: disable=broad-except
                    print(f"Error: {e}")
                    return []

        tasks = []
        for doc in docs:
            for chunk in doc:
                tasks.append(process_chunk(chunk["content"]))

        for result in tqdm_async(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Generating using SelfQA",
        ):
            try:
                qas = await result
                for qa in qas:
                    final_results.update(qa)
            except Exception as e:  # pylint: disable=broad-except
                print(f"Error: {e}")
        return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        help="Raw context jsonl path.",
        default="resources/input_examples/json_demo.json",
        type=str,
    )
    parser.add_argument(
        "--data_type",
        help="Data type of input file. (Raw context or chunked context)",
        choices=["raw", "chunked"],
        default="raw",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        help="Output file path.",
        default="cache/data/self-qa.json",
        type=str,
    )

    args = parser.parse_args()

    load_dotenv()

    llm_client = OpenAIClient(
        model=os.getenv("SYNTHESIZER_MODEL"),
        api_key=os.getenv("SYNTHESIZER_API_KEY"),
        base_url=os.getenv("SYNTHESIZER_BASE_URL"),
    )

    self_qa = SelfQA(llm_client=llm_client)

    if args.data_type == "raw":
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
            data = [[chunk] for chunk in data]
    elif args.data_type == "chunked":
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

    results = self_qa.generate(data)

    # Save results
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
