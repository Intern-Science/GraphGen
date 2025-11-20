# https://arxiv.org/pdf/2401.14367

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

PROMPT_TEMPLATE = """Instruction: Given the next [document], create a [question] and [answer] pair that are grounded \
in the main point of the document, don't add any additional information that is not in the document. The [question] is \
by an information-seeking user and the [answer] is provided by a helping AI Agent.

[document]: Scrumptious Sweet Co. factory ...

### Response:
[question]: Where was the movie Chitty Chitty Bang Bang filmed?
[answer]: Chitty Chitty Bang Bang was filmed in a variety of locations in England and Germany. Some ...

[document]: The series stars Eugene Levy as Johnny Rose ...

### Response:
[question]: What is the plot of the show Schitt's Creek?
[answer]: The show Schitt's Creek is about a wealthy family who loses their fortune and is forced to rebuild their \
lives in a small town. The show follows the family as they adjust to their new life in the town and learn to \
appreciate the simple things in life.

[document]: 2016's countdown broke several Hottest 100 records ...

### Response:
[question]: What was the most popular song on the 2016 Hottest 100?
[answer]: The most popular song on the 2016 Hottest 100 was "Never Be Like You" by Flume. This was the first time that \
an electronic dance music producer topped the countdown.

[document]: In Greek mythology, Persephone ...

### Response:
[question]: Who is Persephone in Greek mythology and what is her significance?
[answer]: In Greek mythology, Persephone, also known as Kore, is the daughter of Zeus and Demeter. She ...

[document]: {doc}

### Response:"""


def _post_process(content: str) -> tuple:
    if "[question]:" in content and "[answer]:" in content:
        question = content.split("[question]: ")[1].split("[answer]: ")[0]
        answer = content.split("[answer]: ")[1]
        return question, answer
    return None, None


@dataclass
class Genie:
    llm_client: OpenAIClient = None
    max_concurrent: int = 1000

    def generate(self, docs: List[List[dict]]) -> List[dict]:
        loop = create_event_loop()
        return loop.run_until_complete(self.async_generate(docs))

    async def async_generate(self, docs: List[List[dict]]) -> dict:
        final_results = {}
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_chunk(content: str):
            async with semaphore:
                prompt = PROMPT_TEMPLATE.format(doc=content)
                return await self.llm_client.generate_answer(prompt)

        tasks = []
        for doc in docs:
            for chunk in doc:
                tasks.append(process_chunk(chunk["content"]))

        for result in tqdm_async(
            asyncio.as_completed(tasks), total=len(tasks), desc="Generating using Genie"
        ):
            try:
                question, answer = _post_process(await result)
                if question and answer:
                    final_results[compute_content_hash(question)] = {
                        "question": question,
                        "answer": answer,
                    }
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
        default="cache/data/genie.json",
        type=str,
    )

    args = parser.parse_args()

    load_dotenv()

    llm_client = OpenAIClient(
        model=os.getenv("SYNTHESIZER_MODEL"),
        api_key=os.getenv("SYNTHESIZER_API_KEY"),
        base_url=os.getenv("SYNTHESIZER_BASE_URL"),
    )

    genie = Genie(llm_client=llm_client)

    if args.data_type == "raw":
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
            data = [[chunk] for chunk in data]
    elif args.data_type == "chunked":
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

    results = genie.generate(data)

    # Save results
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
