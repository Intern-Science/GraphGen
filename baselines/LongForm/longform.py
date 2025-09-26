import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from tqdm.asyncio import tqdm as tqdm_async

from graphgen.models import OpenAIModel
from graphgen.utils import compute_content_hash, create_event_loop

PROMPT_TEMPLATE = """Instruction: X
Output:{doc}

What kind of instruction could this be the answer to?
X:"""


@dataclass
class LongForm:
    llm_client: OpenAIModel = None
    max_concurrent: int = 1000

    def generate(self, docs: List[List[dict]]) -> List[dict]:
        loop = create_event_loop()
        return loop.run_until_complete(self.async_generate(docs))

    async def async_generate(self, docs: List[List[dict]]) -> dict:
        final_results = {}
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_chunk(content: str):
            async with semaphore:
                question = await self.llm_client.generate_answer(content)
                return {
                    compute_content_hash(question): {
                        "question": question,
                        "answer": content,
                    }
                }

        tasks = []
        for doc in docs:
            for chunk in doc:
                tasks.append(process_chunk(chunk["content"]))

        for result in tqdm_async(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Generating using LongForm",
        ):
            try:
                qa = await result
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
        default="cache/data/longform.json",
        type=str,
    )

    args = parser.parse_args()

    load_dotenv()

    llm_client = OpenAIModel(
        model_name=os.getenv("SYNTHESIZER_MODEL"),
        api_key=os.getenv("SYNTHESIZER_API_KEY"),
        base_url=os.getenv("SYNTHESIZER_BASE_URL"),
    )

    longform = LongForm(llm_client=llm_client)

    if args.data_type == "raw":
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
            data = [[chunk] for chunk in data]
    elif args.data_type == "chunked":
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

    results = longform.generate(data)

    # Save results
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
