import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from typing import List

import networkx as nx
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as tqdm_async

from graphgen.models import NetworkXStorage, OpenAIModel, Tokenizer
from graphgen.utils import create_event_loop

QA_GENERATION_PROMPT = """
Create an agriculture examination question for advanced agricultural students that tests the relationship between {src} and {tgt}. The relationship is: {path}. The question should:
    1. Be in multiple choice format (4 options)
    2. Require agriculture reasoning along the relationship
    3. Include a brief farm or field scenario
    4. Not directly mention the relationship in the question stem
    5. Have one clearly correct answer
Format:
    <Question>
        [Farm or Field Scenario]
    </Question>
    <Options>
        A. [Option]
        B. [Option]
        C. [Option]
        D. [Option]
    </Options>
    <Answer>:
        [Correct Option Letter]
    </Answer>
"""


def _post_process(text: str) -> dict:
    try:
        q = text.split("<Question>")[1].split("</Question>")[0].strip()
        opts = text.split("<Options>")[1].split("</Options>")[0].strip().splitlines()
        opts = [o.strip() for o in opts if o.strip()]
        ans = text.split("<Answer>:")[1].strip()[0].upper()
        return {
            "question": q,
            "options": opts,
            "answer": ord(ans) - ord("A"),
            "raw": text,
        }
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error in post-processing: {e}")
        return {}


@dataclass
class BDS:
    llm_client: OpenAIModel = None
    max_concurrent: int = 1000

    def generate(self, tasks: List[dict]) -> List[dict]:
        loop = create_event_loop()
        return loop.run_until_complete(self._async_generate(tasks))

    async def _async_generate(self, tasks: List[dict]) -> List[dict]:
        sem = asyncio.Semaphore(self.max_concurrent)

        async def job(item):
            async with sem:
                path_str = " -> ".join([f"({h},{r},{t})" for h, r, t in item["path"]])
                prompt = QA_GENERATION_PROMPT.format(
                    src=item["src"], tgt=item["tgt"], path=path_str
                )
                resp = await self.llm_client.generate_answer(prompt)
                return _post_process(resp)

        tasks = [job(it) for it in tasks]
        results = []
        for coro in tqdm_async(asyncio.as_completed(tasks), total=len(tasks)):
            try:
                if r := await coro:
                    results.append(r)
            except Exception as e:  # pylint: disable=broad-except
                print("Error:", e)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        help="GraphML input file path.",
        default="resources/input_examples/graphml_demo.graphml",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        help="Output file path.",
        default="cache/data/bds_qa.jsonl",
        type=str,
    )
    args = parser.parse_args()

    load_dotenv()

    tokenizer_instance: Tokenizer = Tokenizer(
        model_name=os.getenv("TOKENIZER_MODEL", "cl100k_base")
    )
    llm_client = OpenAIModel(
        model_name=os.getenv("SYNTHESIZER_MODEL"),
        api_key=os.getenv("SYNTHESIZER_API_KEY"),
        base_url=os.getenv("SYNTHESIZER_BASE_URL"),
        tokenizer_instance=tokenizer_instance,
    )
    bds = BDS(llm_client=llm_client)

    graph = NetworkXStorage.load_nx_graph(args.input_file)

    MAX_PATH = 20000
    all_paths = []

    G = graph.to_directed() if not graph.is_directed() else graph
    print(G)

    source_nodes = [n for n in G.nodes if G.out_degree(n) > 0][:1000]

    for src in source_nodes:
        for path in nx.all_simple_paths(G, source=src, target=list(G.nodes), cutoff=3):
            if len(path) == 4:
                all_paths.append(path)
                if len(all_paths) >= MAX_PATH:
                    break
            if len(all_paths) >= MAX_PATH:
                break
        if len(all_paths) >= MAX_PATH:
            break

    print(f"Found {len(all_paths)} 4-node paths")

    items = []
    for path in all_paths:
        path_edges = []
        for i in range(len(path) - 1):
            edge_data = G.get_edge_data(path[i], path[i + 1])
            if edge_data is None:
                edge_data = G.get_edge_data(path[i + 1], path[i])
            if edge_data is None:
                print(f"Warning: No edge data between {path[i]} and {path[i+1]}")
                relation = "related_to"
            else:
                relation = edge_data.get("relation", "related_to")
            path_edges.append((path[i], relation, path[i + 1]))
        items.append({"src": path[0], "tgt": path[-1], "path": path_edges})

    print(f"Prepared {len(items)} items for question generation")

    qa_pairs = bds.generate(items)
    print(f"Generated {len(qa_pairs)} QA pairs")

    # Save results
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=4, ensure_ascii=False)
