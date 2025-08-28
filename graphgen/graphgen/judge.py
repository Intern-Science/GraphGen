import os
import argparse
import asyncio
from dotenv import load_dotenv

from .models import NetworkXStorage, JsonKVStorage, OpenAIModel
from .operators import judge_statement

sys_path = os.path.abspath(os.path.dirname(__file__))

load_dotenv()

def calculate_average_loss(graph: NetworkXStorage):
    """
    Calculate the average loss of the graph.

    :param graph: NetworkXStorage
    :return: float
    """
    edges = asyncio.run(graph.get_all_edges())
    total_loss = 0
    for edge in edges:
        total_loss += edge[2]['loss']
    return total_loss / len(edges)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=os.path.join(sys_path, "cache"), help='path to load input graph')
    parser.add_argument('--output', type=str, default='cache/output/new_graph.graphml', help='path to save output')

    args = parser.parse_args()

    llm_client = OpenAIModel(
        model_name=os.getenv("TRAINEE_MODEL"),
        api_key=os.getenv("TRAINEE_API_KEY"),
        base_url=os.getenv("TRAINEE_BASE_URL")
    )

    graph_storage = NetworkXStorage(
        args.input,
        namespace="graph"
    )
    average_loss = calculate_average_loss(graph_storage)
    print(f"Average loss of the graph: {average_loss}")

    rephrase_storage = JsonKVStorage(
        os.path.join(sys_path, "cache"),
        namespace="rephrase"
    )

    new_graph = asyncio.run(judge_statement(llm_client, graph_storage, rephrase_storage, re_judge=True))

    graph_file = asyncio.run(graph_storage.get_graph())

    new_graph.write_nx_graph(graph_file, args.output)

    average_loss = calculate_average_loss(new_graph)
    print(f"Average loss of the graph: {average_loss}")
