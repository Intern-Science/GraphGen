import argparse
import os
import time
from importlib.resources import files

import yaml
from dotenv import load_dotenv

from graphgen.engine import Context, Engine, collect_ops
from graphgen.graphgen import GraphGen
from graphgen.utils import logger, set_logger

sys_path = os.path.abspath(os.path.dirname(__file__))

load_dotenv()


def set_working_dir(folder):
    os.makedirs(folder, exist_ok=True)


def save_config(config_path, global_config):
    if not os.path.exists(os.path.dirname(config_path)):
        os.makedirs(os.path.dirname(config_path))
    with open(config_path, "w", encoding="utf-8") as config_file:
        yaml.dump(
            global_config, config_file, default_flow_style=False, allow_unicode=True
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        help="Config parameters for GraphGen.",
        default=files("graphgen").joinpath("configs", "aggregated_config.yaml"),
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory for GraphGen.",
        default=sys_path,
        required=True,
        type=str,
    )

    args = parser.parse_args()

    working_dir = args.output_dir

    with open(args.config_file, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    unique_id = int(time.time())

    output_path = os.path.join(working_dir, "data", "graphgen", f"{unique_id}")
    set_working_dir(output_path)

    set_logger(
        os.path.join(output_path, f"{unique_id}.log"),
        if_stream=True,
    )
    logger.info(
        "GraphGen with unique ID %s logging to %s",
        unique_id,
        os.path.join(working_dir, f"{unique_id}.log"),
    )

    graph_gen = GraphGen(unique_id=unique_id, working_dir=working_dir)

    # share context between different steps
    ctx = Context(config=config, graph_gen=graph_gen)
    ops = collect_ops(config, graph_gen)

    # run operations
    Engine(max_workers=config.get("max_workers", 4)).run(ops, ctx)

    save_config(os.path.join(output_path, "config.yaml"), config)
    logger.info("GraphGen completed successfully. Data saved to %s", output_path)


if __name__ == "__main__":
    main()
