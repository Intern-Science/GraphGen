import argparse
import os
import time
from importlib.resources import files

import yaml
from dotenv import load_dotenv

from .graphgen import GraphGen
from .utils import logger, set_logger

sys_path = os.path.abspath(os.path.dirname(__file__))

load_dotenv()


def set_working_dir(folder):
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, "data", "graphgen"), exist_ok=True)
    os.makedirs(os.path.join(folder, "logs"), exist_ok=True)


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
    set_working_dir(working_dir)

    with open(args.config_file, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    output_data_type = config["output_data_type"]
    unique_id = int(time.time())
    set_logger(
        os.path.join(
            working_dir, "logs", f"graphgen_{output_data_type}_{unique_id}.log"
        ),
        if_stream=True,
    )
    logger.info(
        "GraphGen with unique ID %s logging to %s",
        unique_id,
        os.path.join(
            working_dir, "logs", f"{unique_id}_graphgen_{output_data_type}.log"
        ),
    )

    graph_gen = GraphGen(working_dir=working_dir, unique_id=unique_id, config=config)

    graph_gen.insert()

    if config["search"]["enabled"]:
        graph_gen.search()

    # Use pipeline according to the output data type
    if output_data_type in ["atomic", "aggregated", "multi_hop"]:
        if "quiz_and_judge_strategy" in config and config[
            "quiz_and_judge_strategy"
        ].get("enabled", False):
            graph_gen.quiz()
            graph_gen.judge()
        else:
            logger.warning(
                "Quiz and Judge strategy is disabled. Edge sampling falls back to random."
            )
            graph_gen.traverse_strategy.edge_sampling = "random"
        graph_gen.traverse()
    elif output_data_type == "cot":
        graph_gen.generate_reasoning(method_params=config["method_params"])
    else:
        raise ValueError(f"Unsupported output data type: {output_data_type}")

    output_path = os.path.join(working_dir, "data", "graphgen", str(unique_id))
    save_config(os.path.join(output_path, f"config-{unique_id}.yaml"), config)
    logger.info("GraphGen completed successfully. Data saved to %s", output_path)


if __name__ == "__main__":
    main()
