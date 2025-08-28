import html
import json
import os
import re
from typing import Any

from .log import logger


def pack_history_conversations(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


async def handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return {
        "entity_name": entity_name,
        "entity_type": entity_type,
        "description": entity_description,
        "source_id": entity_source_id,
    }


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


async def handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_source_id = chunk_key
    return {
        "src_id": source,
        "tgt_id": target,
        "description": edge_description,
        "source_id": edge_source_id,
    }


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=4, ensure_ascii=False)


def format_generation_results(
    results: dict[str, Any], output_data_format: str
) -> list[dict[str, Any]]:
    if output_data_format == "Alpaca":
        logger.info("Output data format: Alpaca")
        results = [
            {
                "instruction": item["question"],
                "input": "",
                "output": item["answer"],
            }
            for item in list(results.values())
        ]
    elif output_data_format == "Sharegpt":
        logger.info("Output data format: Sharegpt")
        results = [
            {
                "conversations": [
                    {"from": "human", "value": item["question"]},
                    {"from": "gpt", "value": item["answer"]},
                ]
            }
            for item in list(results.values())
        ]
    elif output_data_format == "ChatML":
        logger.info("Output data format: ChatML")
        results = [
            {
                "messages": [
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["answer"]},
                ]
            }
            for item in list(results.values())
        ]
    else:
        raise ValueError(f"Unknown output data format: {output_data_format}")
    return results
