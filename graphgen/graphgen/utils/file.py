import json


def read_file(input_file: str) -> list:
    """
    Read data from a file based on the specified data type.
    :param input_file
    :return:
    """

    if input_file.endswith(".jsonl"):
        with open(input_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    elif input_file.endswith(".json"):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif input_file.endswith(".txt"):
        with open(input_file, "r", encoding="utf-8") as f:
            data = [line.strip() for line in f if line.strip()]
            data = [{"content": line} for line in data]
    else:
        raise ValueError(f"Unsupported file format: {input_file}")

    return data
