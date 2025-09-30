import json
import os
import sys

import pandas as pd

# pylint: disable=wrong-import-position
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from graphgen.models import Tokenizer


def count_tokens(file, tokenizer_name, data_frame):
    if not file or not os.path.exists(file):
        return data_frame

    if file.endswith(".jsonl"):
        with open(file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    elif file.endswith(".json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            data = [item for sublist in data for item in sublist]
    elif file.endswith(".txt"):
        with open(file, "r", encoding="utf-8") as f:
            data = f.read()
            chunks = [data[i : i + 512] for i in range(0, len(data), 512)]
            data = [{"content": chunk} for chunk in chunks]
    elif file.endswith(".csv"):
        df = pd.read_csv(file)
        if "content" in df.columns:
            data = df["content"].tolist()
        else:
            data = df.iloc[:, 0].tolist()
    else:
        raise ValueError(f"Unsupported file type: {file}")

    tokenizer = Tokenizer(tokenizer_name)

    # Count tokens
    token_count = 0

    for item in data:
        if isinstance(item, dict):
            content = item.get("content", "")
        else:
            content = item
        token_count += len(tokenizer.encode(content))

    _update_data = [[str(token_count), str(token_count * 50), "N/A"]]

    try:
        new_df = pd.DataFrame(_update_data, columns=data_frame.columns)
        data_frame = new_df

    except Exception as e:  # pylint: disable=broad-except
        print("[ERROR] DataFrame操作异常:", str(e))

    return data_frame
