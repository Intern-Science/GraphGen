from dataclasses import dataclass
from typing import Any


@dataclass
class WebuiParams:
    """
    GraphGen parameters
    """

    if_trainee_model: bool
    upload_file: Any  # gr.File
    tokenizer: str
    synthesizer_model: str
    synthesizer_url: str
    trainee_model: str
    trainee_url: str
    api_key: str
    trainee_api_key: str
    chunk_size: int
    chunk_overlap: int
    quiz_samples: int
    partition_method: str
    dfs_max_units: int
    bfs_max_units: int
    leiden_max_size: int
    leiden_use_lcc: bool
    leiden_random_seed: int
    ece_max_units: int
    ece_min_units: int
    ece_max_tokens: int
    ece_unit_sampling: str
    mode: str
    data_format: str
    rpm: int
    tpm: int
    token_counter: Any
