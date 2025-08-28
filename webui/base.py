from dataclasses import dataclass
from typing import Any

@dataclass
class GraphGenParams:
    """
    GraphGen parameters
    """
    if_trainee_model: bool
    input_file: str
    tokenizer: str
    qa_form: str
    bidirectional: bool
    expand_method: str
    max_extra_edges: int
    max_tokens: int
    max_depth: int
    edge_sampling: str
    isolated_node_strategy: str
    loss_strategy: str
    synthesizer_url: str
    synthesizer_model: str
    trainee_model: str
    api_key: str
    chunk_size: int
    rpm: int
    tpm: int
    quiz_samples: int
    trainee_url: str
    trainee_api_key: str
    token_counter: Any
