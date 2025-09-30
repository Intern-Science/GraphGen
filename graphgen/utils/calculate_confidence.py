import math
from typing import List

from graphgen.bases.datatypes import Token


def preprocess_tokens(tokens: List[Token]) -> List[Token]:
    """Preprocess tokens for calculating confidence."""
    tokens = [x for x in tokens if x.prob > 0]
    return tokens


def joint_probability(tokens: List[Token]) -> float:
    """Calculate joint probability of a list of tokens."""
    tokens = preprocess_tokens(tokens)
    logprob_sum = sum(x.logprob for x in tokens)
    return math.exp(logprob_sum / len(tokens))


def min_prob(tokens: List[Token]) -> float:
    """Calculate the minimum probability of a list of tokens."""
    tokens = preprocess_tokens(tokens)
    return min(x.prob for x in tokens)


def average_prob(tokens: List[Token]) -> float:
    """Calculate the average probability of a list of tokens."""
    tokens = preprocess_tokens(tokens)
    return sum(x.prob for x in tokens) / len(tokens)


def average_confidence(tokens: List[Token]) -> float:
    """Calculate the average confidence of a list of tokens."""
    tokens = preprocess_tokens(tokens)
    confidence = [x.prob / sum(y.prob for y in x.top_candidates[:5]) for x in tokens]
    return sum(confidence) / len(tokens)


def yes_no_loss(tokens_list: List[List[Token]], ground_truth: List[str]) -> float:
    """Calculate the loss for yes/no question."""
    losses = []
    for i, tokens in enumerate(tokens_list):
        token = tokens[0]
        assert token.text.lower() in ["yes", "no"]
        if token.text == ground_truth[i]:
            losses.append(1 - token.prob)
        else:
            losses.append(token.prob)
    return sum(losses) / len(losses)


def yes_no_loss_entropy(
    tokens_list: List[List[Token]], ground_truth: List[str]
) -> float:
    """Calculate the loss for yes/no question using entropy."""
    losses = []
    for i, tokens in enumerate(tokens_list):
        token = tokens[0]
        assert token.text.lower() in ["yes", "no"]
        if token.text == ground_truth[i]:
            losses.append(-math.log(token.prob))
        else:
            losses.append(-math.log(1 - token.prob))
    return sum(losses) / len(losses)
