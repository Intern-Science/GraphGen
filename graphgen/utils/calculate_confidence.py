import math
from typing import Dict, List

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


def _normalize_yes_no(tokens: List[Token]) -> Dict[str, float]:
    """
    Mapping yes/no synonyms to their probabilities and normalizing.
    For example, given tokens with probabilities:
    - "yes" (0.6)
    - "yeah" (0.2)
    - "no" (0.1)
    - "nope" (0.1)
    The function will return:
    {"yes": 0.8, "no": 0.2}
    Among them, "yes" and "yeah" are synonyms for "yes",
    while "no" and "nope" are synonyms for "no".
    If neither "yes" nor "no" synonyms are present, it returns:
    {"yes": 0.5, "no": 0.5}
    """
    yes_syno = {
        # English yes synonyms
        "yes",
        "yeah",
        "yea",
        "yep",
        "yup",
        "yay",
        "ya",
        "yah",
        "sure",
        "certainly",
        "absolutely",
        "definitely",
        "exactly",
        "indeed",
        "right",
        "correct",
        "true",
        "t",
        "1",
        # Chinese yes synonyms
        "是",
        "对",
        "好的",
        "行",
        "可以",
        "没错",
        "当然",
        "确实",
        "正确",
        "真",
        "对的",
    }
    no_syno = {
        # English no synonyms
        "no",
        "nope",
        "nop",
        "nah",
        "naw",
        "na",
        "negative",
        "never",
        "not",
        "false",
        "f",
        "0",
        # Chinese no synonyms
        "不",
        "不是",
        "没有",
        "错",
        "不对",
        "不行",
        "不能",
        "否",
        "假的",
    }

    yes_prob = 0.0
    no_prob = 0.0
    for tok in tokens:
        t = tok.text.lower().strip()
        if t in yes_syno:
            yes_prob += tok.prob
        elif t in no_syno:
            no_prob += tok.prob

    total = yes_prob + no_prob
    if total == 0:
        return {"yes": 0.5, "no": 0.5}
    return {"yes": yes_prob / total, "no": no_prob / total}


def yes_no_loss_entropy(
    tokens_list: List[List[Token]], ground_truth: List[str]
) -> float:
    """Calculate the loss for yes/no question using entropy."""
    losses = []
    for toks, gt in zip(tokens_list, ground_truth):
        dist = _normalize_yes_no(toks)
        gt = gt.lower()
        assert gt in {"yes", "no"}
        prob_correct = dist[gt]
        losses.append(-math.log(prob_correct))
    return sum(losses) / len(losses)
