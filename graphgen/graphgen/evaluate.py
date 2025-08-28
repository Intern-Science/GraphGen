"""Evaluate the quality of the generated text using various metrics"""

import os
import json
import argparse
import pandas as pd
from dotenv import load_dotenv
from .models import LengthEvaluator, MTLDEvaluator, RewardEvaluator, TextPair, UniEvaluator
from .utils import logger, set_logger

sys_path = os.path.abspath(os.path.dirname(__file__))
set_logger(os.path.join(sys_path, "cache", "logs", "evaluate.log"))

load_dotenv()

def evaluate_length(corpus, tokenizer_name):
    length_evaluator = LengthEvaluator(
        tokenizer_name=tokenizer_name
    )
    logger.info("Length evaluator loaded")
    scores = length_evaluator.get_average_score(corpus)
    logger.info("Length scores: %s", scores)
    return scores

def evaluate_mtld(corpus):
    mtld_evaluator = MTLDEvaluator()
    logger.info("MTLD evaluator loaded")
    scores = mtld_evaluator.get_average_score(corpus)
    logger.info("MTLD scores: %s", scores)
    min_max_scores = mtld_evaluator.get_min_max_score(corpus)
    logger.info("MTLD min max scores: %s", min_max_scores)
    return scores, min_max_scores

def evaluate_reward(corpus, reward_model_names):
    scores = []
    for reward_name in reward_model_names:
        reward_evaluator = RewardEvaluator(
            reward_name=reward_name
        )
        logger.info("Loaded reward model: %s", reward_name)
        average_score = reward_evaluator.get_average_score(corpus)
        logger.info("%s scores: %s", reward_name, average_score)
        min_max_scores = reward_evaluator.get_min_max_score(corpus)
        logger.info("%s min max scores: %s", reward_name, min_max_scores)
        scores.append({
            'reward_name': reward_name.split('/')[-1],
            'score': average_score,
            'min_max_scores': min_max_scores
        })
        del reward_evaluator
        clean_gpu_cache()
    return scores

def evaluate_uni(corpus, uni_model_name):
    uni_evaluator = UniEvaluator(
        model_name=uni_model_name
    )
    logger.info("Uni evaluator loaded with model %s", uni_model_name)
    uni_scores = uni_evaluator.get_average_score(corpus)
    for key, value in uni_scores.items():
        logger.info("Uni %s scores: %s", key, value)
    min_max_scores = uni_evaluator.get_min_max_score(corpus)
    for key, value in min_max_scores.items():
        logger.info("Uni %s min max scores: %s", key, value)
    del uni_evaluator
    clean_gpu_cache()
    return (uni_scores['naturalness'], uni_scores['coherence'], uni_scores['understandability'],
            min_max_scores['naturalness'], min_max_scores['coherence'], min_max_scores['understandability'])


def clean_gpu_cache():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    import torch.multiprocessing as mp
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder', type=str, default='cache/data', help='folder to load data')
    parser.add_argument('--output', type=str, default='cache/output', help='path to save output')

    parser.add_argument('--tokenizer', type=str, default='cl100k_base', help='tokenizer name')
    parser.add_argument('--reward', type=str, default='OpenAssistant/reward-model-deberta-v3-large-v2',
                        help='Comma-separated list of reward models')
    parser.add_argument('--uni', type=str, default='MingZhong/unieval-sum', help='uni model name')

    args = parser.parse_args()

    if not os.path.exists(args.folder):
        raise ValueError(f"Folder {args.folder} does not exist")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    reward_models = args.reward.split(',')


    results = []

    logger.info("Data loaded from %s", args.folder)
    mp.set_start_method('spawn')

    for file in os.listdir(args.folder):
        if file.endswith('.json'):
            logger.info("Processing %s", file)
            with open(os.path.join(args.folder, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
            data = [TextPair(
                question=data[key]['question'],
                answer=data[key]['answer']
            ) for key in data]

            length_scores = evaluate_length(data, args.tokenizer)
            mtld_scores, min_max_mtld_scores = evaluate_mtld(data)
            reward_scores = evaluate_reward(data, reward_models)
            uni_naturalness_scores, uni_coherence_scores, uni_understandability_scores, \
            min_max_uni_naturalness_scores, min_max_uni_coherence_scores, min_max_uni_understandability_scores \
                = evaluate_uni(data, args.uni)

            result = {
                'file': file,
                'number': len(data),
                'length': length_scores,
                'mtld': mtld_scores,
                'mtld_min_max': min_max_mtld_scores,
                'uni_naturalness': uni_naturalness_scores,
                'uni_coherence': uni_coherence_scores,
                'uni_understandability': uni_understandability_scores,
                'uni_naturalness_min_max': min_max_uni_naturalness_scores,
                'uni_coherence_min_max': min_max_uni_coherence_scores,
                'uni_understandability_min_max': min_max_uni_understandability_scores
            }
            for reward_score in reward_scores:
                result[reward_score['reward_name']] = reward_score['score']
                result[f"{reward_score['reward_name']}_min_max"] = reward_score['min_max_scores']

            results.append(result)

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(args.output, 'evaluation.csv'), index=False)
