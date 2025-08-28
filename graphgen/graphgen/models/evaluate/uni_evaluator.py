# https://github.com/maszhongming/UniEval/tree/main

from dataclasses import dataclass, field
from tqdm import tqdm
from graphgen.models.text.text_pair import TextPair


def _add_questions(dimension: str, question: str, answer: str):
    if dimension == "naturalness":
        cur_input = 'question: Is this a natural response in the dialogue? </s> response: ' + answer
    elif dimension == "coherence":
        cur_input = 'question: Is this a coherent response given the dialogue history? </s> response: ' \
                    + answer + ' </s> dialogue history: ' + question
    elif dimension == "understandability":
        cur_input = 'question: Is this an understandable response in the dialogue? </s> response: ' + answer
    else:
        raise NotImplementedError(
            'The input format for this dimension is still undefined. Please customize it first.')
    return cur_input

@dataclass
class UniEvaluator:
    model_name: str = "MingZhong/unieval-sum"
    dimensions: list = field(default_factory=lambda: ['naturalness', 'coherence', 'understandability'])
    max_length: int = 2560
    results: dict = None

    def __post_init__(self):
        import torch
        self.num_gpus = torch.cuda.device_count()
        self.results = {}

    @staticmethod
    def process_chunk(rank, pairs, model_name, max_length, dimension, return_dict):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        device = f'cuda:{rank}'
        torch.cuda.set_device(rank)

        rank_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        rank_model.to(device)
        rank_model.eval()

        softmax = torch.nn.Softmax(dim=1)

        pos_id = tokenizer("Yes")["input_ids"][0]
        neg_id = tokenizer("No")["input_ids"][0]

        results = []
        with torch.no_grad():
            for pair in tqdm(pairs):
                text = _add_questions(dimension, pair.question, pair.answer)

                tgt = "No"

                encoded_src = tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                )
                encoded_tgt = tokenizer(
                    tgt,
                    max_length=max_length,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                )

                src_tokens = encoded_src['input_ids'].to(device)
                src_mask = encoded_src['attention_mask'].to(device)

                tgt_tokens = encoded_tgt['input_ids'].to(device)[:, 0].unsqueeze(-1)

                output = rank_model(
                    input_ids=src_tokens,
                    attention_mask=src_mask,
                    labels=tgt_tokens,
                    use_cache = False
                )

                logits = output.logits.view(-1, rank_model.config.vocab_size)

                pos_score = softmax(logits)[:, pos_id]  # Yes
                neg_score = softmax(logits)[:, neg_id]
                score = pos_score / (pos_score + neg_score)

                results.append(score.item())

        return_dict[rank] = results

    def evaluate(self, pairs: list[TextPair]) -> list[dict]:
        import torch.multiprocessing as mp
        final_results = []
        for dimension in self.dimensions:
            chunk_size = len(pairs) // self.num_gpus
            chunks = []
            for i in range(self.num_gpus):
                start = i * chunk_size
                end = start + chunk_size
                if i == self.num_gpus - 1:
                    end = len(pairs)
                chunks.append(pairs[start:end])

            # multi-process
            manager = mp.Manager()
            return_dict = manager.dict()
            processes = []

            for rank, chunk in enumerate(chunks):
                p = mp.Process(
                    target=self.process_chunk,
                    args=(rank, chunk, self.model_name, self.max_length, dimension, return_dict)
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            # 合并结果
            results = []
            for rank in range(len(chunks)):
                results.extend(return_dict[rank])

            for p in processes:
                if p.is_alive():
                    p.terminate()
                    p.join()

            final_results.append({
                dimension: results
            })
        return final_results

    def get_average_score(self, pairs: list[TextPair]) -> dict:
        """
        Get the average score of a batch of texts.
        """
        results = self.evaluate(pairs)
        final_results = {}
        for result in results:
            for key, value in result.items():
                final_results[key] = sum(value) / len(value)
                self.results[key] = value
        return final_results

    def get_min_max_score(self, pairs: list[TextPair]) -> dict:
        """
        Get the min and max score of a batch of texts.
        """
        if self.results is None:
            self.get_average_score(pairs)
        final_results = {}
        for key, value in self.results.items():
            final_results[key] = min(value), max(value)
        return final_results
