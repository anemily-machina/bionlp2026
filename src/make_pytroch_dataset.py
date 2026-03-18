"""

python src/make_pytroch_dataset.py EleutherAI/pythia-70m

python src/make_pytroch_dataset.py FacebookAI/roberta-base

python src/make_pytroch_dataset.py distilbert/distilbert-base-uncased

BAAI/bge-m3
"""

from utils import load_json

import argparse
import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


TOKENIZER_PARSE_FOLDER = "./data/tokenized_examples"


class SpanOnlyBioNLP(Dataset):

    def __init__(self, ai_name):

        super().__init__()

        tokenized_folder = os.path.join(TOKENIZER_PARSE_FOLDER, ai_name)

        files = os.listdir(tokenized_folder)
        files = [os.path.join(tokenized_folder, f) for f in files]

        class_sizes = {0: 0, 1: 0}

        examples = []
        for f in files:

            entry = load_json(f)
            entry.pop("sentence")

            token_labels = entry.pop("token_labels")
            input_ids = entry.pop("input_ids")

            entry_labels = []
            for labels in token_labels:

                # span only, igore label categaory
                # no span labels means it is never in a span
                if len(labels) == 0:
                    label = 0
                    class_sizes[0] += 1
                # for special tokens the span label should only be -100
                elif -100 in labels:
                    label = -100

                # otherwise it's in the span of -some- category
                else:
                    label = 1
                    class_sizes[1] += 1

                entry_labels.append(label)

            example = {}
            example["input_ids"] = torch.tensor(input_ids)
            example["labels"] = torch.tensor(entry_labels)

            examples.append(example)

        self.exmaples = examples
        self.class_sizes = class_sizes

    def balanced_weights(self):
        total = sum(self.class_sizes.values())
        weights = [v / total for v in self.class_sizes.values()]
        return weights

    def __len__(self):
        return len(self.exmaples)

    def __getitem__(self, index):

        return self.exmaples[index]


def span_only_collate(batch):

    input_ids = [e["input_ids"] for e in batch]
    input_ids = pad_sequence(input_ids)

    labels = [e["labels"] for e in batch]
    labels = pad_sequence(input_ids, padding_value=-100)

    batch = {"input_ids": input_ids, "labels": labels}

    return batch


def make_dataset(ai_name, span_only=False):

    if span_only:
        dataset = SpanOnlyBioNLP(ai_name)

        return dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ai_name")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    dataset = make_dataset(args.ai_name, span_only=True)

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=span_only_collate,
        # num_workers=0,
        # prefetch_factor=1,
    )

    for b in dataloader:

        print(b)

    exit()
