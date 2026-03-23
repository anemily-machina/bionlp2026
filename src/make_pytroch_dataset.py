"""

python src/make_pytroch_dataset.py EleutherAI/pythia-70m

python src/make_pytroch_dataset.py FacebookAI/roberta-base

python src/make_pytroch_dataset.py distilbert/distilbert-base-uncased

python src/make_pytroch_dataset.py BAAI/bge-m3


"""

from utils import load_json, load_txt_file

import argparse
import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


TOKENIZER_PARSE_FOLDER = "./data/tokenized_examples"
SPLIT_FOLDER = "./data/splits"


class SingleClassBioNLP(Dataset):

    def __init__(self, ai_name, splits, max_size, span_only=False):

        super().__init__()

        if isinstance(splits, str):
            splits = [splits]

        tokenized_folder = os.path.join(TOKENIZER_PARSE_FOLDER, ai_name)

        keys = []
        for split in splits:
            split_file = os.path.join(SPLIT_FOLDER, f"{split}.txt")

            keys += load_txt_file(split_file, lines=True)

        files = [os.path.join(tokenized_folder, f"{k.strip()}.json") for k in keys]

        # DBUGGING
        # files = files[:10]

        examples = []
        for f in files:

            entry = load_json(f)
            entry.pop("sentence")

            token_labels = entry.pop("token_labels")
            input_ids = entry.pop("input_ids")

            num_classes = 2 if span_only else 11
            class_sizes = {i: 0 for i in range(num_classes)}
            entry_labels = []
            for labels in token_labels:

                # span only, igore label categaory
                # no span labels means it is never in a span
                if span_only:
                    if len(labels) == 0:
                        label = 0

                    # for special tokens the span label should only be -100
                    elif -100 in labels:
                        label = -100

                    # otherwise it's in the span of -some- category
                    else:
                        label = 1

                else:
                    if len(labels) == 0:
                        label = 0

                    # for special tokens the span label should only be -100
                    elif -100 in labels:
                        label = -100

                    else:
                        label = labels[0] + 1  # how to handle multiple labels?
                if label != -100:
                    class_sizes[label] += 1

                entry_labels.append(label)

            if len(input_ids) <= max_size:

                example_list = [(input_ids, entry_labels)]

            else:
                example_list = []
                window_size = max_size // 4
                s = 0
                e = max_size
                shift_window = True
                while shift_window:
                    ex = (input_ids[s:e], entry_labels[s:e])
                    example_list.append(ex)

                    if e > len(input_ids):
                        shift_window = False
                    else:
                        s += window_size
                        e += window_size

            for ids, labels in example_list:

                example = {}
                example["input_ids"] = torch.tensor(ids)
                example["labels"] = torch.tensor(labels)

                examples.append(example)

        self.exmaples = examples
        self.class_sizes = class_sizes

    def balanced_weights(self):
        max_class = max(self.class_sizes.values())
        weights = [max_class / v for v in self.class_sizes.values()]
        weights = torch.tensor(weights)
        return weights

    def __len__(self):
        return len(self.exmaples)

    def __getitem__(self, index):

        return self.exmaples[index]


def single_class_collate(batch):

    input_ids = [e["input_ids"] for e in batch]
    input_ids = pad_sequence(input_ids, batch_first=True)

    labels = [e["labels"] for e in batch]
    labels = pad_sequence(labels, padding_value=-100, batch_first=True)

    batch = {"input_ids": input_ids, "labels": labels}

    return batch


def make_dataset(ai_name, split, max_size, single_class=True, span_only=False):

    if single_class:
        dataset = SingleClassBioNLP(ai_name, split, max_size, span_only)

        return dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ai_name")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    dataset = make_dataset(args.ai_name, split="train", max_size=8192, span_only=True)

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=single_class_collate,
        # num_workers=0,
        # prefetch_factor=1,
    )

    for b in dataloader:

        print(b)

    exit()
