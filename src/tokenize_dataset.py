"""

python src/tokenize_dataset.py EleutherAI/pythia-70m

python src/tokenize_dataset.py FacebookAI/roberta-base

python src/tokenize_dataset.py distilbert/distilbert-base-uncased

python src/tokenize_dataset.py BAAI/bge-m3

"""

from utils import (
    load_all_raw_text,
    load_all_annotations,
    load_all_stanza_parses,
    load_json,
    load_tokenizer,
    save_json,
    loadingbar,
    make_folder,
)

import argparse
import json
import os

from transformers import AutoTokenizer

LABELLED_PARSE_FOLDER = "./data/labelled_parses"
make_folder(LABELLED_PARSE_FOLDER)
TOKENIZER_PARSE_FOLDER = "./data/tokenized_examples"
make_folder(TOKENIZER_PARSE_FOLDER)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    args = parser.parse_args()

    return args


def preprocess_parses(all_parses):

    p_parses = {}
    for key, parse in all_parses.items():

        all_words = []

        for sent in parse:

            for word in sent:
                word["labels"] = []

            all_words += sent

        p_parses[key] = all_words

    return p_parses


def parse_category(category: str):

    if not category.startswith("Category "):
        return None
    category = category[9:]

    if category[0] != "1":
        return int(category[0])

    else:
        if category[1] == 0 or category[1] == 1:
            return None
        else:
            return 1


def make_raw_word_token_labels(all_annotations, all_parses):

    for key, parse in loadingbar(all_parses.items()):

        label_fname = os.path.join(LABELLED_PARSE_FOLDER, f"{key}.json")
        if os.path.isfile(label_fname):
            continue

        if key in all_annotations:
            annotations = all_annotations[key]

            annotations = annotations["annotations"]

            for ann in annotations:

                category: str = ann["category"]

                category_label = parse_category(category)

                # if class 10 or 11 or TBD
                if category_label is None:
                    continue

                start_offset = ann["start_offset"]
                end_offset = ann["end_offset"]

                token_i = 0
                curr_token = parse[token_i]
                token_start = curr_token["start_char"]

                test_text = []
                while token_start < end_offset:

                    if start_offset <= token_start:

                        curr_token["labels"].append(category_label)
                        test_text.append(curr_token["text"])

                    token_i += 1
                    if token_i == len(parse):
                        break

                    curr_token = parse[token_i]
                    token_start = curr_token["start_char"]

                test_text = " ".join(test_text)

            for word in parse:
                labels = word["labels"]
                labels = list(set(labels))
                labels = sorted(labels)
                word["labels"] = labels

        save_json(parse, label_fname)


def make_tokenizations(all_parses, model_name):

    tokenized_folder = os.path.join(TOKENIZER_PARSE_FOLDER, model_name)
    make_folder(tokenized_folder)

    tokenizer = load_tokenizer(model_name)

    for key in loadingbar(all_parses.keys()):  # enumerate(all_parses.keys()):

        tokenizer_parse_fname = os.path.join(tokenized_folder, f"{key}.json")

        labelled_parse_fname = os.path.join(LABELLED_PARSE_FOLDER, f"{key}.json")

        parse = load_json(labelled_parse_fname)

        word_tokens = [w["text"].strip() for w in parse]

        sentence = " ".join(word_tokens)

        tokenization = tokenizer(sentence, return_offsets_mapping=True)
        offset_mappings = tokenization["offset_mapping"]
        input_ids = tokenization["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        token_labels = []
        token_ranges = []

        parse_i = 0
        current_entry = parse[parse_i]
        current_word = current_entry["text"]
        current_labels = current_entry["labels"]
        crs = current_entry["start_char"]
        cre = current_entry["end_char"]
        current_range = (crs, cre)
        first = True

        for o_i, offset_mapping in enumerate(offset_mappings):

            map_s, map_e = offset_mapping

            if map_s == map_e:
                token_labels.append([-100])
                token_ranges.append((0, 0))
                continue

            # handling weird case for BGE tokenizer that add a token of size 0
            if tokens[o_i] == "▁":
                token_labels.append([-100])
                token_ranges.append((0, 0))
                continue

            if len(current_word) == 0:
                parse_i += 1
                current_entry = parse[parse_i]
                current_word = current_entry["text"]
                current_labels = current_entry["labels"]
                crs = current_entry["start_char"]
                cre = current_entry["end_char"]
                current_range = (crs, cre)
                first = True

            # only label first token
            if first:
                print("?")
                exit()
                token_labels.append(current_labels)
                token_ranges.append(current_range)
                first = False
            else:
                token_labels.append([-100])
                token_ranges.append((0, 0))

            chunk_size = map_e - map_s

            # print()
            # print(offset_mapping)

            # print(tokens[o_i])
            # print(len(tokens[o_i]))
            # print(input_ids[o_i])
            # print(current_word[:chunk_size])
            # print(current_word)
            # print(sentence[map_s:map_e])
            # print()

            if chunk_size > len(current_word):
                print()
                # print(k_i)
                print("token size mismatch")
                print()
                exit()

            current_word = current_word[chunk_size:]
            current_word = current_word.strip()  # sometimes stanza keeps whitespace

        entry = {
            "sentence": sentence,
            "input_ids": input_ids,
            "token_labels": token_labels,
            "token_ranges": token_ranges,
        }

        save_json(entry, tokenizer_parse_fname)


def main():

    args = parse_args()
    model_name = args.model_name

    all_raw_text = load_all_raw_text()
    all_annotations = load_all_annotations()
    all_parses = load_all_stanza_parses()
    all_parses = preprocess_parses(all_parses)

    make_raw_word_token_labels(all_annotations, all_parses)

    make_tokenizations(all_parses, model_name)


if __name__ == "__main__":
    main()
