import json
import os
import pathlib

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    PreTrainedTokenizer,
)
from tqdm import tqdm

RAW_TEXT_FOLDER = "./data/MedDec/raw_text"
ANNOTATION_FOLDER = "./data/MedDec/data"
STANZA_PARSE_FOLDER = "./data/MedDec/stanza_parses"
CAHCHE_FOLDER = "./data/cache"


def make_folder(path):
    pathlib.Path(path).mkdir(exist_ok=True, parents=True)


def loadingbar(iterator, has_length=True):

    total = len(iterator) if has_length else None
    lb = tqdm(iterator, ncols=60, total=total)
    return lb


def load_tokenizer(name) -> PreTrainedTokenizer:

    if os.path.isdir(name):
        cache_dir = None
    else:
        cache_dir = os.path.join(CAHCHE_FOLDER, name)

    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)

    return tokenizer


def load_ai_model4token_class(name):

    if os.path.isdir(name):
        cache_dir = None
    else:
        cache_dir = os.path.join(CAHCHE_FOLDER, name)

    ai_model = AutoModelForTokenClassification.from_pretrained(
        name, cache_dir=cache_dir
    )

    return ai_model


def load_all_raw_text():

    raw_text = {}

    for f in os.listdir(RAW_TEXT_FOLDER):

        key = f.split(".")[0]

        f = os.path.join(RAW_TEXT_FOLDER, f)

        text = load_txt_file(f)

        raw_text[key] = text

    return raw_text


def load_all_annotations():

    annotations = {}

    for f in os.listdir(ANNOTATION_FOLDER):

        key = f.split(".")[0]

        f = os.path.join(ANNOTATION_FOLDER, f)

        ann = load_json(f)

        annotations[key] = ann

    return annotations


def load_all_stanza_parses():

    parses = {}

    for f in os.listdir(STANZA_PARSE_FOLDER):

        key = f.split(".")[0]

        f = os.path.join(STANZA_PARSE_FOLDER, f)

        ann = load_json(f)

        parses[key] = ann

    return parses


def load_txt_file(fname, lines=False):

    with open(fname, "r") as f_in:
        if lines:
            text = f_in.readlines()
        else:
            text = f_in.read()

    return text


def save_txt_file(text, fname):

    with open(fname, "w") as f_out:

        f_out.write(text)


def load_json(fname):

    with open(fname, "r") as f_in:
        data = json.load(f_in)

    return data


def save_json(data, fname, full=False):

    indent = None if full else 2

    with open(fname, "w") as f_out:
        json.dump(data, f_out, indent=indent)
