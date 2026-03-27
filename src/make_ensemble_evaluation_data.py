from utils import (
    load_ai_model4token_class,
    load_json,
    load_tokenizer,
    load_txt_file,
    loadingbar,
    make_folder,
    save_json,
)

import os

from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
import torch

if torch.cuda.is_available():
    print()
    print("using Cuda")
    print()
    device = torch.device("cuda")
else:
    print()
    print("using cpu")
    print()
    device = torch.device("cpu")


def get_tokenized_text(split, tokenization_folder):

    split_fname = os.path.join("data/splits", f"{split}.txt")
    keys = load_txt_file(split_fname, lines=True)

    all_tokenized_text = {}
    for k in keys:
        k = k.strip()
        text_fname = os.path.join(tokenization_folder, f"{k}.json")

        tokenized_text = load_json(text_fname)

        all_tokenized_text[k] = tokenized_text

    return all_tokenized_text


def make_probs(split, ai_model, tokenization_folder):

    all_tokenized_text = get_tokenized_text(split, tokenization_folder)

    all_probs = {}
    for key, tokenized_text in loadingbar(all_tokenized_text.items()):

        input_ids = torch.tensor(tokenized_text.pop("input_ids"))

        # only one in the training set has this, so it doesn't effect testing
        if len(input_ids) > 8192:
            continue

        attention_mask = torch.tensor([1 for _ in range(len(input_ids))])
        raw_labels = tokenized_text.pop("token_labels")
        labels = []
        for l in raw_labels:
            # not a span, or not labelled data (test data)
            if len(l) == 0:
                labels.append(True)
            # extra tokens from tokenizer
            elif -100 in labels:
                labels.append(False)
            # in a span
            else:
                labels.append(True)

        input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.unsqueeze(0)
        attention_mask = attention_mask.to(device)

        outputs = ai_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        logits = logits.squeeze()

        # testing code outside of actual predictions
        # logits = torch.rand((input_ids.size(1), 3))

        probs = torch.nn.functional.softmax(logits, dim=-1)

        token_ranges = tokenized_text.pop("token_ranges")

        ann_ranges = []

        for prob_row, r, l in zip(probs, token_ranges, labels):

            if not l:
                continue

            # extra tokens from the tokenizer not in original sentence
            if r[0] == r[1]:
                continue

            entry = {"range": r, "probs": prob_row}

            ann_ranges.append(entry)

        all_probs[key] = ann_ranges

    return all_probs


def make_annotations(all_probs, split, eval_folder):

    num_models = len(all_probs)

    all_predications = []
    for example_i, key in enumerate(all_probs[0].keys()):

        ann_ranges = []
        for input_i in range(len(all_probs[0][key])):

            r = all_probs[0][key][input_i]["range"]
            probs = all_probs[0][key][input_i]["probs"]

            for m_i in range(1, num_models):
                probs += all_probs[m_i][key][input_i]["probs"]

            cat = probs.argmax()
            cat = int(cat)

            if cat == 0:
                continue

            entry = {"range": r, "category": cat}

            ann_ranges.append(entry)

        key_annotations = []

        if len(ann_ranges) > 0:
            curr_ann = [ann_ranges[0]]
            curr_cat = ann_ranges[0]["category"]

            for ann_range in ann_ranges[1:]:

                cat = ann_range["category"]

                if cat != curr_cat:

                    key_ann_cat = curr_cat
                    key_ann_so = curr_ann[0]["range"][0]
                    key_ann_eo = curr_ann[-1]["range"][1]

                    key_annotation_entry = {
                        "start_offset": key_ann_so,
                        "end_offset": key_ann_eo,
                        "category": key_ann_cat,
                    }
                    key_annotations.append(key_annotation_entry)

                    curr_cat = cat
                    curr_ann = [ann_range]

                else:
                    curr_ann.append(ann_range)

            key_ann_cat = curr_cat
            key_ann_so = curr_ann[0]["range"][0]
            key_ann_eo = curr_ann[-1]["range"][1]

            key_annotation_entry = {
                "start_offset": key_ann_so,
                "end_offset": key_ann_eo,
                "category": key_ann_cat,
            }
            key_annotations.append(key_annotation_entry)

        all_preds_entry = {"file_name": key, "predictions": key_annotations}
        all_predications.append(all_preds_entry)

    pred_fname = os.path.join(eval_folder, f"{split}.json")
    save_json(all_predications, pred_fname)


def main():
    ai_name = "BAAI/bge-m3"
    checkpoints = [
        "data/checkpoints/single_class5/checkpoint-803",
        "data/checkpoints/single_class7/checkpoint-561",
        "data/checkpoints/single_class6/checkpoint-451",
    ]
    num_classes = 10

    eval_folder = "data/eval567"
    tokenization_folder = f"data/tokenized_examples/{ai_name}"

    make_folder(eval_folder)

    for split in ["val", "test", "train"]:

        all_probs = []
        for checkpoint in checkpoints:

            # ai_model = None

            ai_model = load_ai_model4token_class(ai_name, num_labels=num_classes)
            ai_model.float()

            ai_model = PeftModel.from_pretrained(ai_model, checkpoint)
            ai_model = ai_model.merge_and_unload()

            ai_model.eval()

            ai_model.to(device)

            checkpoint_probs = make_probs(split, ai_model, tokenization_folder)

            all_probs.append(checkpoint_probs)

        make_annotations(all_probs, split, eval_folder)


if __name__ == "__main__":
    with torch.no_grad():
        main()
