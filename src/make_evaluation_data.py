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

from peft import LoraConfig, PeftModel
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


def make_annotaions(split, ai_model, eval_folder, tokenization_folder):

    all_tokenized_text = get_tokenized_text(split, tokenization_folder)

    all_predications = []
    for key, tokenized_text in loadingbar(all_tokenized_text.items()):

        input_ids = torch.tensor(tokenized_text.pop("input_ids"))
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

        input_ids = input_ids.unsqueeze(-1)
        attention_mask = attention_mask.unsqueeze(-1)

        print(input_ids.size())
        print(attention_mask.size())

        outputs = ai_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        print(logits.size())

        exit()

        # logits = torch.rand((input_ids.size(0), 2))

        preds = logits.argmax(axis=-1)
        preds = preds.cpu().numpy()

        token_ranges = tokenized_text.pop("token_ranges")

        ann_ranges = []
        for cat, r, l in zip(preds, token_ranges, labels):

            if not l:
                continue

            # extra tokens from the tokenizer not in original sentence
            if r[0] == r[1]:
                continue

            cat = int(cat)

            entry = {"range": r, "category": cat}

            ann_ranges.append(entry)

        curr_ann = [ann_ranges[0]]
        curr_cat = ann_ranges[0]["category"]

        key_annotations = []
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
    checkpoint = "data/checkpoints/single_class5/checkpoint-803"
    num_classes = 10

    eval_folder = "data/eval5"
    tokenization_folder = f"data/tokenized_examples/{ai_name}"

    make_folder(eval_folder)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["query", "key", "value", "dense"],
        modules_to_save=["classifier"],
        init_lora_weights="pissa_niter_10",
    )

    # ai_model = None
    ai_model = load_ai_model4token_class(ai_name, num_labels=num_classes)
    ai_model.float()

    ai_model = PeftModel.from_pretrained(ai_model, checkpoint, config=lora_config)

    ai_model.eval()

    ai_model.to(device)

    for split in ["train", "val", "test"]:

        make_annotaions(split, ai_model, eval_folder, tokenization_folder)


if __name__ == "__main__":
    with torch.no_grad():
        main()
