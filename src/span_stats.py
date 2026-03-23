from utils import load_json, load_txt_file, loadingbar, save_json

import json
import math
import os


def get_keys():

    splits = [
        "train",
    ]  # "val"]  # only use train?

    keys = []
    for split in splits:

        split_file = os.path.join("./data/splits", f"{split}.txt")

        raw_keys = load_txt_file(split_file, lines=True)

        raw_keys = [k.strip() for k in raw_keys]

        keys += raw_keys

    return keys


def load_annotations(keys):

    ann_folder = "data/MedDec/data"

    all_annotations = {}

    for key in loadingbar(keys):

        f = os.path.join(ann_folder, f"{key}.json")

        ann = load_json(f)

        all_annotations[key] = ann

    return all_annotations


def seperate_annotaions(all_annotations):

    categories = [
        "Category 1:",
        "Category 2:",
        "Category 3:",
        "Category 4:",
        "Category 5:",
        "Category 6:",
        "Category 7:",
        "Category 8:",
        "Category 9:",
        "Category 10",
        "Category 11",
        "TBD",
    ]

    ann_by_cat = {
        cat_str: {key: [] for key in all_annotations.keys()} for cat_str in categories
    }
    for key, annotations in loadingbar(all_annotations.items()):

        for ann in annotations["annotations"]:

            category = ann.pop("category")
            cat_str = category[:11]

            ann_by_cat[cat_str][key].append(ann)

    ignore_cat = ["Category 10", "Category 11", "TBD"]
    for c in ignore_cat:
        ann_by_cat.pop(c)

    # sort annotations per summary
    for cat_str, cat_key_anns in ann_by_cat.items():

        for key in cat_key_anns.keys():

            key_anns = cat_key_anns[key]
            key_anns = sorted(key_anns, key=lambda x: x["start_offset"])
            cat_key_anns[key] = key_anns

    return ann_by_cat


def calc_stats(ann_by_cat):

    stat_types = ["token_size (tokens)", "span_gaps (tokens)", "total_count (spans)"]

    hists = {k: {s_type: {} for s_type in stat_types} for k in ann_by_cat.keys()}

    for cat_str, cat_key_anns in ann_by_cat.items():

        cat_hists = hists[cat_str]

        for key, key_anns in cat_key_anns.items():

            count = len(key_anns)
            if count not in cat_hists["count"]:
                cat_hists["count"][count] = 0
            cat_hists["count"][count] += 1

            raw_text_f = os.path.join("data/MedDec/raw_text", f"{key}.txt")
            raw_text = load_txt_file(raw_text_f)

            d_sizes = [len(ann["decision"].split()) for ann in key_anns]

            gaps = []
            if len(key_anns) > 1:

                for a_i in range(len(key_anns) - 1):
                    prev_end = key_anns[a_i]["end_offset"]
                    next_start = key_anns[a_i + 1]["start_offset"]
                    gap = next_start - prev_end
                    if gap < 0:
                        # print()
                        # print()
                        # print("--------")
                        # print(key_anns[a_i]["decision"])
                        # print("--------")
                        # print(key_anns[a_i + 1]["decision"])
                        # print("--------")
                        # print(prev_end)
                        # print(next_start)
                        # print(key)
                        # print(cat_str)
                        continue
                    gap_text = raw_text[prev_end:next_start]
                    gap_size = len(gap_text.strip().split())
                    gaps.append(gap_size)

            cat_token_hist = cat_hists["token_size"]

            for d_size in d_sizes:
                if d_size not in cat_token_hist:
                    cat_token_hist[d_size] = 0
                cat_token_hist[d_size] += 1

            cat_gap_hist = cat_hists["span_gaps"]
            for gap in gaps:
                if gap not in cat_gap_hist:
                    cat_gap_hist[gap] = 0

                cat_gap_hist[gap] += 1

    for cat_str, cat_hists in hists.items():

        for hist_key in stat_types:
            cat_hist = cat_hists[hist_key]
            cat_hist = list(cat_hist.items())
            cat_hist = sorted(cat_hist, key=lambda x: x[0])
            cat_hist = dict(cat_hist)
            cat_hists[hist_key] = cat_hist

    save_json(hists, "./hist_test.json")

    stats = {k: {} for k in hists.keys()}
    for cat_str, cat_hists in hists.items():
        stats[cat_str] = {}
        for hist_key in stat_types:
            cat_hist = cat_hists[hist_key]

            total = sum([v for v in cat_hist.values()])
            mean = sum([k * v for k, v in cat_hist.items()]) / total

            std = sum([k * (v - mean) ** 2 for k, v in cat_hist.items()]) / total
            std = math.sqrt(std)

            stat_entry = {"mean": mean, "std": std}

            stats[cat_str][hist_key] = stat_entry

    save_json(stats, "./stat_test.json")


def main():

    keys = get_keys()

    all_annotations = load_annotations(keys)

    ann_by_cat = seperate_annotaions(all_annotations)

    stats_by_cat = calc_stats(ann_by_cat)


if __name__ == "__main__":
    main()
