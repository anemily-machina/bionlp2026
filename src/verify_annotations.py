from utils import load_all_raw_text, load_all_annotations

import os


def main():
    raw_text = load_all_raw_text()
    annotations = load_all_annotations()

    mismatch_count = 0

    for k, text in raw_text.items():

        all_text_annotation = annotations[k]["annotations"]

        for ann in all_text_annotation:

            category = ann["category"]

            # ignore these for the task
            if "10" in category or "11" in category:
                continue

            decision = ann["decision"]
            start = ann["start_offset"]
            end = ann["end_offset"]

            actual = text[start:end]

            actual = " ".join(actual.split("\n"))

            if decision != actual:

                mismatch_count += 1

                # print()
                # print()
                # print(decision)
                # print()
                # print()
                # print(actual)
                # print()
                # print()
                # print(ann)
                # print()
                # print()

    print(mismatch_count)


if __name__ == "__main__":
    main()
