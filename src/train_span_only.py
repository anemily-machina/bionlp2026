from make_pytroch_dataset import make_dataset, span_only_collate
from utils import load_ai_model4token_class

import torch
from torch.utils.data import DataLoader

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


def main(ai_name):

    ai_model = load_ai_model4token_class(ai_name)
    ai_model.to(device)

    dataset = make_dataset(ai_name, split="train", max_size=8192, span_only=True)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=span_only_collate,
        shuffle=False,
        # num_workers=0,
        # prefetch_factor=1,
    )

    for batch in dataloader:

        batch = {k: v.to(device) for k, v in batch.items()}

        # labels = batch.pop("labels")

        output = ai_model(**batch)

        print(output)

        exit()

    print(ai_model)


if __name__ == "__main__":
    main("BAAI/bge-m3")
