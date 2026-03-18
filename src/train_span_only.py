from make_pytroch_dataset import make_dataset, span_only_collate
from utils import load_ai_model4token_class, load_tokenizer

import torch
from torch.utils.data import DataLoader

from transformers import Trainer, TrainingArguments

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

    ai_model = load_ai_model4token_class(ai_name, num_labels=2)

    ai_model.to(device)

    dataset = make_dataset(ai_name, split="train", max_size=8192, span_only=True)
    dataset = dataset[:10]

    training_args = TrainingArguments(
        output_dir="./data/checkpoints/test",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=100,
        weight_decay=0.01,
        eval_strategy="epoch",
        # save_strategy="epoch",
        # load_best_model_at_end=True,
        # push_to_hub=False,
        logging_steps=5,
        logging_strategy="steps",
    )

    trainer = Trainer(
        model=ai_model.to(device),
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=span_only_collate,
    )

    trainer.train()


if __name__ == "__main__":
    main("BAAI/bge-m3")
