from make_pytroch_dataset import make_dataset, span_only_collate
from utils import load_ai_model4token_class, load_tokenizer

from evaluate import load as load_metric
import numpy as np

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


def compute_metrics(eval_pred):
    """
    seq size, batch, labels
    """
    # metric1 = load_metric("precision")
    # metric2 = load_metric("recall")
    # metric3 = load_metric("f1")
    metric4 = load_metric("accuracy")

    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    print()
    print()
    print(logits)
    print(labels)
    print(len(labels))
    print(len(logits))
    print()
    print()

    predications = []
    references = []
    for logit_ex, label_ex in zip(logits, labels):
        print()
        print(len(label_ex))
        print(len(logit_ex))

        print()

    predictions = np.argmax(logits, axis=-1)

    # precision = metric1.compute(
    #     predictions=predictions, references=labels, average="micro"
    # )["precision"]
    # recall = metric2.compute(
    #     predictions=predictions, references=labels, average="micro"
    # )["recall"]
    # f1 = metric3.compute(predictions=predictions, references=labels, average="micro")[
    #     "f1"
    # ]
    accuracy = metric4.compute(predictions=predictions, references=labels)["accuracy"]

    # return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}
    return {"accuracy": accuracy}


def main(ai_name):

    ai_model = load_ai_model4token_class(ai_name, num_labels=2)

    ai_model.to(device)

    dataset = make_dataset(ai_name, split="train", max_size=8192, span_only=True)
    dataset = dataset[:10]

    training_args = TrainingArguments(
        output_dir="./data/checkpoints/test",
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=100,
        weight_decay=0.01,
        eval_strategy="epoch",
        # save_strategy="epoch",
        # load_best_model_at_end=True,
        # push_to_hub=False,
        logging_steps=3,
        logging_strategy="steps",
    )

    trainer = Trainer(
        model=ai_model.to(device),
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=span_only_collate,
        # compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main("BAAI/bge-m3")
