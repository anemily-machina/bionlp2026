from make_pytroch_dataset import make_dataset, span_only_collate
from utils import (
    CustomCallback,
    CustomTrainer,
    load_ai_model4token_class,
    load_tokenizer,
)

import evaluate
import numpy as np

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from transformers import TrainingArguments


seqeval = evaluate.load("seqeval")

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


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        p
        for prediction, label in zip(predictions, labels)
        for (p, l) in zip(prediction, label)
        if l != -100
    ]
    true_labels = [
        l
        for prediction, label in zip(predictions, labels)
        for (p, l) in zip(prediction, label)
        if l != -100
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def main(ai_name):

    ai_model = load_ai_model4token_class(ai_name, num_labels=2)
    ai_model.to(device)

    dataset = make_dataset(ai_name, split="train", max_size=8192, span_only=True)
    balanced_weights = dataset.balanced_weights()
    balanced_weights = balanced_weights.to(device)

    _loss_fn = CrossEntropyLoss(weight=balanced_weights)

    def compute_loss_func(outputs, labels, num_items_in_batch):

        logits = outputs.logits
        logits = logits.view(-1, 2)
        labels = labels.view(-1)

        loss = _loss_fn(logits, labels)

        return loss

    training_args = TrainingArguments(
        output_dir="./data/checkpoints/test",
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=100,
        # weight_decay=0.01,
        eval_strategy="epoch",
        # save_strategy="epoch",
        # load_best_model_at_end=True,
        # push_to_hub=False,
        logging_steps=25,
        logging_strategy="steps",
    )

    trainer = CustomTrainer(
        model=ai_model.to(device),
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=span_only_collate,
        compute_loss_func=compute_loss_func,
        train_log_iter=50,
    )

    trainer.add_callback(CustomCallback(trainer))

    trainer.train()


if __name__ == "__main__":
    main("BAAI/bge-m3")
