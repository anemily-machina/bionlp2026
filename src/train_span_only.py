from make_pytroch_dataset import make_dataset, span_only_collate
from utils import CustomTrainer, load_ai_model4token_class, load_tokenizer

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

    print(ai_model)

    exit()

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

    trainer = CustomTrainer(
        model=ai_model.to(device),
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=span_only_collate,
        compute_metrics=compute_metrics,
        compute_loss_func=CrossEntropyLoss,
    )

    trainer.train()


if __name__ == "__main__":
    main("BAAI/bge-m3")
