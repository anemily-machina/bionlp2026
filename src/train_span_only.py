from make_pytroch_dataset import make_dataset, single_class_collate
from utils import (
    CustomCallback,
    CustomTrainer,
    load_ai_model4token_class,
    load_tokenizer,
)

import evaluate
import numpy as np

from peft import inject_adapter_in_model, LoraConfig

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

    train_dataset = make_dataset(ai_name, split="train", max_size=8192, span_only=False)
    balanced_weights = train_dataset.balanced_weights()
    balanced_weights = balanced_weights.to(device)

    val_dataset = make_dataset(ai_name, split="val", max_size=8192, span_only=False)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["query", "key", "value", "dense"],
        modules_to_save=["classifier"],
        init_lora_weights="pissa_niter_10",
    )

    ai_model = load_ai_model4token_class(
        ai_name, num_labels=int(balanced_weights.size(0))
    )
    ai_model.float()
    ai_model = inject_adapter_in_model(lora_config, ai_model)
    ai_model.to(device)

    _loss_fn = CrossEntropyLoss(weight=balanced_weights)

    def compute_loss_func(outputs, labels, num_items_in_batch):

        logits = outputs.logits
        logits = logits.view(-1, 2)
        labels = labels.view(-1)

        loss = _loss_fn(logits, labels)

        return loss

    training_args = TrainingArguments(
        output_dir="./data/checkpoints/test",
        learning_rate=1e-3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1000,
        weight_decay=0.01,
        warmup_steps=10,
        eval_strategy="epoch",
        logging_steps=10,
        logging_strategy="steps",
        save_total_limit=1,
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=4321,
        gradient_accumulation_steps=8,
    )

    trainer = CustomTrainer(
        model=ai_model.to(device),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=single_class_collate,
        compute_loss_func=compute_loss_func,
        train_log_iter=100,
        num_classes=int(balanced_weights.size(0)),
    )

    trainer.add_callback(CustomCallback(trainer))

    trainer.train()


if __name__ == "__main__":
    main("BAAI/bge-m3")
