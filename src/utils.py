import json
import os
import pathlib

from typing import Any

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    PreTrainedTokenizer,
    Trainer,
)
from transformers.integrations.deepspeed import deepspeed_sp_compute_loss
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_utils import _is_peft_model


from tqdm import tqdm

RAW_TEXT_FOLDER = "./data/MedDec/raw_text"
ANNOTATION_FOLDER = "./data/MedDec/data"
STANZA_PARSE_FOLDER = "./data/MedDec/stanza_parses"
CAHCHE_FOLDER = "./data/cache"


def make_folder(path):
    pathlib.Path(path).mkdir(exist_ok=True, parents=True)


def loadingbar(iterator, has_length=True):

    total = len(iterator) if has_length else None
    lb = tqdm(iterator, ncols=60, total=total)
    return lb


def load_tokenizer(name) -> PreTrainedTokenizer:

    if os.path.isdir(name):
        cache_dir = None
    else:
        cache_dir = os.path.join(CAHCHE_FOLDER, name)

    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)

    return tokenizer


def load_ai_model4token_class(name, num_labels):

    if os.path.isdir(name):
        cache_dir = None
    else:
        cache_dir = os.path.join(CAHCHE_FOLDER, name)

    ai_model = AutoModelForTokenClassification.from_pretrained(
        name, cache_dir=cache_dir, num_labels=num_labels
    )

    return ai_model


def load_all_raw_text():

    raw_text = {}

    for f in os.listdir(RAW_TEXT_FOLDER):

        key = f.split(".")[0]

        f = os.path.join(RAW_TEXT_FOLDER, f)

        text = load_txt_file(f)

        raw_text[key] = text

    return raw_text


def load_all_annotations():

    annotations = {}

    for f in os.listdir(ANNOTATION_FOLDER):

        key = f.split(".")[0]

        f = os.path.join(ANNOTATION_FOLDER, f)

        ann = load_json(f)

        annotations[key] = ann

    return annotations


def load_all_stanza_parses():

    parses = {}

    for f in os.listdir(STANZA_PARSE_FOLDER):

        key = f.split(".")[0]

        f = os.path.join(STANZA_PARSE_FOLDER, f)

        ann = load_json(f)

        parses[key] = ann

    return parses


def load_txt_file(fname, lines=False):

    with open(fname, "r") as f_in:
        if lines:
            text = f_in.readlines()
        else:
            text = f_in.read()

    return text


def save_txt_file(text, fname):

    with open(fname, "w") as f_out:

        f_out.write(text)


def load_json(fname):

    with open(fname, "r") as f_in:
        data = json.load(f_in)

    return data


def save_json(data, fname, full=False):

    indent = None if full else 2

    with open(fname, "w") as f_out:
        json.dump(data, f_out, indent=indent)


class CustomTrainer(Trainer):

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Args:
            model (`nn.Module`):
                The model to compute the loss for.
            inputs (`dict[str, torch.Tensor | Any]`):
                The input data for the model.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Whether to return the model outputs along with the loss.
            num_items_in_batch (Optional[torch.Tensor], *optional*):
                The number of items in the batch. If not passed, the loss is computed
                using the default batch size reduction logic.

        Returns:
            The loss of the model along with its output if return_outputs was set to True

        Subclass and override for custom behavior. If you are not using `num_items_in_batch` when computing your loss,
        make sure to overwrite `self.model_accepts_loss_kwargs` to `False`. Otherwise, the loss calculation might be slightly inaccurate when performing gradient accumulation.
        """
        pc = getattr(self.accelerator, "parallelism_config", None)
        if (
            pc is not None
            and pc.sp_backend == "deepspeed"
            and pc.sp_enabled
            and self.model.training
        ):
            return deepspeed_sp_compute_loss(
                self.accelerator, model, inputs, return_outputs, pc
            )

        if (
            self.label_smoother is not None or self.compute_loss_func is not None
        ) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}

        outputs = model(**inputs)
        logits = outputs.logits

        print()
        print()
        print(outputs)
        print()
        print(labels)
        print()
        print(outputs.size())
        print(labels.size())
        print()

        # User-defined compute_loss function
        if self.compute_loss_func is not None:
            if labels is None:
                print(
                    "Trainer: `compute_loss_func` is defined but `labels=None`. "
                    "Your custom loss function will still be called with labels=None. "
                )
            loss = self.compute_loss_func(
                outputs,
                labels,
                num_items_in_batch=num_items_in_batch,
            )
        # Default HF loss handling (label smoothing) if no custom loss function
        elif labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            model_name = (
                unwrapped_model.base_model.model._get_name()
                if _is_peft_model(unwrapped_model)
                else unwrapped_model._get_name()
            )
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= (
                self.accelerator.num_processes
                if self.args.n_gpu <= 1
                else self.args.n_gpu
            )

        return (loss, outputs) if return_outputs else loss
