import os

import pickle
import datetime
import time
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from numpy.random import default_rng
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from datasets import load_dataset, concatenate_datasets
import evaluate
from peft import get_peft_model, TaskType, PromptEncoderConfig, PeftConfig, PeftModel
from transformers import AutoTokenizer, TrainerState, TrainerControl, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback
from trl import DataCollatorForCompletionOnlyLM


def get_data_path(args, dataset):
    if args.dataset.lower() == "sst2":
        data_path = "llm_unlearning/sst2" # provide your local path
    elif args.dataset.lower() == 'yelp':
        data_path = "llm_unlearning/yelp" # provide your local path
    else:
        raise NotImplementedError
    return data_path


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    # argmax to get the token ids
    return logits.argmax(dim=-1)


def get_logits_from_base_model(base_model, data_collator, dataset, batch_size=32):
    """
    Run the base_model on D_train (concatenated train_retain+train_forget),
    moving every batch to the device where base_model lives, and
    cache the “prompt‑stripped” logits in a dict by sample index.
    """
    train_loader = DataLoader(dataset['train'], collate_fn=data_collator, batch_size=batch_size)
    device = next(base_model.parameters()).device
    base_model.eval()

    original_logits = {}
    for batch in tqdm(train_loader, desc="Caching base logits"):
        # 1) pop control cols
        indices = batch.pop('index').to(device)
        batch.pop('is_forget', None)
        # 2) move all remaining inputs to model device
        batch = {k: v.to(device) for k, v in batch.items()}

        # 3) forward pass
        with torch.no_grad():
            logits = base_model(**batch).logits

        # 4) strip away prompt tokens (mask == -100) and cache
        labels = batch['labels']
        mask = labels != -100
        stripped = logits[mask].detach().cpu()

        # 5) store each sample’s output under its original index
        for i, idx in enumerate(indices.cpu().tolist()):
            original_logits[idx] = stripped[i]

    return original_logits

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if logits.ndim == 3:
        preds = np.argmax(logits, axis=-1)
    else:
        preds = logits

    min_seq_len = min(preds.shape[1], labels.shape[1])
    preds = preds[:, :min_seq_len]
    labels = labels[:, :min_seq_len]
    mask = labels != -100

    flat_preds  = preds[mask]
    flat_labels = labels[mask]

    return {
        "accuracy":  accuracy_score(flat_labels, flat_preds),
        "f1":        f1_score(flat_labels, flat_preds, average="weighted"),
        "precision": precision_score(flat_labels, flat_preds, average="micro"),
        "recall":    recall_score(flat_labels, flat_preds, average="micro"),
    }


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.eval_dataset['train_retain'],
                                   metric_key_prefix="eval_train_retrain")
            self._trainer.evaluate(eval_dataset=self._trainer.eval_dataset['train_forget'],
                                   metric_key_prefix="eval_train_forget")
            return control_copy