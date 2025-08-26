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
        logits = logits[0]
    return logits.argmax(dim=-1)


def get_logits_from_base_model(base_model, data_collator, dataset, batch_size=32):
    train_loader = DataLoader(dataset['train'], collate_fn=data_collator, batch_size=batch_size)
    device = next(base_model.parameters()).device
    base_model.eval()

    original_logits = {}
    for batch in tqdm(train_loader, desc="Caching base logits"):
        indices = batch.pop('index').to(device)
        batch.pop('is_forget', None)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = base_model(**batch).logits

        labels = batch['labels']
        mask = labels != -100
        stripped = logits[mask].detach().cpu()
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

class PrefixTuning(nn.Module):
    def __init__(self, base_model, config, num_layers, prefix_length, hidden_size):
        super().__init__()
        self.model = base_model
        self.config = config
        self.num_layers = num_layers
        self.prefix_length = prefix_length
        self.hidden_size = hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # learnable prefix keys & values for each layer
        self.K_p = nn.Parameter(torch.randn(num_layers, prefix_length, hidden_size))
        self.V_p = nn.Parameter(torch.randn(num_layers, prefix_length, hidden_size))
        self.dropout = nn.Dropout(getattr(config, "dropout", 0.1))

    def get_prefix(self, batch_size: int, device: torch.device):
        """Return list of (key, value) tuples shaped (B, H, P, D) for each layer."""
        past_key_values = []
        for i in range(self.num_layers):
            # (1, P, H*D) -> (B, P, H*D)
            K = self.K_p[i].unsqueeze(0).expand(batch_size, -1, -1)
            V = self.V_p[i].unsqueeze(0).expand(batch_size, -1, -1)
            # reshape -> (B, P, H, D) -> (B, H, P, D)
            K = K.view(batch_size, self.prefix_length, self.num_heads, self.head_dim) \
                 .permute(0, 2, 1, 3).contiguous().to(device)
            V = V.view(batch_size, self.prefix_length, self.num_heads, self.head_dim) \
                 .permute(0, 2, 1, 3).contiguous().to(device)
            past_key_values.append((self.dropout(K), self.dropout(V)))
        return past_key_values


def inject_prefix_to_model(model, prefix_module):
    original_forward = model.forward

    def modified_forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Get prefix past key values
        prefix_past_key_values = prefix_module.get_prefix(batch_size=batch_size, device=device)
        kwargs["past_key_values"] = prefix_past_key_values
        kwargs.setdefault("use_cache", False)

        # Extend attention mask to include prefix length
        prefix_length = prefix_past_key_values[0][0].size(-2)  # (batch, num_heads, prefix_len, head_dim)
        prefix_attention_mask = torch.ones(batch_size, prefix_length, device=device, dtype=attention_mask.dtype)
        extended_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        return original_forward(
            input_ids=input_ids,
            attention_mask=extended_attention_mask,
            labels=labels,
            **kwargs
        )

    import types
    model.forward = types.MethodType(modified_forward, model)
    return model


def compute_pinv_matrix(P: torch.Tensor) -> torch.Tensor:
    """
    Compute the Moore Penrose pseudo-inverse of a 2D matrix P.
    Input:  P of shape (p, d)
    Output: P_pinv of shape (d, p)
    """
    # Move to CPU / NumPy, do the pinv, then back to the same device
    P_cpu = P.detach().cpu().numpy()
    pinv_cpu = np.linalg.pinv(P_cpu)             # (d, p)
    return torch.from_numpy(pinv_cpu).to(P.device)

def apply_pinv_to_prefix(prefix_module: PrefixTuning) -> None:
    """
    Replace each layers learned K_p and V_p with their pseudo-inverse counterparts:
      K_p[i] ← (pinv(K_p[i]))^T
      V_p[i] ← (pinv(V_p[i]))^T

    After calling this, your prefix module will *negate* the subspace
    learned on the forget set at inference time.
    """
    # Loop over layers
    for i in range(prefix_module.num_layers):
        # K_p[i]: (prefix_length, hidden_size)
        K = prefix_module.K_p[i]
        V = prefix_module.V_p[i]

        # compute pinv: shapes (hidden_size, prefix_length)
        K_pinv = compute_pinv_matrix(K)
        V_pinv = compute_pinv_matrix(V)

        # store their transposes back into the module
        # i.e. K_p[i] becomes (prefix_length, hidden_size) again
        prefix_module.K_p.data[i] = K_pinv.T
        prefix_module.V_p.data[i] = V_pinv.T
    return prefix_module

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