import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
torch.cuda.set_device(0)

from torch.utils.data import DataLoader
from torchinfo import summary

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from peft import PeftModel
from trl import DataCollatorForCompletionOnlyLM

from utils_pi import (
    get_data_path,
    compute_metrics,
    preprocess_logits_for_metrics,
    get_logits_from_base_model,
    CustomCallback,
    PrefixTuning,
    inject_prefix_to_model,
    apply_pinv_to_prefix
)

def get_args():
    parser = ArgumentParser(description="LLM Unlearning using PI-Prefix method")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_checkpoints", type=str, required=True)
    parser.add_argument("--logits_path", type=str, default=None)
    parser.add_argument("--forget_size", type=float, default=1.0)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--set_pad_id", action="store_true")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--prefix_hidden_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    return parser.parse_args()


def get_pi_prefix_model(model_checkpoints, prefix_length, prefix_hidden_size):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_checkpoints,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoints, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    num_layers = len(base_model.model.decoder.layers)  # OPT: model.model.decoder.layers
    prefix_tuning_module = PrefixTuning(
        base_model=base_model.model,
        config=base_model.config,
        num_layers=num_layers,
        prefix_length=prefix_length,
        hidden_size=prefix_hidden_size
    )

    summary(prefix_tuning_module)

    return prefix_tuning_module, base_model, tokenizer


def get_dataset_and_collator(data_path, tokenizer, forget_size, max_length):
    prompt_template = lambda text, label: f"""### Text: {text}\n\n### Question: What is the sentiment of the given text?\n\n### Sentiment: {label}"""
    response_template = "### Sentiment:"

    def preprocess(examples):
        return tokenizer(prompt_template(examples['text'], examples['label_text']),
                         truncation=True, max_length=max_length)

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )

    data = load_dataset(data_path)
    if forget_size < 1.0:
        rng = default_rng(seed=42)
        idx = rng.choice(data['train_forget'].num_rows, int(forget_size * data['train_forget'].num_rows), replace=False)
        data['train_forget'] = data['train_forget'].select(idx)

    random_labels = ['neutral', 'unknown']
    forget_flipped = deepcopy(data['train_forget']).map(lambda x: {"label_text": random.choice(random_labels)})
    forget_flipped = forget_flipped.map(lambda x: {"is_forget": 1})
    data['train_retain'] = data['train_retain'].map(lambda x: {"is_forget": 0})
    data['train'] = concatenate_datasets([data['train_retain'], forget_flipped])

    data['train'] = data['train'].map(lambda x, idx: {"index": idx}, with_indices=True)

    data = data.map(preprocess)
    data = data.remove_columns(["text", "label", "label_text"])
    data.set_format("torch")

    return data, collator


def train_prefix_manual(prefix_tuning_module, model, tokenizer, dataset, collator, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    num_layers = len(model.model.decoder.layers)  # OPT: model.model.decoder.layers
    prefix_tuning = PrefixTuning(
        base_model=model.model,
        config=model.config,
        num_layers=num_layers,
        prefix_length=args.prefix_length,
        hidden_size=model.config.hidden_size
    ).to(device)

    inject_prefix_to_model(model, prefix_tuning)

    optimizer = torch.optim.AdamW(prefix_tuning.parameters(), lr=args.lr)

    dataloader = DataLoader(dataset['train'], batch_size=args.train_batch_size,
                            shuffle=True, collate_fn=collator)

    for epoch in range(args.num_epochs):
        total_loss = 0.0
        for step, batch in enumerate(tqdm(dataloader)):
            # input tensors
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )


            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")

        print(f"Epoch {epoch} Avg Loss: {total_loss / len(dataloader):.4f}")
    
    return prefix_tuning


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_LOG_MODEL"] = "all"  # log your models

    if 'llama-2-7b' in args.model_checkpoints.lower():
        model_name = 'llama-2-7b-hf'
    elif 'llama-2-13b' in args.model_checkpoints.lower():
        model_name = 'llama-2-13b-hf'
    elif 'opt-1.3b' in args.model_checkpoints.lower():
        model_name = 'opt-1.3b'

    os.environ["WANDB_PROJECT"] = f'pi_prefix_{model_name}_{args.dataset.lower()}' 

    data_path = get_data_path(args, args.dataset)

    model, base_model, tokenizer = get_pi_prefix_model(args.model_checkpoints, args.prefix_length, args.prefix_hidden_size)
    dataset, collator = get_dataset_and_collator(data_path, tokenizer, args.forget_size, args.max_length)

    if args.logits_path is None:
        args.logits_path = f'saved_logits/{model_name}_{args.dataset.lower()}-{args.forget_size}.pkl'
    if not os.path.exists(args.logits_path):
        print("Saving original logits from base model...")
        original_logits = get_logits_from_base_model(base_model, collator, dataset)
        with open(args.logits_path, "wb") as f:
            pickle.dump({k.item(): v.cpu().numpy() for k, v in original_logits.items()}, f)
    with open(args.logits_path, "rb") as f:
        original_logits = pickle.load(f)

    if args.output_path is None:
        args.output_path = f'unlearn_checkpoints/pi_prefix_{model_name}_{args.dataset.lower()}-{args.forget_size}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        os.makedirs(args.output_path, exist_ok=True)
        with open(os.path.join(args.output_path, 'args.txt'), 'w') as f:
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")

    start = time.perf_counter()
    trained_prefix = train_prefix_manual(model, base_model, tokenizer, dataset, collator, args)
    print(f"Training completed in {time.perf_counter() - start:.2f}s")

    # Save the trained prefix
    prefix_save_path = os.path.join(args.output_path, 'trained_prefix.pt')
    torch.save(trained_prefix.state_dict(), prefix_save_path)

    # Apply PI transformation
    trained_prefix_inverse = apply_pinv_to_prefix(trained_prefix)

    # Save base model
    base_model.save_pretrained(args.output_path)

    # Save tokenizer
    tokenizer.save_pretrained(args.output_path)

    # Save prefix module separately
    prefix_save_path = os.path.join(args.output_path, "prefix_tuning.pt")
    torch.save(trained_prefix_inverse.state_dict(), prefix_save_path)

    # Save model metadata
    with open(os.path.join(args.output_path, "prefix_meta.json"), "w") as f:
        import json
        json.dump({
            "prefix_length": args.prefix_length,
            "hidden_size": trained_prefix_inverse.hidden_size,
            "num_layers": trained_prefix_inverse.num_layers
        }, f)


if __name__ == "__main__":
    args = get_args()
    main(args)