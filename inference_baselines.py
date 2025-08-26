import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # Use only GPU 5

import numpy as np
from argparse import ArgumentParser
from copy import deepcopy

import torch
from torchinfo import summary

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from utils import get_data_path, compute_metrics, preprocess_logits_for_metrics, CustomCallback

POS_WEIGHT, NEG_WEIGHT = (1.0, 1.0)

def get_args():
    parser = ArgumentParser(description="Run inference on merged QLoRA model")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--model_checkpoints", type=str, required=True, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--output_path", type=str, required=False, help="Output path for evaluation")
    return parser.parse_args()


def load_model_and_tokenizer(model_checkpoint, max_length):
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, device_map="auto", trust_remote_code=True)

    for param in model.parameters():
        param.requires_grad = False

    # OPT models need use_fast=False for tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Tokenizer and model loaded successfully.")
    summary(model)

    return model, tokenizer



def get_dataset_and_collator(dataset_name, tokenizer, max_length):
    prompt_template = lambda text, label: f"""### Text: {text}\n\n### Question: What is the sentiment of the given text?\n\n### Sentiment: {label}"""

    def _preprocessing_sentiment(examples):
        return {"text": prompt_template(examples['text'], examples['label_text'])}

    data = load_dataset(get_data_path(args, dataset_name))
    data = data.map(_preprocessing_sentiment, batched=False)
    data = data.remove_columns(['label', 'label_text'])
    data.set_format("torch")

    response_template = "\n### Sentiment:"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    return data, collator


def main(args):
    if 'opt-1.3b' in args.model_checkpoints.lower():
        model_name = 'opt-1.3b'
    else:
        raise ValueError("Unsupported model. Please confirm model is OPT-1.3b or update the tokenizer load line.")

    os.environ["WANDB_LOG_MODEL"] = "all"
    os.environ["WANDB_PROJECT"] = f'inference_qlora_{model_name}_{args.dataset}'

    param_path = os.path.dirname(args.model_checkpoints)
    with open(os.path.join(param_path, 'arguments.txt'), 'r') as f:
        parameters = f.readlines()
        params = {k.strip(): v.strip() for k, v in (line.split(':') for line in parameters)}

    max_length = int(params['max_length'])

    model, tokenizer = load_model_and_tokenizer(args.model_checkpoints, max_length)

    dataset, collator = get_dataset_and_collator(
        args.dataset,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=float(params['lr']),
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=int(params['train_batch_size']),
        per_device_eval_batch_size=int(params['eval_batch_size']),
        num_train_epochs=int(params['num_epochs']),
        weight_decay=float(params['weight_decay']),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        group_by_length=True,
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        fp16=True,
        report_to="wandb",
        run_name=f'epoch={params["num_epochs"]}_lr={params["lr"]}',
        max_grad_norm=0.3,
    )

    if params.get('set_pad_id', 'False') == 'True':
        model.config.pad_token_id = model.config.eos_token_id

    if model.device.type != 'cuda':
        model = model.to('cuda')

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        dataset_text_field='text',
        max_seq_length=max_length,
        tokenizer=tokenizer,
        train_dataset=None,
        eval_dataset={
            "train_retain": dataset['train_retain'],
            "train_forget": dataset['train_forget'],
            "test_retain": dataset['test_retain'],
            "test_forget": dataset['test_forget'],
        },
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(CustomCallback(trainer))
    trainer.evaluate()


if __name__ == "__main__":
    args = get_args()
    main(args)
