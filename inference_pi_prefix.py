import os
import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DataCollatorForCompletionOnlyLM

from utils_pi import (
    PrefixTuning,
    inject_prefix_to_model,
    apply_pinv_to_prefix,
    get_data_path,
)


@torch.no_grad()
def extract_class_label(tokenizer, token_ids):
    """Decodes tokens and maps sentiment words to class labels."""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids[token_ids != -100]
        token_ids = token_ids.tolist()

    decoded = tokenizer.decode(token_ids, skip_special_tokens=True).lower()

    if "positive" in decoded:
        return 1
    elif "negative" in decoded:
        return 0
    else:
        return -1  # Unknown or invalid


def evaluate(model, prefix_module, tokenizer, dataloader, device):
    model.eval()
    model.to(device)

    inject_prefix_to_model(model, prefix_module)
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False
            )

            pred_ids = torch.argmax(outputs.logits, dim=-1)
            
            for pred, label in zip(pred_ids, labels):
                pred_label = extract_class_label(tokenizer, pred)
                true_label = extract_class_label(tokenizer, label)

                if pred_label != -1 and true_label != -1:
                    all_preds.append(pred_label)
                    all_labels.append(true_label)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    print("\nConfusion Matrix:")
    print("               Predicted")
    print("             |    0    |    1    |")
    print("         --------------------------")
    print(f"Actual  0    |  {cm[0][0]:>5}  |  {cm[0][1]:>5}  |")
    print(f"        1    |  {cm[1][0]:>5}  |  {cm[1][1]:>5}  |")

    return acc, f1

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoints", type=str, required=True)
    parser.add_argument("--prefix_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--prefix_hidden_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    return parser.parse_args()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(args.model_checkpoints).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoints, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    # Load trained prefix
    num_layers = len(base_model.model.decoder.layers)
    prefix_module = PrefixTuning(
        base_model=base_model.model,
        config=base_model.config,
        num_layers=num_layers,
        prefix_length=args.prefix_length,
        hidden_size=args.prefix_hidden_size
    ).to(device)
    prefix_module.load_state_dict(torch.load(args.prefix_path, map_location=device))
    prefix_module.eval()

    # Apply pseudo-inverse for unlearning
    apply_pinv_to_prefix(prefix_module)

    # Prepare collator and datasets
    response_template = "### Sentiment:"
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

    def prompt_template(text, label):
        return f"### Text: {text}\n\n### Question: What is the sentiment of the given text?\n\n### Sentiment: {label}"

    def preprocess(example):
        return tokenizer(prompt_template(example["text"], example["label_text"]),
                         truncation=True, max_length=args.max_length)

    data_path = get_data_path(args, args.dataset)
    dataset = load_dataset(data_path)

    # Evaluate on all 4 splits
    for split in ["train_forget", "train_forget","test_retain", "test_forget"]:
        print(f"\nEvaluating on `{split}` split...")
        data_split = dataset[split].map(preprocess).remove_columns(["text", "label", "label_text"])
        data_split.set_format("torch")

        dataloader = torch.utils.data.DataLoader(data_split,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 collate_fn=collator)

        acc, f1 = evaluate(base_model, prefix_module, tokenizer, dataloader, device)
        print(f"{split.upper()} → Accuracy: {acc:.4f} | F1: {f1:.4f}")


if __name__ == "__main__":
    args = get_args()
    main(args)