# Pseudo-Inverse Prefix Tuning (PI-Prefix) for Effective Unlearning in LLMs  

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]() [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()  

## 📌 Overview  
This repository contains the implementation of **Pseudo-Inverse Prefix Tuning (PI-Prefix)**, a parameter-efficient fine-tuning approach for **machine unlearning in Large Language Models (LLMs)**.  

PI-Prefix enables **targeted forgetting** of specific training data by learning prefix parameters on the forget set and applying a **pseudo-inverse transformation** to remove their influence—without requiring access to the retain dataset.  

Key highlights:  
- **Effective forgetting**: Forget set accuracy → near random.  
- **Retention without retain data**: Preserves generalization without accessing retained samples.  
- **Scalable & interpretable**: Efficient, parameter-light unlearning.  

---

## 📂 Datasets  
We evaluate our method on:  
- **SST-2** (Stanford Sentiment Treebank)  
- **Yelp Polarity**  

Datasets are automatically handled via the Hugging Face `datasets` library.  
Forget/retain splits used in experiments are available in:  
- `sst2/`  
- `yelp/`  

---

## ⚙️ Installation  

### Dependencies  
```bash
pip install -r requirements.txt
```

> ✅ Results are reported with **Python 3.10.14**.  

---

## 🚀 Usage  

### 1. Fine-tuning with QLoRA (pre-training before unlearning)  
```bash
python qlora.py --dataset=sst2 --model_name=facebook/opt-1.3b        --max_length=1024 --set_pad_id --lr=1e-4        --train_batch_size=32 --eval_batch_size=32 --num_epochs=2        --weight_decay=0.001        --lora_rank=16 --lora_alpha=64 --lora_dropout=0.1 --lora_bias=none
```

---

### 2. Baseline Unlearning Methods  
```bash
python baselines.py --dataset=sst2 --model_checkpoints=[path_to_checkpoints]        --max_length=1024 --set_pad_id --lr=1e-4        --train_batch_size=32 --eval_batch_size=32 --num_epochs=10        --weight_decay=0.001 --unlearn_method=gradient_ascent
```

Supported baselines:  
- `gradient_ascent`  
- `random_label -- Not included in paper` 
- `gradient_ascent_kl`  
- `gradient_ascent_descent`  

---

### 3. PI-Prefix Unlearning  
```bash
python pi_prefix.py --dataset=sst2 --model_checkpoints=[path_to_checkpoints]        --max_length=1024 --set_pad_id --lr=1e-4        --train_batch_size=32 --eval_batch_size=32 --num_epochs=10        --weight_decay=0.001        --forget_size=1.0 --ptuning_num_tokens=20        --ptuning_hidden_size=128 --alpha=0.5 --beta=0.5
```

⚠️ `model_checkpoints` should point to QLoRA fine-tuned checkpoints.  
To directly train prefix tokens on pre-trained LLMs, **skip loading QLoRA parameters**.  

---

### 4. Inference with PI-Prefix  
```bash
python inference_pi_prefix.py --dataset=sst2        --model_checkpoints=[path_to_pi_prefix_checkpoints]        --forget_size=1.0
```

---

## ⏱️ Runtime Comparison  
(All values are reported in **minutes**.) 

| Dataset   | Method | Train Forget Runtime | Train Retain Runtime | Test Forget Runtime | Test Retain Runtime |
| --------- | ------ | -------------------- | -------------------- | ------------------- | ------------------- |
| **SST-2** | QLoRA  | 5.52                 | 151.25               | 2.49                | 65.04               |
|           | GA     | 5.15                 | 145.85               | 2.32                | 62.45               |
|           | GAGD   | 5.11                 | 144.84               | 2.30                | 62.05               |
|           | GAKL   | 5.15                 | 148.39               | 2.32                | 63.80               |
|           | PIPT   | 6.47                 | -               | 2.91                | 78.30               |
| **Yelp**  | QLoRA  | 37.63                | 729.37               | 16.99               | 315.40              |
|           | GA     | 35.51                | 693.13               | 16.00               | 299.05              |
|           | GAGD   | 35.63                | 694.31               | 16.05               | 299.55              |
|           | GAKL   | 36.17                | 689.44               | 16.30               | 297.45              |
|           | PIPT   | 44.68                | -               | 20.10               | 372.64              | 

## 📄 Citation

---

## 🤝 Acknowledgements  
- [Hugging Face Transformers](https://github.com/huggingface/transformers)  
- [Hugging Face Datasets](https://github.com/huggingface/datasets)  
- [PEFT Library](https://github.com/huggingface/peft)  
- Inspiration from prior work on parameter-efficient fine-tuning & unlearning.  
- Some utilities adapted from [karuna-bhaila/llm_unlearning](https://github.com/karuna-bhaila/llm_unlearning).  
