# Pseudo-Inverse Prefix Tuning (PI-Prefix) for Effective Unlearning in LLMs  

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]() [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()  

## 📌 Overview  
This repository contains the implementation of **Pseudo-Inverse Prefix Tuning (PI-Prefix)**, a parameter-efficient fine-tuning approach for **machine unlearning in Large Language Models (LLMs)**.  

PI-Prefix enables **targeted forgetting** of specific training data by learning prefix parameters on the forget set and applying a **pseudo-inverse transformation** to remove their influence—without requiring access to the retain dataset.  

Key highlights:  
- 🧹 **Effective forgetting**: Forget set accuracy → near random.  
- 🎯 **Retention without retain data**: Preserves generalization without accessing retained samples.  
- ⚡ **Scalable & interpretable**: Efficient, parameter-light unlearning.  

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

### 1. Clone the repository  
```bash
git clone https://github.com/your-repo/pi-prefix.git
cd pi-prefix
```

### 2. Install dependencies  
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
- `random_label`  
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

## 📊 Results  
- Forget set accuracy drops to random baseline.  
- Retain performance preserved **without access to retain dataset**.  
- Outperforms gradient-ascent style baselines in stability & efficiency.  

(Detailed results and plots are in the thesis/report.)  

---

## 📄 Citation  
If you use this code, please cite:  

```bibtex
@article{your2025piprefix,
  title={Pseudo-Inverse Prefix Tuning for Effective Unlearning in LLMs},
  author={Your Name and ...},
  year={2025},
  journal={ACL 2025 (Under Review)}
}
```  

---

## 🙏 Acknowledgements  
- [Hugging Face Transformers](https://github.com/huggingface/transformers)  
- [Hugging Face Datasets](https://github.com/huggingface/datasets)  
- [PEFT Library](https://github.com/huggingface/peft)  
- Inspiration from prior work on parameter-efficient fine-tuning & unlearning.  
- Some utilities adapted from [karuna-bhaila/llm_unlearning](https://github.com/karuna-bhaila/llm_unlearning).  
