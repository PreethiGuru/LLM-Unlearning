# LLM-Unlearning
This repository contains the implementation of Pseudo-Inverse Prefix Tuning (PI-Prefix), a parameter-efficient fine-tuning approach for machine unlearning in Large Language Models (LLMs). 

# Pseudo-Inverse Prefix Tuning (PI-Prefix) for Effective Unlearning in LLMs  

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]() [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()  

## 📌 Overview  
This repository contains the implementation of **Pseudo-Inverse Prefix Tuning (PI-Prefix)**, a parameter-efficient fine-tuning approach for **machine unlearning in Large Language Models (LLMs)**.  

PI-Prefix enables **targeted forgetting** of specific training data by learning prefix parameters on the forget set and applying a **pseudo-inverse transformation** to remove their influence—without requiring access to the retain dataset.  

Our experiments on **SST-2** and **Yelp** sentiment classification demonstrate that PI-Prefix:  
- Achieves **effective forgetting** (forget set accuracy → near random).  
- **Preserves generalization** on retain data even without accessing it.  
- Provides a **scalable and interpretable unlearning** solution.  

## 🚀 Installation  
```bash
git clone https://github.com/<your-username>/PI-Prefix.git
cd PI-Prefix
pip install -r requirements.txt
