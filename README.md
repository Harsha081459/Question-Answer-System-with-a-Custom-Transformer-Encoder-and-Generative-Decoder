# 📚 Question-Answering System with a Custom Transformer Encoder and Generative Decoder

A complete, end-to-end Question-Answering system built **from scratch** using custom Transformer architectures. This project implements both **Extractive QA** (span-based answer selection) and **Generative QA** (free-form answer generation) pipelines, trained on the SQuAD dataset and deployed as a premium web application.

<p align="center">
  <a href="https://huggingface.co/spaces/hv-123/QA-Engine">
    <img src="https://img.shields.io/badge/🤗%20Live%20Demo-QA%20Engine-blue?style=for-the-badge" alt="Live Demo"/>
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=for-the-badge&logo=pytorch" alt="PyTorch"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi" alt="FastAPI"/>
  &nbsp;
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License"/>
</p>

---

## 🌐 Live Demo

The application is deployed and publicly accessible on HuggingFace Spaces:

**🔗 [https://huggingface.co/spaces/hv-123/QA-Engine](https://huggingface.co/spaces/hv-123/QA-Engine)**

> Enter a passage of text and ask a question about it. The app uses either an extractive or a generative model to find or create the most relevant answer.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Custom Transformer Encoder** | Built from scratch with MLM pre-training on Wikipedia |
| **Extractive QA** | Fine-tuned on SQuAD v2 for span-based answer extraction |
| **Generative QA** | Custom hybrid decoder that generates free-form answers |
| **No-Answer Detection** | SQuAD v2 style — knows when a question is unanswerable |
| **Premium Web UI** | Dark mode, glassmorphism, neon glow orbs, responsive design |
| **Cloud Optimized** | Dockerized with CPU thread optimization for free-tier deployment |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Phase 1: Pre-training              │
│   Custom Transformer Encoder (MLM on Wikipedia)     │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐   ┌──────────────────────┐
│   Phase 2:      │   │   Phase 3:           │
│  Extractive QA  │   │  Generative QA       │
│  (SQuAD v2)     │   │  (Hybrid Decoder)    │
│  Span Selection │   │  Free-form Answers   │
└────────┬────────┘   └──────────┬───────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │   FastAPI Web App     │
         │   Premium Dark UI     │
         │   HuggingFace Spaces  │
         └───────────────────────┘
```

---

## 📁 Project Structure

```
├── app.py                              # FastAPI web server
├── static/                             # Frontend UI
│   ├── index.html                      # Main HTML page
│   ├── style.css                       # Premium dark mode styles
│   └── main.js                         # API interaction & AbortController
│
├── mlm_pretraining.py                  # Phase 1: Custom encoder pre-training (MLM)
│
├── extractive_finetuning.py            # Phase 2: SQuAD fine-tuning for extractive QA
├── extractive_inference.py             # Phase 2: Extractive QA inference
├── extractive_error_analysis.py        # Phase 2: Error analysis
├── extractive_threshold_tuning.py      # Phase 2: No-answer threshold tuning
│
├── main_hybrid_decoder.py              # Phase 3: Custom hybrid decoder architecture
├── standard_generative_decoder.py      # Phase 3: Standard Transformer decoder baseline
├── generative_data.py                  # Phase 3: Data pipeline for generative QA
├── generative_finetuning.py            # Phase 3: Multi-stage generative training


#                      ▼
#          ┌───────────────────────┐
#          │   FastAPI Web App     │
#          │   Premium Dark UI     │
#          │   HuggingFace Spaces  │
#          └───────────────────────┘
# ```
# 
# ---
# 
# ## 📁 Project Structure
# 
# ```
# ├── app.py                              # FastAPI web server
# ├── static/                             # Frontend UI
# │   ├── index.html                      # Main HTML page
# │   ├── style.css                       # Premium dark mode styles
# │   └── main.js                         # API interaction & AbortController
# │
# ├── mlm_pretraining.py                  # Phase 1: Custom encoder pre-training (MLM)
# │
# ├── extractive_finetuning.py            # Phase 2: SQuAD fine-tuning for extractive QA
# ├── extractive_inference.py             # Phase 2: Extractive QA inference
# ├── extractive_error_analysis.py        # Phase 2: Error analysis
# ├── extractive_threshold_tuning.py      # Phase 2: No-answer threshold tuning
# │
# ├── main_hybrid_decoder.py              # Phase 3: Custom hybrid decoder architecture
# ├── standard_generative_decoder.py      # Phase 3: Standard Transformer decoder baseline
# ├── generative_data.py                  # Phase 3: Data pipeline for generative QA
# ├── generative_finetuning.py            # Phase 3: Multi-stage generative training
