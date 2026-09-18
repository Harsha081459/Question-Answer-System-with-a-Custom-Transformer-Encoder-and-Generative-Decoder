# 📚 Question-Answering System with a Custom Transformer Encoder and Generative Decoder

A complete, end-to-end Question-Answering system built **from scratch** using custom Transformer architectures. This project implements both **Extractive QA** (span-based answer selection) and **Generative QA** (free-form answer generation) pipelines, trained on the SQuAD dataset and deployed as a premium web application.

<p align="center">
  <a href="https://github.com/Harsha081459/Question-Answer-System-with-a-Custom-Transformer-Encoder-and-Generative-Decoder/actions/workflows/ci.yml">
    <img src="https://github.com/Harsha081459/Question-Answer-System-with-a-Custom-Transformer-Encoder-and-Generative-Decoder/actions/workflows/ci.yml/badge.svg" alt="CI"/>
  </a>
  &nbsp;
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
├── generative_evaluation.py            # Phase 3: EM/F1/ROUGE-L/BLEU evaluation
├── generative_inference.py             # Phase 3: Generative QA inference
│
├── compare_extractive_models.py        # Model comparison scripts
├── compare_generative_models.py        # Model comparison scripts
├── generate_report_figures.py          # Report figure generation
├── generate_all_report_figures.py      # All report figures
├── generate_encoder_training_dynamics.py # Training dynamics visualization
│
├── Dockerfile                          # HuggingFace Spaces deployment
├── requirements.txt                    # Python dependencies
├── README_GENERATIVE_DECODER.md        # Detailed generative QA documentation
├── README_MLM_PRETRAINING.md           # Detailed pre-training documentation
│
├── NLP project.pdf                     # Project specification
├── report.pdf                          # Final project report
└── report.tex                          # LaTeX source for the report
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12 (tested locally and in CI)
- CPU inference is supported; training requires substantially more resources.

### Installation
```bash
git clone https://github.com/Harsha081459/Question-Answer-System-with-a-Custom-Transformer-Encoder-and-Generative-Decoder.git
cd Question-Answer-System-with-a-Custom-Transformer-Encoder-and-Generative-Decoder

# Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Run Locally
```bash
python download_models.py
python download_models.py --check
uvicorn app:app --host 127.0.0.1 --port 7860
```
Then open [http://localhost:7860](http://localhost:7860) in your browser.

> `download_models.py` fetches the public FP16 artifacts from `hv-123/QA-Engine`
> at fixed revision `7fda22dce2a8855a49e55e7016c1547dc3d0cd94` into `models_fp16/`.
> Downloads need internet; serving uses local files. `/health` is liveness and
> `/ready` is 200 only after both checkpoints load. Without weights the UI still
> opens, but `/predict` returns an actionable 503. Corrupt/incompatible weights
> are startup failures, not silently substituted models.

Try extractive mode with context `Python was created by Guido van Rossum and released in 1991.` and question `Who created Python?`. The checkpoint-backed regression test expects `Guido van Rossum`. Long passages use padded overlapping windows. Requests have bounded text and decoding sizes; predictions are serialized to limit memory pressure.

The **generative head is experimental**: the published checkpoint can produce its no-answer template even for answerable questions, including the example above. This is a model-quality limitation, not a successful answer. The API exposes `predicted_no_answer` in addition to gate scores.

For Docker: run the downloader first, then `docker build -t custom-qa .` and `docker run --rm -p 127.0.0.1:7860:7860 custom-qa`. The image includes locally downloaded weights; do not publish it unintentionally. GitHub does not redeploy the separate HF Space automatically.

### Tests
```bash
pip install -r requirements-dev.txt
pytest -q
```
The default suite runs on CPU with tiny model configs and checks API validation, missing-model readiness and long-passage inference. No checkpoints, datasets or network are required. The separate CI smoke job downloads the fixed artifacts and tests both real heads.

After downloading weights, run the smoke test with `RUN_QA_SMOKE=1 python -m pytest tests/test_checkpoint_smoke.py -q` (PowerShell: `$env:RUN_QA_SMOKE='1'` before the command). This is an inference regression, not a full SQuAD benchmark.

---

## 📊 Model Performance

### Extractive QA (SQuAD v2)

Historical results below were transcribed into the original project report; they are not freshly reproduced by CI. The downloadable `models_fp16/extractive/squad_v2_threshold_tuning.json` is stronger provenance for the custom checkpoint: on 11,873 validation examples it records **52.42 EM / 56.02 F1 at threshold 0**, and **55.71 EM / 58.16 F1 at threshold -1.752252** selected on that validation set. This supports a rounded **58.2 threshold-tuned validation F1**, not an untouched test-set result. The earlier 55.8 EM entry is retained in the historical report but differs from the archived 55.71.

The comparison models are differently pretrained/fine-tuned systems, so these are not controlled architecture ablations. Raw cross-model comparison JSON files are not committed; superiority claims need a fresh reproducible evaluation.

| Model | EM | F1 |
|-------|----|----|
| RoBERTa-base-SQuAD2 *(ceiling reference)* | 80.4 | 83.3 |
| BERT-large-finetuned | 66.8 | 68.3 |
| **Our scratch encoder (archived tuned validation)** | **55.71** | **58.16** |
| DistilBERT-distilled-SQuAD | 54.6 | 55.3 |

### Generative QA

The original report describes the following results on a 1k validation subset. The raw prediction/evaluation files and exact run manifest are not committed. Treat these as **historical reported values**, not verified performance of the downloadable deployment checkpoint; its smoke test does not reproduce this table.

| Model | EM | F1 | ROUGE-L | BLEU |
|-------|----|----|---------|------|
| T5-base | 35.8 | 42.1 | 42.2 | 15.4 |
| **Our generative decoder** | **34.1** | **39.9** | **40.8** | **23.1** |
| T5-small | 31.7 | 38.1 | 38.2 | 13.4 |
| Flan-T5-small | 26.4 | 32.0 | 32.2 | 10.7 |

The report used greedy decoding, `max_new_tokens=12` and length penalty 0.4. Reproducing comparisons requires matching checkpoints, subsets and no-answer scoring. The provided T5 baselines are loaded without task-specific fine-tuning in the comparison script, so do not interpret the table as general superiority over T5.

### Pre-training convergence

Final validation perplexity after 20k MLM steps: **5.15**. Decoder fine-tuning loss
converged smoothly from 0.906 to 0.713.

### Known limitation

No-answer calibration on the **generative** head is the weakest part of the system: the
log-probability thresholding mechanism does not reliably separate genuinely unanswerable
questions from low-confidence answers, so we do not report a standalone no-answer accuracy
for it. An ablation that upweighted unanswerable examples 3× made this worse, not better
(F1 39.9 → 37.4), which is why the uniform-weight model was kept as the final configuration.
The extractive head's CLS null-threshold gating is the more dependable of the two.

### Reproducing the numbers

| Results table | Script |
|---|---|
| Extractive QA (our encoder vs. RoBERTa-base-SQuAD2, BERT-large, DistilBERT on the full SQuAD v2 validation set) | `compare_extractive_models.py` |
| Generative QA (our decoder vs. T5-small, T5-base, Flan-T5-small on the shared 1k validation subset) | `compare_generative_models.py` |
| Generative EM/F1/ROUGE-L/BLEU for a single checkpoint | `generative_evaluation.py` |

The comparison scripts download SQuAD and selected public baselines; `--output_json` and `--output_csv` select their outputs. The generative comparison defaults to **4,000** examples and an older training checkpoint; it does not reproduce the historical 1k table by default. Use `--help`, specify the intended checkpoint/subset and preserve the full run configuration.

To evaluate only the downloaded extractive checkpoint, use:

```bash
python compare_extractive_models.py --custom_model_dir models_fp16/extractive --custom_config_dir models_fp16/extractive --hf_models "" --max_eval_examples 100
```

Remove `--max_eval_examples 100` for full validation. This can be expensive on CPU. Do not label a 100-example smoke evaluation as full-set performance.

---

## 🔬 Training Pipeline

### Phase 1: Pre-training (MLM)
Pre-trained a custom Transformer encoder from scratch on Wikipedia using Masked Language Modeling:
```bash
python mlm_pretraining.py --out_dir checkpoints_pretrain_base_seq256 --seq_len 256 --max_steps 20000
```

### Phase 2: Extractive QA
Fine-tuned the pre-trained encoder on SQuAD v2 for span-based question answering:
```bash
python extractive_finetuning.py --checkpoint_dir checkpoints_pretrain_base_seq256/step_20000 --dataset squad_v2 --output_dir checkpoints_qa_squad --max_length 256 --doc_stride 64
```

### Phase 3: Generative QA
Trained a custom hybrid decoder on top of the encoder using a multi-stage curriculum:
```bash
python generative_finetuning.py \
  --decoder_variant hybrid \
  --tokenizer_path checkpoints_pretrain_base_seq256/step_20000 \
  --pretrain_ckpt checkpoints_pretrain_base_seq256/step_20000/checkpoint.pt \
  --output_dir checkpoints_generative_qa
```

See [README_GENERATIVE_DECODER.md](README_GENERATIVE_DECODER.md) for the full multi-stage training recipe.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Deep Learning** | PyTorch, Transformers (tokenizer only) |
| **Models** | Custom Transformer Encoder + Hybrid Decoder |
| **Training Data** | SQuAD v1, SQuAD v2, Wikipedia |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | HTML5, CSS3 (Glassmorphism), Vanilla JS |
| **Deployment** | Docker, HuggingFace Spaces |
| **Evaluation** | EM, F1, ROUGE-L, BLEU |

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
