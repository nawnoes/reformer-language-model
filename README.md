# Reformer Language Models (Korean)

Korean language model experiments using **Reformer** (2020) implementations from `lucidrains/reformer-pytorch`.

This repository is a research/learning snapshot that explores three training objectives:

- **Masked Language Modeling (MLM)** (BERT-style)
- **Autoregressive Language Modeling** (causal LM)
- **Replaced Token Detection (RTD)** (ELECTRA-style)

Korean README: `README.ko.md`

## Reformer overview (why Reformer)

Transformers have well-known memory/time bottlenecks for long sequences:

- **Self-attention cost**: full attention is `O(L^2)` in time and memory for sequence length `L`.
- **Activation memory**: deep stacks of layers require storing activations for backpropagation.
- **Large feed-forward blocks**: FFN activations can dominate memory in practice.

Reformer attempts to improve scalability using ideas such as:

- **LSH attention** to approximate full attention and reduce complexity (often described as `O(L log L)`).
- **Reversible residual layers (RevNet)** to trade activation memory for recomputation.
- **Chunked feed-forward** to reduce peak memory usage.
- **Axial positional embeddings** for long sequences.

## Data

- **Korean Wikipedia** (used to build the training corpus)

## Tokenizer / vocab

This project uses a **WordPiece** tokenizer (BPE-family subword tokenizer).

- See `vocab/README.md` for how the WordPiece vocabulary was generated and tested.

## Environment (snapshot)

- Google Colab Pro
- NIPA GPU resources
  - GPU quota: 10 TF
  - GPU: RTX 6000 (24 GB)

## Setup

```bash
pip install -r requirements.txt
```

## What’s included

### 1) Masked Language Modeling (MLM)

Goal: pretrain a BERT-style model with masked token prediction (without NSP/SOP in this project).

- Notebook: `reformer-mlm-pretraining.ipynb`
- Script: `pretrain/mlm-model.py`
- Example figure: `images/mlm.png`

Example configuration used in this repo (“Reformer-mlm-small”, roughly half of BERT-base depth):

```txt
max_len = 512
batch_size = 128
dim = 512
depth = 6
heads = 8
causal = False
```

### 2) Autoregressive language modeling

Goal: train a causal language model for next-token prediction / text generation.

- Notebook: `reformer-autoregressive-pretraining.ipynb`
- Script: `pretrain/autoregressive-model.py`
- Examples: `example/train-auto-regressive.py`, `example/autoregressive_model_text_generation.py`

### 3) Replaced Token Detection (ELECTRA-style)

Goal: train a discriminator to detect whether each token is “original” or “replaced”, using a generator/discriminator setup.

- Notebook: `reformer-electra-pretraining.ipynb`
- Script: `pretrain/electra-model.py`
- Example: `example/electra_train_example.py`
- Pretraining plot: `images/electra_loss_graph_1_epoch.png`

## Quickstart (best-effort)

1. Install dependencies: `pip install -r requirements.txt`
2. Prepare your corpus and vocabulary (see `vocab/README.md`).
3. Check configs under `config/` and data under `data/`.
4. Run one of the pretraining scripts:

```bash
python pretrain/mlm-model.py
python pretrain/autoregressive-model.py
python pretrain/electra-model.py
```

Notes:

- This repo is not packaged as a single CLI; scripts may contain hard-coded paths and expect local files under `data/`.
- The notebooks are the most reliable starting point for reproducing the original experiments.

## Fine-tuning (KorQuAD v1.0)

This repo contains notebooks for KorQuAD fine-tuning:

- `finetuning/korquad-reformer-mlm.ipynb`
- `finetuning/korquad-reformer-electra.ipynb`

Reported snapshot results (KorQuAD v1.0):

| Model | Exact Match (EM) | F1 |
|---|---:|---:|
| Reformer-ELECTRA-small | 52.04 | 78.53 |
| KoBERT | 51.75 | 79.15 |

## Repository layout

- `pretrain/`: pretraining scripts (MLM / autoregressive / ELECTRA-style)
- `finetuning/`: KorQuAD fine-tuning notebooks
- `example/`: small runnable examples and utilities
- `vocab/`: tokenizer/vocabulary notes
- `config/`, `data/`, `dataset/`, `model/`, `util/`: supporting code and assets

## References

- “The Reformer - Pushing the limits of language modeling” (Colab): https://colab.research.google.com/drive/1MYxvC4RbKeDzY2lFfesN-CvPLKLk00CQ
- `lucidrains/reformer-pytorch`: https://github.com/lucidrains/reformer-pytorch
- `lucidrains/electra-pytorch`: https://github.com/lucidrains/electra-pytorch
- WordPiece vocab note (Korean): https://monologg.kr/2020/04/27/wordpiece-vocab/

