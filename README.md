
# Fashion Product Recommender — Two-Tower Architecture

This repository implements a production-ready, notebook-driven recommender system for Amazon Fashion products using a modern two-tower deep learning architecture. The project is fully modularized for reproducibility, extensibility, and GitHub best practices.

## Project Overview

**Goal:** Build a scalable recommender system that leverages both user interaction history and multimodal product features (text and images) to provide high-quality product recommendations.

**Key Features:**
- End-to-end pipeline: data loading, k-core filtering, embedding, model training, evaluation, and qualitative analysis
- Two-tower neural architecture: user tower (GRU over history), item tower (text + image fusion)
- Modern Python packaging, CLI, config, and artifact validation
- Example tests and diagnostics for model health

## Pipeline Flow

1. **Data Loading & Preprocessing**
  - Download Amazon Fashion metadata and reviews
  - Apply k-core filtering (default: k=3) to ensure dense user/item interactions
  - Clean and structure product catalog

2. **Item Embeddings**
  - Encode product titles and categories using Sentence Transformers (all-mpnet-base-v2)
  - Prepare item embedding matrix and tokenized text for model input

3. **User Interaction Sequences**
  - Build user histories from filtered events
  - Split into train/validation/novelty sets for robust evaluation

4. **Model Architecture**
  - **User Tower:** GRU encodes user history into a dense vector
  - **Item Tower:** Fuses text and image features for each product
  - Contrastive loss with pop-pool negative sampling

5. **Training & Evaluation**
  - Train with pop-pool contrastive objective
  - Evaluate with Recall@K, NDCG@K, and MRR on multiple validation splits
  - Use FAISS for fast nearest-neighbor retrieval

6. **Qualitative & Interactive Analysis**
  - Show example predictions and allow interactive playground for custom user histories

## Repository Structure

```text
.
├── recotwotower.ipynb         # Main notebook pipeline
├── scripts/
│   ├── train.py               # CLI entry: training
│   └── evaluate.py            # CLI entry: evaluation
├── src/
│   └── recommender/
│       ├── __init__.py
│       ├── cli.py             # CLI logic
│       ├── config.py          # Pydantic config
│       ├── logging_utils.py   # Logging setup
│       ├── pipeline.py        # Artifact checks, summary
│       └── utils.py           # Utilities
├── tests/
│   └── test_config.py         # Example unit tests
├── .github/workflows/ci.yml   # GitHub Actions CI
├── pyproject.toml             # Packaging, dependencies
├── requirements.txt           # Pinned requirements
├── .env.example               # Example environment config
├── Makefile                   # Dev commands
├── .pre-commit-config.yaml    # Lint/format hooks
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## Quickstart

1. **Create environment and install:**
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -U pip
  pip install -e .[dev]
  ```

2. **Optional: set up environment variables**
  ```bash
  cp .env.example .env
  ```

3. **Run checks and tests:**
  ```bash
  reco check
  reco summary
  pytest
  ```


## Notes

- Default notebook settings: `DENSE_K = 3`, `SEQ_LEN = 15`
- Artifact checks expect these files in repo root:
  - `recotwotower.ipynb`
  - `item_index_v11.faiss`
  - `item_tower_vecs_v11.npy`

## GitHub Readiness Checklist

- [x] Source package under `src/`
- [x] Reproducible dependency definitions
- [x] CLI and script entry points
- [x] Unit tests + CI
- [x] Ignore rules for local/cache/artifact files
