# multimodal-fashion-recommender

**Live demo:** https://huggingface.co/spaces/htinos/multimodal-fashion-recommender
(`POST /recommend` with `{"history": ["Swim Trunk", "Sunglasses"], "top_k": 5}`)

This repository implements a production-ready, multimodal recommender system for Amazon Fashion products using a modern two-tower deep learning architecture. The system leverages both text and image features for each product, enabling richer and more accurate recommendations than unimodal approaches. The project is fully modularized for reproducibility, extensibility, and GitHub best practices.

## Project Overview (Multimodal)

**Goal:** Build a scalable recommender system that leverages both user interaction history and multimodal product features (text and images) to provide high-quality product recommendations. The multimodal approach fuses textual and visual information, allowing the model to understand products more holistically and deliver superior recommendations.

**Key Features (Multimodal):**
- End-to-end pipeline: data loading, k-core filtering, multimodal embedding (text + image), model training, evaluation, and qualitative analysis
- Two-tower neural architecture: user tower (transformer encoder over history), item tower (fuses text and CLIP image features)
- Multimodal product representation for richer, more robust recommendations
- Modern Python packaging, CLI, config, and artifact validation
- Example tests and diagnostics for model health

## Pipeline Flow

1. **Data Loading & Preprocessing**
  - Download Amazon Fashion metadata and reviews
  - Apply k-core filtering (default: k=3) to ensure dense user/item interactions
  - Clean and structure product catalog

2. **Item Embeddings (Multimodal: Text + Image)**
  - Encode product titles and categories using Sentence Transformers (all-mpnet-base-v2)
  - Extract visual features from product images using CLIP
  - Fuse text and image features for each product to create a multimodal embedding
  - Prepare item embedding matrix and tokenized text for model input

3. **User Interaction Sequences**
  - Build user histories from filtered events
  - Split into train/validation/novelty sets for robust evaluation

4. **Model Architecture (Multimodal)**
  - **User Tower:** Transformer encoder over user history into a dense vector
  - **Item Tower:** Fuses text and image features for each product using a multimodal approach
  - Contrastive loss with pop-pool negative sampling

5. **Training & Evaluation (Multimodal)**
  - Train with pop-pool contrastive objective on multimodal embeddings
  - Evaluate with Recall@K, NDCG@K, and MRR on multiple validation splits
  - Use FAISS for fast nearest-neighbor retrieval

6. **Qualitative & Interactive Analysis**
  - Show example predictions and allow interactive playground for custom user histories

## Repository Structure (Multimodal)

```text
.
├── scripts/
│   ├── train.py               # CLI entry: training
│   └── evaluate.py            # CLI entry: evaluation
├── src/
│   └── recommender/
│       ├── __init__.py
│       ├── api.py             # FastAPI serving layer (/health, /status, /recommend)
│       ├── cli.py             # CLI logic
│       ├── config.py          # Pydantic config
│       ├── data.py            # Raw data download, k-core filtering, sequences
│       ├── models.py          # Two-tower architecture (user/item towers)
│       ├── retrieval.py       # FAISS retrieval + ranking metrics
│       ├── logging_utils.py   # Logging setup
│       ├── pipeline.py        # Public API wrapper
│       ├── train.py           # Training/eval/recommend orchestration
│       └── utils.py           # Utilities
├── tests/
│   ├── conftest.py            # Synthetic offline dataset fixture
│   ├── test_config.py         # Example unit tests
│   └── test_integration_pipeline.py
├── .github/workflows/ci.yml   # GitHub Actions CI
├── pyproject.toml             # Packaging, dependencies
├── requirements.txt           # Pinned requirements
├── .env.example               # Example environment config
├── Makefile                   # Dev commands
├── Dockerfile                 # Container image for the FastAPI service
├── docker-compose.yml         # Local run with a mounted data volume
├── .pre-commit-config.yaml    # Lint/format hooks
├── MODEL_CARD.md              # Architecture, data, results, ablations
├── LICENSE                    # MIT License
└── README.md                  # This file
```


## How to Run 
You can run the full multimodal pipeline and all major steps from the command line using the provided Python scripts and CLI:

**Universal entry point:**

```bash
python main.py <command>
# where <command> is one of: check, summary, train, evaluate
```

Or use the CLI directly:

```bash
reco check
reco summary
reco train
reco evaluate
```


All major steps (artifact check, summary, training, evaluation) are available via main.py. This is the recommended way to run the project for reproducibility and automation.

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


## Serving (FastAPI + Docker)

Once you have trained artifacts (`reco train`), you can serve recommendations
over HTTP:

```bash
pip install -e ".[api]"
uvicorn recommender.api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"history": ["Swim Trunk", "Sunglasses", "Flip Flop"], "top_k": 5}'
```

Or run the whole thing in a container, mounting your data/artifacts directory:

```bash
docker compose up --build
```

`GET /health` is a plain liveness check; `GET /status` reports which trained
artifacts are present; `POST /recommend` returns 503 (rather than silently
kicking off a multi-minute training run) if the model hasn't been trained yet.

## Notes

- Default settings: `DENSE_K = 3`, `SEQ_LEN = 15`
- Data and artifacts default to `<repo_root>/data` (override with `RECO_DRIVE_DIR`). Artifact
  checks expect these files there:
  - `item_index_v11.faiss` (multimodal index)
  - `item_tower_vecs_v11.npy` (multimodal item vectors)
- If you install the `train` extra and see the process abort during training/evaluation
  with an OpenMP error, it's a known conflict between the OpenMP runtimes bundled in
  `torch` and `faiss-cpu`. It's already worked around internally via `KMP_DUPLICATE_LIB_OK`,
  but if you hit it anyway, set `KMP_DUPLICATE_LIB_OK=TRUE` in your shell before running.



## Example Usage

You can use the recommender system programmatically in Python to get recommendations for a custom user history. For example:

```python
from src.recommender.pipeline import recommend_for_history

my_history = ["Swim Trunk", "Sunglasses", "Flip Flop"]
recommendations = recommend_for_history(my_history)
for rec in recommendations:
  print(rec)
```

Replace `my_history` with any list of product names or IDs representing a user's interaction history. The function `recommend_for_history` should return the top recommended products for the given history.

You can also run the main pipeline steps using the provided scripts or CLI commands.


## Model Card

See [MODEL_CARD.md](MODEL_CARD.md) for architecture details, real-data
training/evaluation results (Recall@10/NDCG@10/MRR@10), ablation studies
(sequence-aware vs. mean-pooling retrieval, real vs. fallback embeddings,
CLIP image embeddings), and known limitations.

## Repository Name

**GitHub:** [Sonith-Bingi/multimodal-fashion-recommender](https://github.com/Sonith-Bingi/multimodal-fashion-recommender)


