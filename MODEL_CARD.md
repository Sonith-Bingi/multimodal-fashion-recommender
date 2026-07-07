# Model Card: Multimodal Fashion Recommender

## Overview

A two-tower retrieval model for next-item recommendation over the Amazon
Fashion catalog. A user tower encodes interaction history sequentially; an
item tower fuses text, image, and learned ID embeddings for each product.
Both towers project into a shared 256-dim space and are trained jointly with
a contrastive objective, so retrieval at serving time is nearest-neighbor
search (FAISS) between a user vector and the item catalog.

## Architecture

- **User tower**: input projection → learned positional embeddings (max
  history length 64) → a `nn.TransformerEncoder` (not a GRU, despite older
  documentation) → an MLP head down to 256-dim, L2-normalized.
- **Item tower**: three per-item signals — a text projection, a CLIP image
  projection (gated by a learned sigmoid gate, initialized near-zero so the
  model must earn its way into using image features), and a learned item-ID
  embedding — fused via multi-head self-attention over the 3 tokens, then
  averaged and normalized.
- **Loss**: in-batch/pop-pool contrastive loss (temperature-scaled softmax
  cross-entropy against a pool of popular training-target items), with
  history items masked out of the negative pool except when they coincide
  with the true target.
- **Retrieval**: FAISS `IndexFlatIP` over L2-normalized item vectors; items
  already in a user's history are excluded from returned candidates by
  design (a fashion recommender's job here is novel-item discovery, not
  reordering).

## Data

Real Amazon Fashion data from `McAuley-Lab/Amazon-Reviews-2023` (Hugging
Face Hub), filtered with iterative k-core filtering (default `k=3`: every
retained user and item must have ≥3 interactions with the retained set).

| | raw | dense (k=3) |
|---|---|---|
| users | 2,035,398 | 5,277 |
| items | — | 5,015 |
| events | — | 24,316 |

This is an extremely sparse interaction graph even after filtering (~4.6
events/user, ~4.85 events/item) — raising `k` to 5 collapses the dense set
to 285 items / 266 users / 1,601 events, illustrating how steep k-core
percolation is on this data. `k=3` is the usable operating point.

## Training

- AdamW, lr=5e-4, weight_decay=1e-2, cosine annealing, up to 40 epochs with
  early stopping (patience 10, EMA-smoothed validation loss), 2-epoch
  temperature warmup.
- Fully deterministic given a fixed `random_seed` (default 42):
  `torch.manual_seed()` before model construction and a seeded `DataLoader`
  shuffle generator. Verified: two independent `train()` + `evaluate()`
  calls on identical data produce byte-identical metrics.
- Text embeddings: `sentence-transformers/all-mpnet-base-v2` when available,
  falling back to a deterministic hash-based encoder otherwise (keeps CI and
  offline development fast; not a substitute for real embeddings, see
  results below).
- Image embeddings: CLIP ViT-B/32 over real product images (capped at 5,000
  target items), falling back to zero vectors if unavailable.

## Evaluation methodology — and a critical correction

Recall@10 / NDCG@10 / MRR@10 are computed by holding out one interaction per
user as the target and using the rest as history. Because retrieval
excludes every item already in a user's history from its candidates (by
design — see above), **a target that happens to be a repeat of something
already in that user's history can never be retrieved, regardless of model
quality.**

The naive split (hold out the literal last interaction) hits this on
**48.8% of users at k=3** (2,576/5,277) — real Amazon Fashion data has a lot
of repeat purchases/re-reviews. Evaluating on that split silently
guarantees ~49% of examples to score exactly zero no matter how good the
model is, which both deflates absolute metrics and compresses the measured
gap between model variants (since all variants score zero on the same
unwinnable half).

The fix: evaluate only on genuinely novel targets (the most recent position
in each user's history where the held-out item isn't a repeat of anything
earlier). This is standard practice in sequential-recommendation research
(e.g. SASRec/BERT4Rec-style benchmarks typically deduplicate repeat
interactions before evaluation for exactly this reason), applied
symmetrically to every model variant below — it is not a post-hoc filter
picked because it improved a specific number.

## Results

All results below: real Amazon Fashion data, `dense_k=3`, `seed=42`
(deterministic), sequence-tower vs. a mean-of-history-item-vectors baseline
using the *same* trained item embeddings — isolating the effect of how a
user's history is aggregated into a query vector from the effect of the
item embeddings themselves.

### Headline result (real text + real CLIP, evaluated on genuinely novel targets)

| metric | sequence tower | mean-pooling baseline |
|---|---|---|
| recall@10 | **0.1695** | 0.1020 |
| ndcg@10 | **0.1132** | 0.0560 |
| mrr@10 | **0.0961** | 0.0421 |

Recall@10 of 17% against a 5,015-item catalog is ~85x better than random
chance (10/5015 ≈ 0.2%). The sequence tower beats the mean-pooling baseline
by **+66% (recall), +102% (ndcg), +128% (mrr)** relative.

### Does the sequence-aware tower actually help? Depends on the data.

| dataset | sequence tower recall@10 | mean-pool recall@10 | relative lift |
|---|---|---|---|
| synthetic (i.i.d. cluster-affinity users, no real order dependence) | 0.389 | 0.378 | ~tied |
| real Amazon Fashion (genuine purchase sequences) | 0.1695 | 0.1020 | **+66%** |

On synthetic data engineered without genuine sequential structure, a
transformer user tower has nothing extra to exploit over a naive average —
the two methods tie. On real purchase sequences, which plausibly do have
order-dependent structure (recency, evolving taste, complementary
purchases), the sequence-aware tower wins decisively. The architecture's
complexity is justified by *this* result, not by assumption.

### Does adding CLIP image embeddings help? Only measurable once training was made deterministic

Controlled ablation, identical seed, identical architecture, only the image
branch differs (all-zero vs. real CLIP vectors), evaluated **before** the
val-split fix above (so these numbers are on the naive/deflated split, but
internally comparable to each other):

| condition | recall@10 (seq) | ndcg@10 (seq) | mrr@10 (seq) |
|---|---|---|---|
| text-only | 0.0490 | 0.0272 | 0.0205 |
| text + real CLIP | **0.0560** (+14%) | **0.0284** (+4%) | 0.0201 (flat) |

An earlier, *unseeded* comparison had suggested CLIP made things worse
(recall 0.0555→0.0505). That comparison trained two separate models with no
RNG control, so part of the swing was random initialization noise, not a
real effect of adding images — a reminder that ablations are only as
trustworthy as the determinism of the training run underneath them.

## Known limitations

- Single run per condition above (no repeated-seed variance estimate);
  directions are consistent across metrics and re-runs, but not
  statistically bulletproof.
- CLIP embeddings capped at 5,000 target items and computed once; not
  re-validated at `dense_k=5` (that slice collapsed to 285 items, too small
  to be a meaningful comparison point).
- Text-embedding ablation isolates "real vs. fallback embeddings" but not
  a from-scratch fine-tuned text encoder — `all-mpnet-base-v2` is used
  frozen.
- No hyperparameter search; epoch budget, learning rate, and temperature
  bounds are fixed defaults, not tuned per condition.

## Engineering bugs found and fixed during this work

Documented here because they materially changed what the numbers above
mean, and because finding them required actually running the full pipeline
end-to-end on real data rather than trusting that the code "should" work:

1. **Trained model could never load for inference.** `register_popular_pool()`
   registers training-only buffers that a strict `load_state_dict()` at
   inference time rejected outright — every `evaluate()`/`recommend_for_history()`
   call silently fell back to mean-pooling, with no error, ever. (No CI
   coverage existed for this path until torch/faiss were added to CI.)
2. **No training determinism.** No `torch.manual_seed()` anywhere; every
   `train()` call produced a different model, so ablations were confounded
   with random initialization noise until fixed.
3. **Catalog cache silently discarded training data.** `prepare_data()`'s
   cache-hit path returned empty `user_events`, so every `train()` call
   after the first produced 0 training sequences with no error — until a
   dense-events side-car cache was added.
4. **Evaluation used the wrong split.** `evaluate()` read `splits["val"]`
   (includes repeat-purchase targets, structurally unretrievable given the
   retrieval design) instead of the already-computed `splits["val_novel"]`,
   deflating every reported metric ~2-4x.

## Reproducing these results

```bash
pip install -e ".[dev,ci]"   # or .[train] for real text/image encoders too
export RECO_DRIVE_DIR=/path/to/data   # meta_Amazon_Fashion.jsonl + Amazon_Fashion.jsonl
reco train
reco evaluate
```

Training is deterministic given the same data and `RECO_RANDOM_SEED`
(default 42).
