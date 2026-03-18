## Example Usage (from Notebook)

You can use the following code to interactively test the recommender with custom user histories (see notebook for full context):

```python
my_history = ["Swim Trunk", "Sunglasses", "Flip Flop"]
play_with_model(my_history)

my_history = ["Dress Shirt", "Silk Tie", "Cufflinks"]
play_with_model(my_history)

my_history = ["Leggings", "Beanie", "Gloves"]
play_with_model(my_history)

my_history = ["disney", "pixar", "kids"]
play_with_model(my_history)
```

#### Example Output

**1. ["Swim Trunk", "Sunglasses", "Flip Flop"]**

```
================================================================================
 🛒 BUILDING YOUR CUSTOM HISTORY
================================================================================
  [Found] 'Swim Trunk' -> [1156] Awesome 360 Men's Swim Trunks Beach Board Shorts, Black and Green Leaf L
  [Found] 'Sunglasses' -> [17] AMEXI Rectangle Polarized Sunglasses for Women Men Classic Vintage Retro Frame UV Protection
  [Found] 'Flip Flop' -> [224] Reef Draftsmen Mens Leather Sandals | Bottle Opener Flip Flops For Men With Soft Cushion Footbed | Waterproof

================================================================================
 ✨ MODEL PREDICTIONS (What you should buy next)
================================================================================
  1. Watches for Men Ultra-Thin Minimalist Waterproof Fashion Wrist Watch for Men with Stainless Steel Mesh Band (Black)
     Similarity: 0.431  |  Cat:   |  Price: $0.0
--------------------------------------------------------------------------------
  2. Linksoul Chambray Shorts – Mens Casual High Performance Flex Fit Cotton Blend Golf Apparel (Sand)
     Similarity: 0.408  |  Cat:   |  Price: $0.0
--------------------------------------------------------------------------------
  3. Contixo W2 Fast Wireless Charging Charger Pad | Ultra-Thin Slim Design for Qi Compatible Smartphones iPhone 8/8 Plus/X/XS/XS Max/XR Samsung Galaxy S9/S9 Plus/S8/S8 Plus/S7/Note 8/9
     Similarity: 0.388  |  Cat:   |  Price: $0.0
--------------------------------------------------------------------------------
  4. NCCB Girls Sandals Open Toe Hook Loop Sandals Cherry Anti-skid Flat Sandals for little girls with Strappy Leatherette Brown Size 8
     Similarity: 0.386  |  Cat:   |  Price: $0.0
--------------------------------------------------------------------------------
  5. Moyabo Women's Summer Casual T Shirt Dresses Short Sleeve Ruched Bodycon T Shirt Short Mini Dresses with Faux Button Black Large
     Similarity: 0.383  |  Cat:   |  Price: $0.0
--------------------------------------------------------------------------------
```

**2. ["Dress Shirt", "Silk Tie", "Cufflinks"]**

```
================================================================================
 🛒 BUILDING YOUR CUSTOM HISTORY
================================================================================
  [Found] 'Dress Shirt' -> [473] Gollnwe Men's Slim Fit French Cuff Stretch Bamboo Solid Dress Shirt White L
  [Found] 'Silk Tie' -> [184] Baked Apple Red Silk Tie and Pocket Square Paul Malone Red Line
  [Found] 'Cufflinks' -> [106] Alizeal Mens Classic Paisley Bow Tie, Hanky and Cufflinks Set

================================================================================
 ✨ MODEL PREDICTIONS (What you should buy next)
================================================================================
  1. Paul Malone Mens Silk Tie with Pocket Square and Cufflinks
     Similarity: 0.587  |  Cat:   |  Price: $0.0
--------------------------------------------------------------------------------
  2. PenSee Mens Solid Bowtie Woven Self Tie Bow Ties Tuxedo & Wedding Bowties
     Similarity: 0.577  |  Cat:   |  Price: $11.99
--------------------------------------------------------------------------------
  3. Bronze and Black Paul Malone Silk Tie and Pocket Square
     Similarity: 0.575  |  Cat:   |  Price: $0.0
--------------------------------------------------------------------------------
  4. Red and Black Silk Tie and Pocket Square Paul Malone Red Line
     Similarity: 0.557  |  Cat:   |  Price: $0.0
--------------------------------------------------------------------------------
  5. OURS Women's Notch Neck Floral Embroidery Shirt Tunic Top White (S, White)
     Similarity: 0.549  |  Cat:   |  Price: $0.0
--------------------------------------------------------------------------------
```

**3. ["Leggings", "Beanie", "Gloves"]**

```
================================================================================
 🛒 BUILDING YOUR CUSTOM HISTORY
================================================================================
  [Found] ' Leggings' -> [7] Aoxjox Women's High Waist Workout Gym Vital Seamless Leggings Yoga Pants
  [Found] ' Beanie' -> [21] Shinywear Lovely Baby Girls Knit Hats Bow-Knots Crochet Beanie Caps for Toddlers (White)
  [Found] ' Gloves' -> [27] MAGILINK Leather Gloves for Women Genuine Sheepskin, Womens Gloves Warm Thinsulate Lined, Winter Gloves Touchscreen Driving

================================================================================
 ✨ MODEL PREDICTIONS (What you should buy next)
================================================================================
  1. Sumind Women's Floral Lace Gloves Vintage Opera Gloves Elegant Short Gloves Fingerless Gloves for 1920s Opera Parties (Black 11, Wrist Length)
     Similarity: 0.477  |  Cat:   |  Price: $0.0
--------------------------------------------------------------------------------
  2. TOPGOMES Baby Bibs, Comfortable Soft Adjustable Fit Waterproof Bibs with BPA Free Silicone, Set of 2 Colors (Muted/Apricot)
     Similarity: 0.477  |  Cat:   |  Price: $0.0
--------------------------------------------------------------------------------
  3. WiliW Women's Footless Tights Black Control Top Opaque Pantyhose 2 Pairs Hold & Stretch XL
     Similarity: 0.456  |  Cat:   |  Price: $0.0
--------------------------------------------------------------------------------
  4. WANDER Women's Athletic Running Socks 3-6 Pairs Thick Cushion Ankle Socks for Women Sport Low Cut Socks 6-9/9-12
     Similarity: 0.446  |  Cat:   |  Price: $0.0
--------------------------------------------------------------------------------
  5. Womens Western Zipper Low Heel Round Toe Ankle Booties Natural US 7
     Similarity: 0.428  |  Cat:   |  Price: $0.0
--------------------------------------------------------------------------------
```

**4. ["disney", "pixar", "kids"]**

```
================================================================================
 🛒 BUILDING YOUR CUSTOM HISTORY
================================================================================
  [Found] 'disney' -> [196] Disney Girls 3-Pack T-Shirts: Wide Variety Includes Minnie, Frozen, Princess, Moana, Toy Story, and Lilo and Stitch
  [Found] 'pixar' -> [2347] Disney-Pixar Cars 3 Large 15.5-inch Reusable Shopping Tote or Gift Bag, 3-Pack
  [Found] 'kids' -> [10] Dapper&Doll Kids Apron and Chef Hat Gift Set - Toddler & Kid Sizes - Super Cute & Fun

================================================================================
 ✨ MODEL PREDICTIONS (What you should buy next)
================================================================================
  1. Accutime Kids LOL Surprise Turquoise Educational Learning Touchscreen Smart Watch Toy for Girls, Boys, Toddlers - Selfie Cam, Learning Games, Alarm, Calculator, Pedometer & More (Model: LOL4320OMGAZ)
     Similarity: 0.600  |  Cat:   |  Price: $31.42
--------------------------------------------------------------------------------
  2. Auxo Jumpsuits for Women Casual Summer Short Sleeve Loose Wide Legs Rompers with Pockets Black 2XL
     Similarity: 0.558  |  Cat:   |  Price: $0.0
--------------------------------------------------------------------------------
  3. Easter Basket Filler - Emoji LED Party Favor Rings, Fun Easter Egg Filler Toy - Stocking Stuffers For Kids -24 Count
     Similarity: 0.537  |  Cat:   |  Price: $9.95
--------------------------------------------------------------------------------
  4. Dokotoo Women's Loose Plus Size Jumpsuits for Women Adjustable Spaghetti Strap Stretchy Wide Leg Boho Floral Print One Piece Sleeveless Long Pant Romper Jumpsuit with Pockets Red X-Large
     Similarity: 0.531  |  Cat:   |  Price: $24.99
--------------------------------------------------------------------------------
  5. Bentibo Women's Floral Tribal Aztec Pattern Printed Leggings Stretch Pocket Jogger Pants Hot Pink S
     Similarity: 0.529  |  Cat:   |  Price: $0.0
--------------------------------------------------------------------------------
```
## Example Usage (from Notebook)

You can use the following code to interactively test the recommender with custom user histories (see notebook for full context):

```python
my_history = ["Swim Trunk", "Sunglasses", "Flip Flop"]
play_with_model(my_history)

my_history = ["Dress Shirt", "Silk Tie", "Cufflinks"]
play_with_model(my_history)

my_history = ["Leggings", "Beanie", "Gloves"]
play_with_model(my_history)

my_history = ["disney", "pixar", "kids"]
play_with_model(my_history)
```

This will show top recommendations for each custom user history, demonstrating the model's multimodal retrieval capabilities.
# multimodal-fashion-recommender

This repository implements a production-ready, multimodal recommender system for Amazon Fashion products using a modern two-tower deep learning architecture. The system leverages both text and image features for each product, enabling richer and more accurate recommendations than unimodal approaches. The project is fully modularized for reproducibility, extensibility, and GitHub best practices.

## Project Overview (Multimodal)

**Goal:** Build a scalable recommender system that leverages both user interaction history and multimodal product features (text and images) to provide high-quality product recommendations. The multimodal approach fuses textual and visual information, allowing the model to understand products more holistically and deliver superior recommendations.

**Key Features (Multimodal):**
- End-to-end pipeline: data loading, k-core filtering, multimodal embedding (text + image), model training, evaluation, and qualitative analysis
- Two-tower neural architecture: user tower (GRU over history), item tower (fuses text and CLIP image features)
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
  - **User Tower:** GRU encodes user history into a dense vector
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


## Notes

- Default settings: `DENSE_K = 3`, `SEQ_LEN = 15`
- Artifact checks expect these files in repo root:
  - `item_index_v11.faiss` (multimodal index)
  - `item_tower_vecs_v11.npy` (multimodal item vectors)

## Repository Name

**GitHub:** [Sonith-Bingi/multimodal-fashion-recommender](https://github.com/Sonith-Bingi/multimodal-fashion-recommender)


