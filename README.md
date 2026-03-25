# Financial Reconciliation System

An ML-powered system that automatically matches transactions between bank statements and a check register, inspired by Peter Chew's research on reframing financial reconciliation as a cross-language information retrieval problem.

## Quick Start

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py run

# Run with match table output
python main.py run --show-matches

# Export results to CSV
python main.py run --export results.csv

# Run without sentence-transformers (faster, uses TF-IDF fallback)
python main.py run --no-embeddings
```

## Architecture

### Two-Phase Matching Pipeline

#### Phase 1: Unique Amount Matching
Transactions with amounts appearing exactly once in both datasets are matched automatically. This provides ~93% coverage (286/308) with 100% precision and serves as the "parallel corpus" seed for Phase 2.

**Confidence scoring:**
- Base confidence = 1.0 for unique amount matches
- Penalized for date differences > 5 days
- Penalized for transaction type mismatches

#### Phase 2: Hybrid SVD + Embedding Matching (ML)
Remaining unmatched transactions (22/308) are handled by a hybrid ML approach:

1. **Term Alignment via PMI** — Build a co-occurrence matrix between bank description tokens and register description tokens from Phase 1 matches. Compute pointwise mutual information for cross-vocabulary alignment.

2. **Sentence Embeddings** — Use `sentence-transformers` (`all-MiniLM-L6-v2`) to embed transaction descriptions into a shared semantic space. Falls back to TF-IDF character n-grams if unavailable.

3. **SVD Dimensionality Reduction** — Project aligned term vectors into a reduced space (k=40 components) per the paper's approach.

4. **Ensemble Scoring** — Combine SVD similarity (40%), embedding similarity (30%), numerical feature similarity (30%), plus an exact-amount match bonus.

5. **Hungarian Algorithm** — Optimal 1-to-1 assignment using the ensemble similarity matrix.

6. **Iterative Refinement** — High-confidence ML matches (>0.85) are added to the seed corpus, and matching is re-run on remaining transactions for up to 3 iterations.

### Project Structure

```
├── main.py                # CLI entry point
├── src/
│   ├── data_loader.py     # CSV parsing & normalization
│   ├── preprocessor.py    # Text normalization, feature engineering
│   ├── unique_matcher.py  # Phase 1: unique amount matching
│   ├── ml_matcher.py      # Phase 2: hybrid SVD + embedding matching
│   ├── evaluator.py       # Precision/Recall/F1 computation
│   ├── reconciler.py      # Pipeline orchestrator
│   └── utils.py           # Shared utilities & data classes
├── tests/                 # Unit tests (39 tests)
├── requirements.txt
├── bank_statements.csv    # Input: 308 bank transactions
└── check_register.csv     # Input: 308 register transactions
```


## Challenges & Design Decisions

### Why Hybrid Approach?
- **Unique amount matching** alone handles ~93% of cases but can't disambiguate same-amount transactions
- **Pure embedding matching** struggles with the vocabulary mismatch (bank uses "BP GAS #1775", register uses "Fill up")
- **The hybrid** uses PMI alignment to bridge the vocabulary gap (like the paper's cross-language IR approach) while leveraging modern embeddings for semantic understanding

### Key Challenges Addressed
1. **Vocabulary mismatch**: Bank statements use merchant names ("KROGER #6864"), register uses categories ("Grocery store"). Solved via PMI alignment + semantic embeddings.
2. **Date discrepancies**: Register dates often precede bank dates by 1-5 days. Handled via date tolerance in confidence scoring.
3. **Amount near-duplicates**: Some transactions share the same amount. Solved by Phase 2 disambiguation using text + date features.
4. **Typos**: Both datasets contain typos ("AAZON.COM", "Grocey store", "Insuranc payment"). Handled via character n-gram embeddings and robust text normalization.
5. **Small amount differences**: Some matching transactions differ by pennies (e.g., $207.03 vs $207.04). The ensemble scoring accounts for this.

### Limitations & Future Work
- **Scalability**: The Hungarian algorithm is O(n³); for large datasets, approximate matching (greedy or beam search) would be needed
- **Cold start**: Phase 2 depends on Phase 1 seed matches; if few unique amounts exist, the parallel corpus may be too small
- **Anomaly detection**: Currently matches all transactions; could add an "unmatched/anomaly" category for genuine mismatches
- **Active learning**: Could incorporate human feedback on uncertain matches to improve iteratively

## Running Tests

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

All 39 tests cover: data loading, preprocessing, unique matching, ML matching, and evaluation metrics.
