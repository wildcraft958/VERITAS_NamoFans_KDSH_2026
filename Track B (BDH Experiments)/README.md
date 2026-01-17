# BDH: Baby Dragon Hatchling Experiments

> **KDSH 2026 Track B** | Team NaMoFans | *Experimental Exploration*

A biologically-inspired Hebbian network approach for narrative consistency detection. This experimental track explores **synaptic drift** as a signal for detecting contradictions between character backstories and novel content.

---

## Context: Why BDH?

Before developing VERITAS (Track A), we explored a biologically-inspired hypothesis:

> *"Can a Hebbian network detect inconsistencies through 'Synaptic Drift'?"*

**Inspiration**: [The Dragon Hatchling](https://pathway.com/bdh) (Kosowski et al., 2025) — *"The Missing Link Between Transformer Information Storage and Brain Scale-Free Models"*

### The Hypothesis

```
1. PRIME     → Feed novel into synaptic matrix (establish "Golden State" W*)
2. PROBE     → Prime network with backstory (produce updated state W')
3. MEASURE   → Quantify drift: δ = ||W' - W*||₂

High Drift = Contradiction | Low Drift = Consistent
```

**Principle**: *"Neurons that fire together, wire together."*

---

## Results & Learnings

### Performance

| Model | Accuracy |
|-------|----------|
| Gemini 2.5 Pro (Naive) | 65% |
| **BDH Exploration** | **66%** |
| VERITAS (Track A) | 83% |

**Verdict**: +1pp over Gemini, but not sufficient for the challenge.

### Success Cases
- **Thematic Inconsistencies** (Tone/Mood): Continuous embeddings capture "vibe" well
- Detects stylistic mismatches effectively

### Failure Cases
- **Factual Contradictions** (Dates/Events): Cannot reliably detect discrete logic errors
- Example: "1990 + 50 = 2015?" — BDH sees two plausible concepts; discrete logic sees a math error

### Why We Pivoted to VERITAS

| Factor | BDH Limitation |
|--------|----------------|
| **Scale & Latency** | Synaptic matrix materialization = 3-5x latency per sample |
| **Evidence Distribution** | 88% of contradictions require multi-region synthesis; absence of evidence is hard to flag |
| **Data Scarcity** | Only 80 training samples vs. 10k needed for deep learning |
| **Interpretability** | Competition requires verbatim evidence; drift magnitude is opaque |

> **Core Insight**: Verification is a **Satisfiability** problem (Yes/No), not a **Similarity** problem (Maybe). We needed discrete constraints, not continuous states.

---

## Architecture

### Multi-Signal Pipeline

| Signal | Weight | Description |
|--------|--------|-------------|
| **NLI** | 0.35 | DeBERTa-v3 contradiction detection |
| **Heuristic (BDH)** | 0.25 | Synaptic Drift measurement |
| **Evidence** | 0.20 | Cross-encoder passage matching |
| **Temporal** | 0.10 | Allen's Interval Algebra (timeline) |
| **Causal** | 0.10 | Constraint graph violations |

### Synaptic Drift Detection

```
Novel Text (100k+ words)
    ↓
Coreference Resolution (He → "Thalcave")
    ↓
Chunked Embedding (BERT, 512 tokens)
    ↓
Hebbian State Building (Synaptic Matrix W*)
    ↓
Novel Indexed ✓

Backstory Text
    ↓
Prime Network → W'
    ↓
Drift: δ = ||W' - W*||₂
    ↓
Threshold Check → ACCEPT/REJECT
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/wildcraft958/BDH.git
cd BDH

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_trf
```

### Verify Setup

```bash
python scripts/verify_setup.py
```

### Run Evaluation

```bash
# Full evaluation with visualizations
python scripts/run_evaluation.py --mode full --visualize

# Quick test (10 samples)
python scripts/run_evaluation.py --mode quick --samples 10

# CPU-only mode
python scripts/run_evaluation.py --mode full --device cpu
```

### Results Location

Results saved to `evaluation_output/`:
- `evaluation_report.json` — Full metrics and per-sample results
- `confusion_matrix.png` — Confusion matrix visualization
- `signal_contributions.png` — Signal analysis chart
- `visualizations/` — Per-sample drift visualizations

---

## Usage

### Clean Interface

```python
from scripts.bdh import NarrativeConsistencyChecker

checker = NarrativeConsistencyChecker(device="cuda")
checker.load_novel("novel.txt")
result = checker.check("Alice was born in London...")

print(result.consistent)   # True / False
print(result.confidence)   # 0.0 - 1.0
print(result.rationale)    # Human-readable explanation
```

### Direct Pipeline

```python
from src.pipeline import create_sota_pipeline

pipeline = create_sota_pipeline(device="cuda")
pipeline.ingest_novel(novel_text)
verdict = pipeline.check_consistency(backstory_text)
```

---

## Memory Requirements

| Mode | GPU VRAM | System RAM | Time (80 samples) |
|------|----------|------------|-------------------|
| **GPU (Recommended)** | 4-5 GB | 8 GB | ~10-15 min |
| **CPU Only** | — | 16-20 GB | ~60-90 min |

**Model Sizes**:
- DeBERTa-v3-large (NLI): ~1.5 GB
- all-mpnet-base-v2 (Embeddings): ~420 MB
- cross-encoder/nli-deberta-v3-base: ~500 MB
- SpaCy en_core_web_trf: ~1 GB

---

## Project Structure

```
BDH/
├── src/
│   ├── pipeline/
│   │   └── sota_pipeline.py      # Entry point
│   ├── signals/
│   │   ├── bdh_scanner.py        # Hebbian Learning (Synaptic Drift)
│   │   ├── nli_classifier.py     # DeBERTa NLI classification
│   │   ├── temporal_graph.py     # Allen's Interval Algebra
│   │   ├── causal_extractor.py   # Constraint graph violations
│   │   └── cross_encoder_evidence.py
│   ├── models/
│   │   ├── modeling_bdh.py       # BDHForCausalLM
│   │   └── configuration_bdh.py
│   └── utils/
│       ├── visualization.py
│       └── coreference.py
│
├── scripts/
│   ├── run_evaluation.py
│   ├── verify_setup.py
│   └── generate_submission.py
│
├── tests/
├── dataset/
└── evaluation_output/
```

---

## Key Insights for Future Work

1. **Hebbian networks excel at thematic coherence** but struggle with factual precision
2. **Synaptic drift is a useful signal** but insufficient alone—needs symbolic reasoning
3. **Multi-signal fusion** (NLI + drift + temporal + causal) improves robustness
4. **Discrete constraints** (VERITAS approach) outperform continuous embeddings for verification tasks

---

## References

1. **Kosowski et al.** (BDH, 2025) — Baby Dragon Hatchling architecture
2. **Sinha et al.** (PRELUDE, 2025) — Benchmark dataset

---

## Relationship to Track A

This experimental Track B informed the development of **VERITAS** (Track A):

- BDH's thematic detection capability → Inspired the "Simulator" persona stream
- Failure on factual contradictions → Led to FActScore atomic decomposition
- Need for interpretability → Resulted in Evidence Ledger design
- Multi-signal approach → Evolved into 5-layer architecture

See `../Track A (VERITAS)/` for the production solution achieving **83% accuracy**.

---

## License

MIT
