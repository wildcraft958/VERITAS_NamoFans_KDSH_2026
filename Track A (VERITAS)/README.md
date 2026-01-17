# VERITAS: Verification Engine for Reasoning over Intertextual Assertions in Stories

> **KDSH 2026 Track A Submission** | Team NaMoFans

A 5-layer constraint-driven architecture for narrative consistency verification. Given a 100k+ word novel and a character backstory, VERITAS determines if the backstory is **consistent (1)** or **contradictory (0)** with the novel's narrative.

---

## The Challenge

**Input**: 100,000-word novel + 2-5k word character backstory

**Core Question**: Does this backstory satisfy ALL narrative constraints across the entire novel WITHOUT contradictions?

**Critical Insight**: This is not about writing quality or plausibility—it is a problem of **pure satisfiability**.

### Why SOTA Systems Fail

Current approaches optimize for *plausibility* (sounds reasonable), but verification requires *satisfiability* (constraints hold everywhere):

| Failure Mode | Example |
|-------------|---------|
| **False Positive via Local Similarity** | RAG finds "mountains" keyword, accepts claim. But Page 300 says "grew up in desert" |
| **False Positive via Prior Bias** | LLM sees fantasy context, accepts "prophetic dreamer"—but novel has no magic system |

> *"The question isn't 'Does this sound right?' The question is 'Are all constraints satisfied?'"*

---

## Performance Results

| Model | Accuracy |
|-------|----------|
| Gemini 2.5 Pro (Naive) | 65% |
| Baseline RAG | 68% |
| BDH Exploration (Track B) | 66% |
| **VERITAS** | **83%** |

**Key Metrics**:
- **83% Accuracy** (+18pp vs Gemini 2.5 Pro)
- **91% Precision** (+39pp vs Gemini)
- **0.87 F1 Score**
- **8.5s Inference Time** (Cloud Native, No GPU required)
- **Zero Training** (Pre-trained models only)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    VERITAS 5-Layer Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1: Ingestion & Atomization                               │
│           Backstory → FActScore Decomposition → Atomic Claims   │
│           Novel → Zero-Shot Classification                      │
│                                                                 │
│  Layer 2: Semantic Hierarchy                                    │
│           Novel → RAPTOR Tree Indexing                          │
│           "Zoomable" plot mapping (forest AND trees)            │
│                                                                 │
│  Layer 3: Relational Entanglement                               │
│           HippoRAG Knowledge Graph                              │
│           Time-bound character relationships                    │
│                                                                 │
│  Layer 4: Dual-Vector Retrieval                                 │
│           Historian Stream (Macro-Context: plot, timeline)      │
│           Simulator Stream (Micro-Context: dialogue, motives)   │
│                                                                 │
│  Layer 5: Conservative Adjudication                             │
│           Evidence Ledger → Final Verdict + Confidence          │
│           1 Contradiction > 10 Supports                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

> **Design Philosophy**: *"We replaced the 'black box' with a modular pipeline that mirrors human investigative steps."*

---

## Layer Details

### Layer 1: FActScore Decomposition
**Based on**: Min et al., EMNLP 2023

Breaks backstory into atomic, testable claims:

```
Input:  "Born in 1990, Alice grew up in a small town. After university, 
         she moved to the city to pursue her career."

Output: • Born in 1990
        • Grew up in a small town
        • Moved to the city after university
        • Pursuing her career
```

### Layer 2: RAPTOR Tree Indexing
**Based on**: Sarthi et al., ICLR 2024

Creates hierarchical semantic index:
- **Root Level**: Novel-wide summaries
- **Middle Levels**: Chapter/section summaries  
- **Leaf Level**: Text chunks

### Layer 3: HippoRAG Knowledge Graph
**Based on**: Zamani et al., NeurIPS 2024

Tracks evolving character relationships with time-stamped edges:
```
Ch. 1:  Protagonist ──[Friend]──→ Minor Character
Ch. 20: Protagonist ──[Betrayal]──→ Minor Character
Ch. 30: Protagonist ──[Enemy]──→ Minor Character
```

### Layer 4: Dual-Vector Retrieval

Prevents "needle-in-haystack" blindness:
- **Macro-Context (Tree)**: Broad plot points, settings, timeline
- **Micro-Context (Graph)**: Specific dialogue, interactions, motives

### Layer 5: Evidence Ledger & Adjudication

Produces audit-ready verdicts:

| Claim | Verdict | Evidence Type | Citation |
|-------|---------|---------------|----------|
| Served in 2010 | REJECT | Contradiction | Ch. 15: "Discharged in 2005" |
| Born in London | ACCEPT | Support | Ch. 3: "London-born" |
| Met King | REJECT | Missing Evidence | No royal visit in timeline |

---

## The 5 Asymmetric Decision Rules

The adjudicator applies these rules sequentially (first match wins):

1. **HARD CONTRADICTION**: Any contradiction weight ≥ threshold → REJECT
2. **MULTIPLE MISSING**: ≥2 high-impact missing evidence → REJECT  
3. **LOW PERSONA ALIGNMENT**: Simulator alignment < threshold → REJECT
4. **STRONG SUPPORT**: ≥3 supporting + high alignment → ACCEPT
5. **DEFAULT**: Conservative rejection (insufficient evidence)

> **Design Philosophy**: *It's easier to prove a claim false than true. One hard contradiction is fatal.*

---

## External Framework References

| Layer | Framework | Repository | Paper |
|-------|-----------|------------|-------|
| Layer 1 | FActScore | [shmsw25/FActScore](https://github.com/shmsw25/FActScore) | EMNLP 2023 |
| Layer 2 | RAPTOR | [parthsarthi03/raptor](https://github.com/parthsarthi03/raptor) | ICLR 2024 |
| Layer 3 | HippoRAG | [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) | NeurIPS 2024 |
| Workflow | LangGraph | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | — |

Local copies included in `external_frameworks/` for reference.

---

## Quick Start

### Prerequisites

```bash
# Required API Keys
export OPENAI_API_KEY="sk-your-openai-api-key"

# Optional: Alternative providers
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

### Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .

# Copy and configure environment
cp .env.example .env
```

### Usage

```bash
# Single prediction
python -m kdsh.main --single --backstory story.txt --novel novel.txt

# Batch processing
python -m kdsh.main --batch train.csv --novels-dir Books/ --output results.csv

# Start API server
python -m kdsh.main --serve
```

### API Usage

```bash
# Start server
python -m kdsh.main --serve

# Make prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"backstory": "...", "novel_text": "..."}'
```

---

## Project Structure

```
src/kdsh/
├── layer1/         # Atomic Constraint Decomposition (FActScore-inspired)
├── layer2/         # Semantic Hierarchy (RAPTOR tree indexing)
├── layer3/         # Relational Entanglement (HippoRAG + Historian/Simulator)
├── layer4/         # Evidence Ledger Aggregation
├── layer5/         # Conservative Adjudication (5 Asymmetric Rules)
├── workflow/       # LangGraph Pipeline Integration
└── api/            # API Server

external_frameworks/  # Reference implementations
├── FActScore/        # Atomic fact extraction (EMNLP 2023)
├── raptor/           # Recursive tree indexing (ICLR 2024)
└── HippoRAG/         # Knowledge graph retrieval (NeurIPS 2024)

tests/               # Test suite
sample_outputs/      # Example evidence ledgers and verdicts
```

---

## Running Tests

```bash
# Install test dependencies
uv pip install pytest pytest-cov

# Run all tests
pytest tests/ -v --log-cli-level=INFO

# Run with coverage
pytest tests/ --cov=kdsh --cov-report=html
```

---

## Key References

1. **Sinha et al.** (PRELUDE, 2025) — Benchmark dataset
2. **Min et al.** (FActScore, EMNLP 2023) — Atomic claim decomposition
3. **Sarthi et al.** (RAPTOR, ICLR 2024) — Recursive tree indexing
4. **Zamani et al.** (HippoRAG, NeurIPS 2024) — Knowledge graph for character tracking
5. **Pathway AI** (LangGraph, 2025) — Infrastructure

---

## License

MIT
