# VERITAS: Verification Engine for Reasoning over Intertextual Assertions in Stories

<div align="center">

**Team NaMoFans | KDSH 2026 | Track A Submission**

[![Accuracy](https://img.shields.io/badge/Accuracy-83%25-brightgreen?style=for-the-badge)](.)
[![Precision](https://img.shields.io/badge/Precision-91%25-blue?style=for-the-badge)](.)
[![F1](https://img.shields.io/badge/F1-0.87-orange?style=for-the-badge)](.)
[![Inference](https://img.shields.io/badge/Inference-8.5s-purple?style=for-the-badge)](.)

</div>

---

## Quick Navigation for Judges

| Resource | Description |
|----------|-------------|
| [Final Report](NamoFans_KDSH_2026_Final_Report.pdf) | Complete technical documentation |
| [Extended Report](NamoFans_KDSH_2026_Extended_Technical_Report.pdf) | Detailed appendices and analysis |
| [Results](results.csv) | Full evaluation outputs |
| [Track A Code](Track%20A%20(VERITAS)/) | **Primary submission** - VERITAS implementation |
| [Track B Code](Track%20B%20(BDH%20Experiments)/) | Experimental BDH exploration |

---

## The Challenge

**Given**: A 100,000-word novel + 2-5k word character backstory

**Task**: Determine if the backstory is **consistent (1)** or **contradictory (0)** with the novel

**Key Insight**: This is a **satisfiability** problem, not a plausibility problem. One contradiction invalidates everything.

---

## Results Summary

| Model | Accuracy | Notes |
|-------|----------|-------|
| Gemini 2.5 Pro (Baseline) | 65% | SOTA LLM |
| Baseline RAG | 68% | Standard retrieval |
| BDH Exploration (Track B) | 66% | Hebbian learning |
| **VERITAS (Track A)** | **83%** | **+18pp improvement** |

**Key Metrics**:
- **83% Accuracy** — Exceeds human-level gap
- **91% Precision** — Minimal false positives  
- **8.5s Inference** — Cloud-native, no GPU required
- **Zero Training** — Pre-trained models only

---

## Architecture Overview

VERITAS uses a **5-layer constraint-driven pipeline** that mirrors human investigative reasoning:

```
+---------------------------------------------------------------------+
|                         VERITAS Pipeline                            |
+---------------------------------------------------------------------+
|                                                                     |
|  +-----------+    +-----------+    +-------------------------+      |
|  | BACKSTORY |    |   NOVEL   |    |       LAYER 1           |      |
|  |  (Input)  |--->|  (Input)  |--->|  Atomic Decomposition   |      |
|  +-----------+    +-----------+    |  (FActScore)            |      |
|                                    +-----------+-------------+      |
|                                                |                    |
|                                                v                    |
|                                    +-------------------------+      |
|                                    |       LAYER 2           |      |
|                                    |  Semantic Hierarchy     |      |
|                                    |  (RAPTOR Tree)          |      |
|                                    +-----------+-------------+      |
|                                                |                    |
|                                                v                    |
|                                    +-------------------------+      |
|                                    |       LAYER 3           |      |
|                                    |  Relational Entanglement|      |
|                                    |  (HippoRAG Graph)       |      |
|                                    +-----------+-------------+      |
|                                                |                    |
|                                                v                    |
|                          +---------------------+-------------------+|
|                          |           LAYER 4                       ||
|                          |      Dual-Vector Retrieval              ||
|                          |  +----------+    +--------------+       ||
|                          |  |Historian |    |  Simulator   |       ||
|                          |  | (Macro)  |    |   (Micro)    |       ||
|                          |  +----+-----+    +------+-------+       ||
|                          +-------+-----------------+---------------+|
|                                  |                 |                |
|                                  v                 v                |
|                          +-------------------------------------+    |
|                          |           LAYER 5                   |    |
|                          |    Conservative Adjudication        |    |
|                          |    +---------------------------+    |    |
|                          |    |      Evidence Ledger      |    |    |
|                          |    | +-----+--------+--------+ |    |    |
|                          |    | |Claim|Verdict |Citation| |    |    |
|                          |    | +-----+--------+--------+ |    |    |
|                          |    +---------------------------+    |    |
|                          |              |                      |    |
|                          |              v                      |    |
|                          |    FINAL VERDICT + CONFIDENCE       |    |
|                          +-------------------------------------+    |
+---------------------------------------------------------------------+
```

---

## Repository Structure

```
NamoFans_KDSH_2026/
|
+-- README.md                                    # This file
+-- NamoFans_KDSH_2026_Final_Report.pdf          # Primary documentation
+-- NamoFans_KDSH_2026_Extended_Technical_Report.pdf
+-- results.csv                                  # Evaluation results
|
+-- Track A (VERITAS)/                           # PRIMARY SUBMISSION
|   +-- README.md                                # Track A documentation
|   +-- src/kdsh/                                # Core implementation
|   |   +-- layer1/                              # Atomic fact extraction
|   |   +-- layer2/                              # Semantic hierarchy
|   |   +-- layer3/                              # Relational graphs
|   |   +-- layer4/                              # Evidence aggregation
|   |   +-- layer5/                              # Adjudication engine
|   |   +-- workflow/                            # LangGraph orchestration
|   +-- external_frameworks/                     # Reference implementations
|   |   +-- FActScore/                           # EMNLP 2023
|   |   +-- raptor/                              # ICLR 2024
|   |   +-- HippoRAG/                            # NeurIPS 2024
|   +-- tests/                                   # Test suite
|   +-- sample_outputs/                          # Example evidence ledgers
|
+-- Track B (BDH Experiments)/                   # Experimental exploration
    +-- README.md                                # Track B documentation
    +-- src/                                     # BDH implementation
    +-- research/                                # BDH papers & experiments
```

---

## The 5 Layers Explained

### Layer 1: Atomic Decomposition
**Based on**: FActScore (Min et al., EMNLP 2023)

Breaks backstory into testable atomic claims:
```
"Born in 1990, Alice grew up in a small town"
    |
    v
* Born in 1990
* Grew up in a small town
```

### Layer 2: Semantic Hierarchy  
**Based on**: RAPTOR (Sarthi et al., ICLR 2024)

Creates "zoomable" novel index—see the forest AND the trees.

### Layer 3: Relational Entanglement
**Based on**: HippoRAG (Zamani et al., NeurIPS 2024)

Tracks evolving character relationships with time-stamped edges.

### Layer 4: Dual-Vector Retrieval
- **Historian Stream**: Macro-context (plot, timeline, settings)
- **Simulator Stream**: Micro-context (dialogue, motives, interactions)

### Layer 5: Conservative Adjudication

Applies 5 asymmetric decision rules:

| Priority | Rule | Action |
|----------|------|--------|
| 1 | Hard contradiction found | REJECT |
| 2 | >=2 missing critical evidence | REJECT |
| 3 | Low persona alignment | REJECT |
| 4 | Strong multi-source support | ACCEPT |
| 5 | Insufficient evidence | REJECT (conservative) |

> **Design Philosophy**: *One contradiction is fatal. It's easier to prove false than true.*

---

## Track B: BDH Experiments

Before VERITAS, we explored a **biologically-inspired hypothesis**:

> *"Can a Hebbian network detect contradictions through synaptic drift?"*

**Result**: 66% accuracy—good for thematic consistency, but failed on factual contradictions.

**Key Learning**: Verification needs **discrete logic**, not continuous similarity. This insight directly informed VERITAS's constraint-driven design.

---

## Key References

1. **Sinha et al.** (PRELUDE, 2025) — Benchmark dataset
2. **Min et al.** (FActScore, EMNLP 2023) — Atomic claim decomposition
3. **Sarthi et al.** (RAPTOR, ICLR 2024) — Recursive tree indexing
4. **Zamani et al.** (HippoRAG, NeurIPS 2024) — Knowledge graph retrieval
5. **Kosowski et al.** (BDH, 2025) — Baby Dragon Hatchling (Track B inspiration)

---

## Quick Start

```bash
cd "Track A (VERITAS)"

# Install
pip install -e .

# Set API key
export OPENAI_API_KEY="sk-..."

# Run single prediction
python -m kdsh.main --single --backstory story.txt --novel novel.txt

# Batch processing
python -m kdsh.main --batch train.csv --novels-dir Books/ --output results.csv
```

---

## Team NaMoFans

KDSH 2026 Track A Final Submission

---

<div align="center">

**Thank you for reviewing our submission!**

</div>
