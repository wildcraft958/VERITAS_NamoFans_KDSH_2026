# Sample Outputs

This directory contains sample evidence ledger outputs demonstrating the VERITAS audit trail.

## Files

### `evidence_ledger_contradictory.json`
A CONTRADICTORY verdict example showing:
- **Backstory**: Claims about Thalcave having alpine/mountain heritage
- **Result**: REJECT (Rule 1: Hard Contradiction triggered)
- **Key Finding**: Character is explicitly described as "Patagonian of the plains" who "refuses to leave the prairies"

### `evidence_ledger_consistent.json`
A CONSISTENT verdict example showing:
- **Backstory**: Claims about Thalcave being raised by a shepherd with a protective mission
- **Result**: ACCEPT (Rule 4: Strong Support triggered)
- **Key Finding**: Backstory aligns with character's demonstrated behavior in the novel

## Evidence Ledger Structure

Each ledger contains:

1. **Metadata**: Backstory ID, novel ID, timestamp
2. **Final Verdict**: Prediction (0/1), confidence, triggered rule
3. **Atomic Facts**: FActScore-extracted claims from backstory
4. **Constraint Verdicts**: Per-claim analysis with:
   - Historian verdict (evidence-based)
   - Simulator verdict (persona alignment)
   - Key evidence citations with chapter references
5. **Reasoning Chain**: Step-by-step explanation
6. **Adjudication Rules**: Which of the 5 rules were evaluated

## Usage

These samples can be used to:
- Understand the output format for downstream processing
- Validate the audit trail completeness
- Debug edge cases in constraint classification

## Generating New Samples

```python
from kdsh.workflow import run_pipeline

result = run_pipeline(
    backstory="Your backstory text",
    novel_text="Novel content",
    backstory_id="custom_001",
    novel_id="novel_name"
)

# Access the evidence ledger
ledger = result["full_state"]["ledger_entries"]
```
