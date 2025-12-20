# Clinical Trial Matching (Proposer/Skeptic + Ranking)

This repo contains a small clinical-trial matching pipeline built around two LLM agents (Proposer + Skeptic), plus evaluation and ranking utilities. The code parses trial eligibility text into clauses, checks each clause against a patient note, and can rank trials based on missing information.

## Repository layout

- `src/clinical_tools.py`: sentence splitting, criteria parsing, evidence pool helper.
- `src/proposer_agent.py`: Proposer agent prompt + JSON normalization.
- `src/skeptic_agent.py`: Skeptic auditor prompt and verdict logic.
- `src/refine.py`: orchestration loop (propose -> audit -> revise).
- `src/eval/run_eval.py`: evaluate Proposer/Skeptic outputs vs. ground-truth labels.
- `src/eval/run_trialgpt_eval.py`: evaluate TrialGPT outputs vs. ground-truth labels.
- `src/rank/LLM_classifier_ollama.py`: classify NEI clauses for tie-breaks.
- `src/rank/rank_trial.py`: rank trials using I/S/U + NEI tie-breakers.
- `src/rank/ranked_trials.json`: example output.

## Installation

Requirements:
- Python 3.10+ recommended
- Ollama for local model inference (used by Proposer/Skeptic and NEI classification)
- `requests` and `nltk`

Setup (from repo root):
```bash
python -m venv .venv
source .venv/bin/activate
pip install requests nltk
python -m nltk.downloader punkt
```

Ollama (if you have not set it up):
```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

Optional (TrialGPT evaluation):
`src/eval/run_trialgpt_eval.py` expects the TrialGPT repo to be checked out at `../TrialGPT/trialgpt_matching` relative to this repo.

## Quick start

Run the proposer/skeptic loop on one patient note and trial:
```bash
python src/refine.py \
  --patient_file path/to/patients.jsonl \
  --trial_file path/to/trials.jsonl \
  --patient_id PATIENT_ID \
  --trial_id TRIAL_ID \
  --model qwen2.5:7b-instruct
```

Expected input formats:

`patients.jsonl` (JSON Lines, one record per patient):
```json
{"_id":"P1","text":"Patient note with multiple sentences."}
```

`trials.jsonl` (JSON Lines, one record per trial):
```json
{"_id":"T1","text":"Inclusion Criteria:\n- ...\nExclusion Criteria:\n- ..."}
```

The trial `text` should contain the headings "Inclusion Criteria" and "Exclusion Criteria" so `parse_criteria()` can split clauses.

## Usage examples

Evaluate agent decisions against ground truth:
```bash
python src/eval/run_eval.py \
  --eval_file path/to/eval_payload.json \
  --patient_file path/to/patients.jsonl \
  --output_file outputs/eval_results.json
```

`eval_payload.json` shape (minimum):
```json
{
  "patient_record": {"_id":"P1"},
  "clinical_trials": [
    {
      "_id": "T1",
      "inclusion": [{"text":"Clause text", "label":"included"}],
      "exclusion": [{"text":"Clause text", "label":"not exclude"}]
    }
  ]
}
```

Evaluate TrialGPT outputs vs. ground truth (requires TrialGPT repo):
```bash
python src/eval/run_trialgpt_eval.py \
  --eval_file path/to/eval_payload.json \
  --patient_file path/to/patients.jsonl \
  --trial_file path/to/trials.jsonl \
  --output_file outputs/trialgpt_eval_results.json
```

Rank trials with NEI tie-breakers:
```bash
python src/rank/rank_trial.py \
  --predictions outputs/eval_results.json \
  --output outputs/ranked_trials.json \
  --cache outputs/nei_cache.json \
  --model qwen2.5:7b-instruct
```

Notes:
- `rank_trial.py` expects `details` entries to include a `missing_info` field (string). If your predictions do not contain it, NEI classification will still run but will not have missing-info hints.
- Ranking calls Ollama for NEI classification. Set `--base_url` if your Ollama server is not on `http://localhost:11434`.

## Development guide

- Running scripts from repo root:
  - `python src/refine.py` works because `src` is on the module path for that process.
  - `python src/eval/run_eval.py` adjusts `sys.path` to include the repo root.
  - `python src/rank/rank_trial.py` imports from `src/rank`, so run from repo root or from `src/rank`.
- Adding a new model:
  - Update the default model name in `src/proposer_agent.py`, `src/skeptic_agent.py`, and `src/rank/LLM_classifier_ollama.py`.
  - Ensure the Ollama model is pulled locally.
- Changing criteria parsing:
  - `src/clinical_tools.py:parse_criteria()` is a rule-based splitter with simple numeric-threshold detection.
  - If your trial text format differs, adjust the heading detection and clause parsing there.
- Data hygiene:
  - The pipeline expects reasonably clean sentence boundaries; verify `sentence_split()` for your domain.
  - If you use `run_trialgpt_eval.py`, ensure you have `nltk` and the `punkt` tokenizer installed.

## Reference

`src/refine.py`
- `--patient_file`: JSON or JSONL file with `_id` and `text`
- `--trial_file`: JSON or JSONL file with `_id` and `text`
- `--patient_id`: record selector (optional if file has a single record)
- `--trial_id`: record selector (optional if file has a single record)
- `--model`: Ollama model name

`src/eval/run_eval.py`
- `--eval_file`: evaluation payload JSON
- `--patient_file`: patients JSONL
- `--output_file`: results JSON

`src/eval/run_trialgpt_eval.py`
- `--eval_file`: evaluation payload JSON
- `--patient_file`: patients JSONL
- `--trial_file`: trials JSONL (with metadata or full text for criteria)
- `--output_file`: results JSON

`src/rank/rank_trial.py`
- `--predictions`: predictions JSON (with `details`)
- `--output`: ranked trials JSON
- `--cache`: NEI classification cache JSON
- `--model`: Ollama model name
- `--base_url`: Ollama base URL (default `http://localhost:11434`)
