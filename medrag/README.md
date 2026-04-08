# medrag_repro

A decoupled, multi-file reproduction scaffold for:

- MedQAUS (target QA set)
- PubMed abstracts (retrieval corpus)
- Contriever (dense retriever)
- PoisonedRAG black-box attack
- DRS defense

## Project layout

```text
medrag_repro/
  configs/
    minimal_medqaus_pubmed_contriever.yaml
  scripts/
    prepare_data.py
    build_index.py
    generate_poison.py
    eval_attack.py
    run_drs.py
  src/medrag_repro/
    config.py
    datamodels.py
    utils/
      io.py
      seed.py
      text.py
    data/
      medqa_loader.py
      pubmed_loader.py
    retriever/
      contriever.py
      index.py
    llm/
      client.py
      prompts.py
    attacks/
      poisonedrag_blackbox.py
    evaluation/
      rag_eval.py
    defense/
      drs.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

```bash
export OPENAI_API_KEY=your_key
# optional for a local OpenAI-compatible endpoint
# export OPENAI_BASE_URL=http://localhost:8000/v1
```

## Run

```bash
python scripts/prepare_data.py --config configs/minimal_medqaus_pubmed_contriever.yaml
python scripts/build_index.py --config configs/minimal_medqaus_pubmed_contriever.yaml
python scripts/generate_poison.py --config configs/minimal_medqaus_pubmed_contriever.yaml
python scripts/eval_attack.py --config configs/minimal_medqaus_pubmed_contriever.yaml
python scripts/run_drs.py --config configs/minimal_medqaus_pubmed_contriever.yaml
```

## Notes

- The default config uses a streamed PubMed subset for practical first runs.
- The attack is the black-box PoisonedRAG variant: `P = Q ⊕ I`.
- DRS is computed from clean reference docs retrieved by 300 clean queries.
