#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tqdm import tqdm

from medrag_repro.attacks.poisonedrag_blackbox import PoisonedRAGBlackBoxGenerator
from medrag_repro.config import load_config
from medrag_repro.datamodels import QAItem
from medrag_repro.llm.client import load_openai_client
from medrag_repro.utils.io import read_jsonl, write_jsonl
from medrag_repro.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    client = load_openai_client()
    targets = [QAItem(**row) for row in read_jsonl(cfg["paths"]["targets"])]
    gen = PoisonedRAGBlackBoxGenerator(
        client=client,
        model=cfg["poisonedrag"]["generator_model"],
        max_words=cfg["poisonedrag"]["max_words_for_I"],
        max_trials=cfg["poisonedrag"]["max_trials"],
        generator_temperature=cfg["poisonedrag"]["generator_temperature"],
        verifier_temperature=cfg["poisonedrag"]["verifier_temperature"],
    )
    poison = gen.generate_for_targets(targets, cfg["poisonedrag"]["n_poison_per_target"])
    write_jsonl(cfg["paths"]["poison_docs"], poison)
    verified = sum(int(p.generation_verified) for p in poison)
    print(f"Saved {len(poison)} poison docs; verified {verified}/{len(poison)}")


if __name__ == "__main__":
    main()
