# TO-DO: DRS Paper Experiments Not Yet Covered by This Repo

This repo's three use cases showcase DRS and baseline defenses against
data-poisoning attacks, but they don't reproduce every experiment in the
source paper (Xun Xian, Tong Wang, Liwen You, Yanjun Qi (2025).
"Understanding Data Poisoning Attacks for RAG: Insights and Algorithms",
[openreview.net/forum?id=2aL6gcFX7q](https://openreview.net/forum?id=2aL6gcFX7q)
— see [`README.md`](README.md#citation)). This file tracks what the paper
tested that isn't here yet, and a rough plan for closing each gap. See also
[`docs/drs-dual-pca-analysis.md`](docs/drs-dual-pca-analysis.md) for the
paper-consistency check that surfaced most of this.

## Coverage matrix

The paper's Section 5 has four experiment groups (Tables 2-5). Mapped
against this repo's three use cases:

| Paper experiment | Paper's setup | This repo | Status |
| --- | --- | --- | --- |
| §5.1.1 Agent-Driver (Table 2, row 1) | RAG agent for autonomous-driving decisions; attacks: AgentPoison, BadChain, AutoDAN, GCG | No corresponding use case | **Not covered** |
| §5.1.1 ReAct-StrategyQA (Table 2, row 2) | Same 4 attacks against a ReAct StrategyQA agent | `use-cases/strategyqa_agent/` | **Partially covered** — only a BadChain-style/AgentPoison-token backdoor injection; AutoDAN and GCG aren't implemented |
| §5.1.2 Dense retrieval, general QA (Table 3) | BadDPR backdoor attack (Long et al., 2024) on a dense passage retriever, swept across 1%/5%/10%/20% backdoor ratios | No corresponding use case | **Not covered** |
| §5.1.3 Medical QA RAG (Table 4) | PoisonedRAG attack; 3 query sets (MedQAUS, MedMCQA, PubMedQA) × 2 corpora (Textbook, PubMed) × 2 retrievers (Contriever, MedCPT) | `use-cases/medqa_rag/` | **Partially covered** — only MedQAUS × PubMed × Contriever, 1 of the paper's 6 (retriever × query-set) combinations, and on a different corpus (PubMed, not the paper's Textbook) |
| §5.2 DRS-regularized attack (Table 5) | The paper's own adaptive attack: adds a DRS-penalty regularization term to AgentPoison's objective, trading some attack success for a lower DRS detection rate | Not implemented in any attack | **Not covered** |

`use-cases/trial_retrieval/` (clinical-trial retrieval, TrialGPT-style) has
no counterpart in the paper at all — it's an extension of DRS to a
scenario the paper didn't test, not a gap.

## Gap 1: Agent-Driver use case (not covered)

The paper's first, primary experiment scenario — an LLM agent making
autonomous-driving decisions via RAG, attacked to produce unsafe decisions
— has no counterpart here. Closing this means a new `use-cases/` directory
from scratch:

- [ ] Source or reconstruct an autonomous-driving RAG benchmark
  matching the paper's setup (it cites Caesar et al., 2020 / nuScenes as
  the driving-decision data source — check the paper's appendix, not
  included in the copy checked, for exact benchmark details before
  committing to a dataset).
- [ ] A RAG agent loop analogous to `strategyqa_agent/ReAct/local_wikienv.py`
  but for driving decisions instead of StrategyQA.
- [ ] Wire in `drs_defense`, `defenses`, and `attacks` (or a new attack
  module — see Gap 2) the same way the three existing use cases do.
- [ ] A README following the five-part structure (see any existing
  use case's), plus registration in the root README's showcase table.

This is the largest gap — a genuinely new use case, not an extension of an
existing one. Reasonable to defer until the smaller gaps below are closed.

## Gap 2: AutoDAN and GCG attacks (not covered)

`strategyqa_agent`'s attack is a hand-crafted backdoor-trigger injection
(a fixed phrase or a pre-computed adversarial token sequence — see its
README). The paper's Table 2 compares DRS against *four* attacks per
scenario: AgentPoison, BadChain (both represented here, even if
AgentPoison's own trigger-optimization code was removed as out of scope —
see `CLAUDE.md`), plus **AutoDAN** (Liu et al., 2023a — stealthy jailbreak
prompt generation) and **GCG** (Zou et al., 2023 — Greedy Coordinate
Gradient, gradient-based adversarial suffix optimization). Neither is
implemented anywhere in this repo.

- [ ] Decide whether to add GCG/AutoDAN to `attacks/` as new modules
  (`rag_attacks.gcg`, `rag_attacks.autodan`) or keep them use-case-local,
  matching how `attacks/README.md`'s "genuinely different attacks stay
  separate modules" convention already handles `poisonedrag_medqa` vs.
  `poisonedrag_trial`.
- [ ] **Scope tension worth resolving explicitly first:** this repo
  already made a deliberate decision to exclude gradient-based trigger
  optimization (`Agent_Setting/algo/trigger_optimization.py` was deleted
  as "not RAG defense, not a simple attack" — see the design spec in
  `docs/superpowers/specs/`). GCG is exactly this kind of attack
  (white-box, gradient-guided). Revisit that scope decision explicitly
  before implementing GCG, rather than silently reintroducing what was
  previously removed.
- [ ] AutoDAN doesn't need gradient/white-box LLM access (it's a
  genetic-algorithm-style prompt search) and is a more tractable
  starting point than GCG if only one of the two gets prioritized.
- [ ] Extend `strategyqa_agent`'s `--algo` choices (currently
  `{ap, badchain}`) to include the new attack(s), and extend
  `run_strategyqa_inference.py`'s printed comparison / `_format_defense_
  comparison_table` if a cross-attack comparison table is wanted
  alongside the existing cross-defense one.

## Gap 3: BadDPR backdoor-ratio sweep on general-purpose QA (not covered)

The paper's second experiment group backdoors a dense passage retriever
directly (Long et al., 2024's BadDPR — poisoning the retriever's own
training data with query-response pairs, not just injecting corpus
documents) and sweeps the backdoor ratio (1%/5%/10%/20%) to see how
detection rate changes with attack intensity. Every attack in this repo
today is corpus-injection-only (add poisoned documents, retriever
untouched) — there's no retriever-training-time attack anywhere, and no
existing experiment sweeps an attack-intensity parameter the way Table 3
does.

- [ ] This needs retriever fine-tuning infrastructure this repo doesn't
  have yet (`medqa_rag`'s `ContrieverEncoder` and `trial_retrieval`'s
  `MedCPTQueryEncoder` both load frozen pretrained encoders — neither
  supports training/fine-tuning). Scope out whether to add fine-tuning
  support to one of the existing retriever wrappers or write a standalone
  script.
- [ ] Needs a general-purpose (non-medical, non-clinical-trial) QA
  dataset + corpus, matching Long et al. (2024)'s own setup, plus a
  data-poisoning-ratio parameter threaded through generation, training,
  and defense evaluation.
- [ ] Most tractable as a `--backdoor_ratio` sweep option added to
  `medqa_rag`'s pipeline (reusing its existing Contriever + DRS/baseline
  wiring) if a general-domain corpus is swapped in, rather than a fully
  new use case — but the retriever-poisoning mechanism itself (not just
  corpus injection) is still new work regardless of which use case hosts
  it.

## Gap 4: MedMCQA, PubMedQA, Textbook corpus, MedCPT retriever for `medqa_rag` (partially covered — the most tractable gap)

`medqa_rag` already has the PoisonedRAG attack, DRS, and all three
baseline defenses working end-to-end — it's missing 5 of the paper's 6
(retriever × query-set) combinations in Table 4, and uses a different
corpus (PubMed) than the one the table's results are reported against
(Textbook, ~126K docs, Jin et al., 2021).

- [ ] Add `MedMCQADataset`/`PubMedQADataset` loaders alongside the
  existing `src/medrag_repro/data/medqa_loader.py`, following its exact
  shape (writes `{name}_all.jsonl`/`targets.jsonl`/`clean_queries.jsonl`,
  same as `prepare_data.py`'s current MedQA-US step).
- [ ] Add a Textbook corpus loader alongside `pubmed_loader.py` (Jin et
  al., 2021's medical textbook corpus, ~126K documents) as a
  `configs/*.yaml`-selectable alternative to the PubMed corpus already
  supported — `paths.pubmed_corpus` would need to become a more general
  `paths.corpus` (or a second `paths.textbook_corpus` key) once there's a
  second corpus option.
- [ ] Wire MedCPT in as an alternative to Contriever in
  `src/medrag_repro/retriever/`. `trial_retrieval` already has a working
  MedCPT encoder (`poisonrag_experiment/retrieval_utils.py`'s
  `MedCPTQueryEncoder`/`build_medcpt_corpus_index`) — worth checking
  whether that logic can move to `infra/` as a shared retriever module
  (it currently isn't; `infra/` only holds LLM-client and dataset-I/O
  code, not retriever code) rather than being duplicated a second time
  in `medqa_rag`.
- [ ] Once all three query sets and both retrievers are available,
  `scripts/run_defense.py --method all` (already supports comparing all
  four defenses in one run — see `docs/drs-dual-pca-analysis.md`) would
  need a `--query_set`/`--retriever` selector to reproduce each of
  Table 4's 6 rows without needing 6 separate configs.

## Gap 5: DRS-regularized attack (not covered)

The paper's own second contribution (Section 4.2, evaluated in Table 5):
an attack algorithm that adds `lambda_2 * DRS(poisoned data)` as a
regularization term to an existing attack's objective, trading some attack
success rate for a substantially lower DRS detection rate (demonstrated
against AgentPoison specifically). No attack in this repo does this —
`rag_attacks.poisonedrag_medqa` and `rag_attacks.poisonedrag_trial` both
optimize purely for attack success (generate → verify target LLM is
fooled → retry), with no awareness of DRS at generation time.

- [ ] Requires `drs_defense.core` (specifically `fit_drs`/`drs_score`) to
  be callable *during* poison generation, not just at defense-evaluation
  time — meaning the attack-generation code needs access to a fitted
  `DRSModel` on the target's clean reference embeddings while it's still
  producing candidates, not just after the fact.
- [ ] The paper's regularization is `min -O1 + lambda_1*O2 + lambda_2*DRS(poisoned)`
  where `O1`/`O2` are attack-specific (distance-to-adversarial-query,
  distance-within-poisoned-documents). For `rag_attacks.poisonedrag_medqa`'s
  generate → verify → retry loop specifically, the natural place to add
  this is scoring each candidate's DRS alongside its verification result,
  and preferring lower-DRS candidates among those that already pass
  verification — a re-ranking step, not necessarily a full gradient-based
  regularization (this repo's attacks are black-box/prompting-based, not
  gradient-optimized, unlike the paper's AgentPoison target).
- [ ] Would pair naturally with `--compare_defenses`/`--method all`:
  running the DRS-regularized variant and the original variant against
  the same defense comparison table would directly reproduce the paper's
  Table 5 shape (attack method × defense filtering rate).
- [ ] Note on scope: the paper's own Table 5 result is specific to the
  Agent-Driver task (Gap 1) attacked with AgentPoison (Gap 2) — "the
  results for the Agent-Driver task are summarized in Table 5." Fully
  reproducing Table 5 depends on both of those gaps being closed first.
  Applying the same regularization idea to this repo's existing
  PoisonedRAG-style attacks (`poisonedrag_medqa`/`poisonedrag_trial`)
  instead would be a faithful *application* of the paper's technique, not
  a reproduction of its specific reported numbers — worth doing either
  way, but keep the distinction clear when documenting results.

## Suggested order

1. **Gap 4** (MedMCQA/PubMedQA/Textbook/MedCPT for `medqa_rag`) — most
   tractable, reuses everything already built, no new attack/defense
   mechanism needed.
2. **Gap 5** (DRS-regularized attack) — self-contained addition to
   existing attack code, doesn't require a new use case or retriever
   training infrastructure.
3. **Gap 2** (AutoDAN/GCG) — resolve the scope-decision tension first,
   then AutoDAN before GCG if only one is prioritized.
4. **Gap 3** (BadDPR backdoor-ratio sweep) — needs new retriever
   fine-tuning infrastructure this repo doesn't have anywhere yet.
5. **Gap 1** (Agent-Driver use case) — largest scope, a genuinely new
   use case from scratch; reasonable to defer until the above are done
   and the pattern for "add a new use case" (see the root README's
   [Adding a new use case](README.md#adding-a-new-use-case) section) has
   been exercised on smaller additions first.
