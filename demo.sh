#!/usr/bin/env bash
#
# demo.sh -- showcase the RAG data-poisoning attack/defense comparisons in
# this repo. See README.md's "Attack & defense showcase" table for what
# each use case demonstrates, and each use case's own README for full
# detail on the commands this script runs.
#
# Usage:
#   ./demo.sh                       # run every use case that can run in this environment
#   ./demo.sh trial_retrieval       # run just one use case
#   ./demo.sh medqa_rag
#   ./demo.sh strategyqa_agent      # prints instructions + a sample table only
#                                    # (needs its own conda env -- see below)
#   ./demo.sh --dry-run [use_case]  # print the command(s) and sample output, don't execute
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0
TARGET="all"

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      ;;
    trial_retrieval|medqa_rag|strategyqa_agent|all)
      TARGET="$arg"
      ;;
    -h|--help)
      sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

hr() { printf '%s\n' "--------------------------------------------------------------------"; }

check_ollama() {
  if ! curl -s --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Ollama doesn't seem to be running. Start it first:"
    echo "  ollama serve"
    echo "  ollama pull qwen2.5:7b-instruct"
    exit 1
  fi
  if ! curl -s http://localhost:11434/api/tags | grep -q "qwen2.5:7b-instruct"; then
    echo "qwen2.5:7b-instruct isn't pulled yet. Run: ollama pull qwen2.5:7b-instruct"
    exit 1
  fi
}

demo_trial_retrieval() {
  hr
  echo "trial_retrieval: synthetic poisoned trial-record injection into clinical-trial retrieval"
  echo "See use-cases/trial_retrieval/poisonrag_experiment/README.md for full detail."
  hr
  echo "Sample outcome (real output from a run with 3 target patients, --drs_ref_k 200):"
  cat <<'SAMPLE'
Method                  recall@50    recall@100   recall@200
-------------------------------------------------------------
clean (no attack)       0.7052       0.8941       0.9137
poisoned, no defense    0.7052       0.8941       0.9137
poisoned + DRS          0.7052       0.8941       0.9137
poisoned + L2-norm      0.7052       0.8941       0.9137
poisoned + L2-distance  0.7052       0.8941       0.9137
poisoned + perplexity   0.6190       0.8078       0.8275
SAMPLE
  echo
  echo "(Recall alone doesn't show it, but DRS's default reference-set strategy --"
  echo "pooling clean reference docs across all target patients, per the paper's actual"
  echo "Algorithm 2 (pass --no-drs_pool_reference for the old per-query behavior) --"
  echo "catches 3/3 injected poison docs here, vs. 1/3 without pooling, with identical"
  echo "recall either way. See docs/drs-dual-pca-analysis.md.)"
  echo
  echo "(Note: this command overrides --drs_ref_k to 200 but leaves"
  echo "--drs_num_directions at its low default (16), well under the paper's"
  echo "M=100. See poisonrag_experiment/README.md's \"Choosing --drs_ref_k and"
  echo "--drs_num_directions\" section and drs_defense/README.md's hyperparameter"
  echo "guidance before tuning this further -- M and reference-set size need to"
  echo "scale together, not independently.)"
  echo
  echo "Command (run from use-cases/trial_retrieval/):"
  echo "  python -m poisonrag_experiment.run_poisonrag_experiment \\"
  echo "    --corpus sigir --query_type gpt-4-turbo \\"
  echo "    --num_targets 3 --poisons_per_patient 1 \\"
  echo "    --ollama_model qwen2.5:7b-instruct \\"
  echo "    --output_dir results/demo \\"
  echo "    --drs_ref_k 200 --compare_defenses"
  if [ "$DRY_RUN" = "1" ]; then
    echo "(--dry-run: not executing. This takes ~5 min -- it encodes the 3.6k-doc SIGIR corpus with MedCPT.)"
    return
  fi
  echo
  echo "Running now (~5 min, encodes the SIGIR corpus with MedCPT)..."
  (
    cd "$REPO_ROOT/use-cases/trial_retrieval"
    python -m poisonrag_experiment.run_poisonrag_experiment \
      --corpus sigir --query_type gpt-4-turbo \
      --num_targets 3 --poisons_per_patient 1 \
      --ollama_model qwen2.5:7b-instruct \
      --output_dir results/demo \
      --drs_ref_k 200 --compare_defenses
  )
}

demo_medqa_rag() {
  hr
  echo "medqa_rag: PoisonedRAG black-box attack on a medical-QA RAG pipeline"
  echo "See use-cases/medqa_rag/README.md for full detail."
  hr
  echo "Sample outcome (real output from a tiny local run -- 3 targets, 300-doc PubMed corpus):"
  cat <<'SAMPLE'
Method          Detect rate     Clean FPR       Attack success  Retrieval F1
------------------------------------------------------------------------------
none            -               -               1.0000          0.3333
drs             0.0000          0.0345          1.0000          0.3333
l2_norm         0.3333          0.0690          0.6667          0.2353
l2_distance     0.3333          0.0345          0.6667          0.2353
perplexity      0.0000          0.0690          1.0000          0.3333
SAMPLE
  echo
  echo "(DRS loses to the baselines here because this demo's reference set is"
  echo "tiny (n=29 clean reference docs). scripts/sweep_reference_size.py sweeps"
  echo "reference-set size and M (configs/sweep.yaml, still a small local corpus"
  echo "but scaled toward the paper's M=100/300-clean-query setup) and finds DRS's"
  echo "detection power climbs to a PERFECT 3/3 -- at M=100, the paper's own"
  echo "value, no extra tuning -- once the pooled reference set reaches 326 docs"
  echo "(600 clean queries). Every baseline (l2_norm, l2_distance, perplexity)"
  echo "stays at 0/3-1/3 across the whole sweep. This demo just runs far below"
  echo "that threshold on purpose, to finish in ~1-2 min. See"
  echo "docs/drs-dual-pca-analysis.md's \"Crossover confirmed\" section.)"
  echo
  echo "Uses configs/demo.yaml, a small override of the real config (3"
  echo "targets, a 300-doc PubMed corpus, CPU device) so this runs in about"
  echo "a minute instead of the full pipeline's real scale."
  echo
  echo "Commands (run from use-cases/medqa_rag/):"
  echo "  export OPENAI_BASE_URL=http://127.0.0.1:11434/v1 OPENAI_API_KEY=ollama"
  echo "  python scripts/prepare_data.py --config configs/demo.yaml"
  echo "  python scripts/build_index.py --config configs/demo.yaml"
  echo "  python scripts/generate_poison.py --config configs/demo.yaml"
  echo "  python scripts/run_defense.py --config configs/demo.yaml --method all"
  if [ "$DRY_RUN" = "1" ]; then
    echo "(--dry-run: not executing.)"
    return
  fi
  echo
  echo "Running now (~1-2 min)..."
  (
    cd "$REPO_ROOT/use-cases/medqa_rag"
    export OPENAI_BASE_URL=http://127.0.0.1:11434/v1
    export OPENAI_API_KEY=ollama
    python scripts/prepare_data.py --config configs/demo.yaml
    python scripts/build_index.py --config configs/demo.yaml
    python scripts/generate_poison.py --config configs/demo.yaml
    python scripts/run_defense.py --config configs/demo.yaml --method all
  )
}

demo_strategyqa_agent() {
  hr
  echo "strategyqa_agent: backdoor-trigger document injection into a ReAct agent"
  echo "See use-cases/strategyqa_agent/README.md for full detail."
  hr
  echo "This use case needs its own conda env (Python 3.9, CUDA -- see"
  echo "use-cases/strategyqa_agent/README.md's Install section). This"
  echo "script does not attempt to run it automatically -- run it yourself:"
  echo
  echo "  cd use-cases/strategyqa_agent"
  echo "  conda activate agentpoison"
  echo "  python ReAct/run_strategyqa_inference.py \\"
  echo "    --backbone qwen --model dpr --task_type adversarial \\"
  echo "    --enable_drs --compare_defenses \\"
  echo "    --drs_num_directions 200 --drs_quantile 0.99 --drs_top_k 1 \\"
  echo "    --poison_injection_num 229"
  echo
  echo "Illustrative table format (not real run output -- see the README's note):"
  cat <<'SAMPLE'
Method       Detection rate   Clean FPR
---------------------------------------
DRS          0.9170           0.0123
L2-norm      0.1747           0.0100
L2-distance  0.3712           0.0080
Perplexity   0.6550           0.0210
SAMPLE
  echo
  echo "(--drs_num_directions 200 above doesn't match the paper's M=100 --"
  echo "see the README's Notes section. Total reference-set size here is"
  echo "roughly len(test set) * --drs_top_k, deduplicated -- --drs_top_k is"
  echo "the lever to raise first if detection looks weak, before raising"
  echo "--drs_num_directions. See drs_defense/README.md's hyperparameter"
  echo "guidance for why M and reference-set size need to scale together.)"
}

if [ "$TARGET" = "trial_retrieval" ] || [ "$TARGET" = "medqa_rag" ] || [ "$TARGET" = "all" ]; then
  check_ollama
fi

case "$TARGET" in
  trial_retrieval) demo_trial_retrieval ;;
  medqa_rag) demo_medqa_rag ;;
  strategyqa_agent) demo_strategyqa_agent ;;
  all)
    demo_trial_retrieval
    demo_medqa_rag
    demo_strategyqa_agent
    ;;
esac
