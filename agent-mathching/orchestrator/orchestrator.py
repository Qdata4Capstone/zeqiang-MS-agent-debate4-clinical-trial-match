import json
from typing import Dict, Any, List
from agents.proposer_agent import ProposerAgent
from agents.skeptic_agent import SkepticAgent


class Orchestrator:

    def __init__(self, proposer: ProposerAgent, skeptic: SkepticAgent, max_rounds: int = 2):
        self.proposer = proposer
        self.skeptic = skeptic
        self.max_rounds = max_rounds

    def run_one(self, criterion: Dict[str, Any], patient_sentences: List[str]) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print(f"### START LOOP for clause {criterion['clause_id']} ({criterion['type']}) ###")
        print("=" * 80)

        # Round 1: initial decision
        initial = self.proposer.decide(criterion=criterion, patient_sentences=patient_sentences)

        # Skeptic audit on initial
        audit = self.skeptic.audit(criterion=criterion, proposer_output=initial, patient_sentences=patient_sentences)
        print("\n================ SKEPTIC AUDIT (Round 1) ================")
        print(json.dumps(audit, indent=2, ensure_ascii=False))

        # Round 2: proposer revise using skeptic feedback
        revised = self.proposer.revise(
            criterion=criterion,
            patient_sentences=patient_sentences,
            prev_decision=initial,
            skeptic_report=audit
        )
        print("\n--- Proposer Revision Output (Decision Record v2) ---")
        print(json.dumps(revised, indent=2, ensure_ascii=False))

        return {"initial": initial, "audit": audit, "revised": revised}
