from agents.proposer_agent import ProposerAgent
from agents.skeptic_agent import SkepticAgent
from orchestrator.orchestrator import Orchestrator
from utils.tools import build_evidence_pool, parse_criteria

# ---------------- data ----------------
patient = {
    "_id": "sigir-20141",
    "text": (
        "A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain "
        "that began two days earlier for the first time in her life. "
        "The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and "
        "mild dyspnea, but is not increased on inspiration. "
        "The latest episode of pain ended half an hour prior to her arrival. "
        "She is known to have hypertension and obesity. "
        "She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease. "
        "She currently takes no medications. "
        "Physical examination is normal. "
        "The EKG shows nonspecific changes."
    )
}

trial = {
    "_id": "NCT00373828",
    "title": "Non-cardiac Chest Pain Evaluation and Treatment Study (CARPA) - Part 1: Diagnosis.",
    "text": (
        "Inclusion criteria:\n"
        "Acute episode of chest pain of less than 7 days duration as primary reason for admission to a chest pain clinic.\n"
        "Able to read and understand Danish.\n"
        "\n"
        "Exclusion criteria:\n"
        "Percutaneous Coronary Intervention.\n"
        "Coronary Artery Bypass Grafting.\n"
        "Other disease, diagnosed during this admission, which is likely to have caused the acute episode of chest pain."
    )
}

if __name__ == "__main__":
    # 1) Build patient sentences (evidence pool)
    patient_sentences = build_evidence_pool(patient["text"])

    # 2) Parse trial text to criteria
    criteria = parse_criteria(trial["text"])

    # 3) Init agents and orchestrator
    proposer = ProposerAgent(model_name="qwen2.5:7b-instruct")
    skeptic = SkepticAgent(llm_model="qwen2.5:7b-instruct")
    orch = Orchestrator(proposer, skeptic, max_rounds=1)

    # 4) Run a few clauses as a demo (or loop all)
    for crit in criteria:
        _ = orch.run_one(criterion=crit, patient_sentences=patient_sentences)

    # aggregate all criteria results/score
    # evaluate overall trial eligibility