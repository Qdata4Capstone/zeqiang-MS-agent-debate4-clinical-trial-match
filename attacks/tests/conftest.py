"""Make ``trial_retrieval`` importable for tests that exercise
``rag_attacks.poisonedrag_trial``, which imports ``get_conditions`` from
``trial_retrieval``'s ``poisonrag_experiment.retrieval_utils`` at module
level.

``poisonrag_experiment`` is not pip-installed anywhere in this repo; it's
normally only importable because ``use-cases/trial_retrieval/conftest.py``
marks ``use-cases/trial_retrieval/`` as pytest's rootdir when running that
subproject's own tests. Running ``pytest attacks/tests/`` on its own doesn't
get that for free, so mirror it here explicitly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "use-cases" / "trial_retrieval"))
