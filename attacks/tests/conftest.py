"""Make ``Retrieving_stage`` importable for tests that exercise
``rag_attacks.poisonedrag_trial``, which imports ``get_conditions`` from
``Retrieving_stage``'s ``poisonrag_experiment.retrieval_utils`` at module
level.

``poisonrag_experiment`` is not pip-installed anywhere in this repo; it's
normally only importable because ``Retrieving_stage/conftest.py`` marks
``Retrieving_stage/`` as pytest's rootdir when running that subproject's own
tests. Running ``pytest attacks/tests/`` on its own doesn't get that for
free, so mirror it here explicitly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Retrieving_stage"))
