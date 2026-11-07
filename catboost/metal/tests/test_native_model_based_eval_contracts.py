"""Analysis admission depends on the saved baseline, not unused fit settings."""

import os

import pytest

from test_native_model_based_eval import (
    analysis_command, cli, digest, execute, fit_baseline, fold_histories,
    problem, settings,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the rebuilt native Metal model-based-eval CLI",
)


def test_baseline_loading_does_not_budget_unused_training_iterations(cli, tmp_path):
    data = problem(tmp_path / "data")
    config = settings()
    run = tmp_path / "run"
    snapshot, _ = fit_baseline(cli, data, run, config)
    before = digest(snapshot)
    # Analysis loads eight saved trees and grows two per experiment. The fit
    # iteration option must not reserve space for a billion new baseline trees.
    execute(analysis_command(cli, data, run, config | {"-i": 1_000_000_000},
                             snapshot, "tested", offset=4, count=2, size=2, full=True))
    fold_histories(run, [0], 2, 2)
    assert digest(snapshot) == before
