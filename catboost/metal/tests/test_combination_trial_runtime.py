"""Rejectable stochastic oracle overflow without poisoning accepted points."""

import ctypes as ct
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Combination requires actual Apple Silicon Metal",
)


@pytest.fixture(scope="module")
def trial_probe():
    root = Path(__file__).resolve().parents[1]
    source = Path(__file__).with_name("combination_runtime_probe.mm")
    digest = hashlib.sha256(source.read_bytes())
    for header in sorted((root / "native").glob("*.h")):
        digest.update(header.name.encode())
        digest.update(header.read_bytes())
    library = root / ".build" / f"combination_trial_{digest.hexdigest()[:20]}.dylib"
    library.parent.mkdir(exist_ok=True)
    if not library.exists():
        completed = subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
            "-framework", "Foundation", "-framework", "Metal", str(source), "-o", str(library)],
            capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
    loaded = ct.CDLL(str(library))
    operation = loaded.cbm_combination_trial_probe
    operation.argtypes = [ct.c_float, ct.c_uint32, ct.POINTER(ct.c_double), ct.c_char_p, ct.c_size_t]
    operation.restype = ct.c_int
    def evaluate(shift, trial):
        output = np.zeros(5, np.float64)
        error = ct.create_string_buffer(2048)
        status = operation(shift, trial, output.ctypes.data_as(ct.POINTER(ct.c_double)), error, len(error))
        return status, error.value.decode(), output
    return evaluate


@pytest.mark.parametrize("shift", [np.inf, -np.inf, np.nan, np.finfo(np.float32).max])
def test_nonfinite_trial_is_rejected_and_finite_point_recovers(trial_probe, shift):
    status, error, output = trial_probe(shift, True)
    assert status == 0, error
    assert np.isfinite(output[0])
    assert not np.isfinite(output[1])  # rejected by the finite-value line-search predicate
    assert output[2] == output[0]
    assert output[3] == 3  # initial, rejected trial, accepted-point reevaluation
    assert output[4] > 0


@pytest.mark.parametrize("shift", [np.inf, -np.inf, np.nan])
def test_nonfinite_initial_oracle_remains_a_strict_error(trial_probe, shift):
    status, error, _ = trial_probe(shift, False)
    assert status != 0
    assert "Combination" in error and "Nonfinite" in error
