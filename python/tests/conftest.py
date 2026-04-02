"""
conftest.py -- Shared test fixtures for CatBoost-MLX test suite.

Provides reusable fixtures for binary path discovery and binary
availability checking, so individual test files don't need to
duplicate this logic.
"""

import os

import pytest

# REPO_ROOT: Two directories up from tests/ -> python/ -> repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BINARY_PATH = REPO_ROOT


@pytest.fixture(scope="session")
def check_binaries():
    """Session-scoped fixture that skips if C++ binaries are not compiled."""
    csv_train = os.path.join(BINARY_PATH, "csv_train")
    csv_predict = os.path.join(BINARY_PATH, "csv_predict")
    if not (os.path.isfile(csv_train) and os.path.isfile(csv_predict)):
        pytest.skip("Compiled csv_train/csv_predict binaries not found at repo root")


@pytest.fixture
def binary_path():
    """Return the path to the directory containing compiled binaries."""
    return BINARY_PATH
