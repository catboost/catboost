import os
import sys

import pytest


PYTHON_PACKAGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


def _load_setup_module():
    """Import catboost's setup.py without invoking ``setup()``.

    ``setup()`` is guarded by ``if __name__ == '__main__':``, so importing the
    module only runs the top-level imports and function/class definitions.
    """
    sys.path.insert(0, PYTHON_PACKAGE_DIR)
    try:
        import setup as catboost_setup  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"setuptools/wheel not available: {exc}")
    finally:
        sys.path.pop(0)
    return catboost_setup


def test_setup_declares_repo_license_file():
    setup = _load_setup_module()

    # ``get_topsrc_dir()`` resolves to the repo root when building from a source
    # tree (no PKG-INFO alongside setup.py), and to ``catboost_all_src`` when
    # building from an sdist. In both cases the LICENSE lives next to it.
    topsrc_dir = setup.get_topsrc_dir()
    license_path = os.path.join(topsrc_dir, "LICENSE")
    assert os.path.isfile(license_path), f"LICENSE not found at {license_path}"

    # The value passed to setup(license_files=...) must be a path relative to
    # setup.py, because setuptools >= 77 rejects absolute license-file patterns.
    rel = os.path.relpath(license_path, setup.SETUP_DIR)
    assert rel == os.path.join("..", "..", "LICENSE")
    assert not os.path.isabs(rel)
