import pytest

from markupsafe import _native

try:
    from markupsafe import _speedups
except ImportError:
    _speedups = None  # type: ignore


@pytest.fixture(
    scope="session",
    params=(
        _native,
        pytest.param(
            _speedups,
            marks=pytest.mark.skipif(_speedups is None, reason="speedups unavailable"),
        ),
    ),
)
def _mod(request):
    return request.param


@pytest.fixture(scope="session")
def escape(_mod):
    return _mod.escape


@pytest.fixture(scope="session")
def escape_silent(_mod):
    return _mod.escape_silent


@pytest.fixture(scope="session")
def soft_str(_mod):
    return _mod.soft_str
