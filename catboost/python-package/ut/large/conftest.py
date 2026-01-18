import os  # noqa
import sys  # noqa

import pytest

try:
    import catboost_pytest_lib  # noqa
except ImportError:
    sys.path.append(os.path.join(os.environ['CMAKE_SOURCE_DIR'], 'catboost', 'pytest'))
    pytest_plugins = ["lib.common.pytest_plugin"]


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "fails_on_gpu(how): mark test that fails only on GPU"
    )


@pytest.fixture(params=['CPU'])
def task_type(request):
    return request.param
