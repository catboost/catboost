import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "fails_on_gpu(how): mark test that fails only on GPU"
    )


@pytest.fixture(params=['CPU'])
def task_type(request):
    return request.param
