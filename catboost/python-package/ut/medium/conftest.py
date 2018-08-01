import pytest


@pytest.fixture(params=['CPU'])
def task_type(request):
    return request.param
