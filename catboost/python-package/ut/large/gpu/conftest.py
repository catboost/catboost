import catboost.utils
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "fails_on_gpu(how): mark test that fails only on GPU"
    )


GPU_DEVICE_COUNT = catboost.utils.get_gpu_device_count()


def get_fails_on_gpu_mark(metafunc):
    """
    returns Mark or None
    """
    for pytestmark in getattr(metafunc.function, 'pytestmark', []):
        if pytestmark.name == 'fails_on_gpu':
            return pytestmark
    return None


def pytest_generate_tests(metafunc):
    have_param_task_type = 'task_type' in metafunc.fixturenames
    if have_param_task_type:
        if GPU_DEVICE_COUNT > 0:
            fails_on_gpu = get_fails_on_gpu_mark(metafunc)
            if fails_on_gpu:
                how = fails_on_gpu.kwargs.get('how', None)
                xfail_reason = 'Needs fixing on GPU' + (': ' + how if how else '')
                metafunc.parametrize(
                    'task_type',
                    [pytest.param('GPU', marks=pytest.mark.xfail(reason=xfail_reason))]
                )
            else:
                metafunc.parametrize('task_type', ['GPU'])
        else:
            metafunc.parametrize(
                'task_type',
                [pytest.param('GPU', marks=pytest.mark.skip(reason='No GPU devices available'))]
            )
    else:
        metafunc.function = pytest.mark.skip(reason='CPU test on GPU host')(metafunc.function)
