import catboost.utils
import pytest

GPU_DEVICE_COUNT = catboost.utils.get_gpu_device_count()


def pytest_generate_tests(metafunc):
    have_param_task_type = 'task_type' in metafunc.fixturenames
    if have_param_task_type:
        if GPU_DEVICE_COUNT > 0:
            fails_on_gpu = getattr(metafunc.function, 'fails_on_gpu', None)
            if fails_on_gpu:
                how = fails_on_gpu.kwargs.get('how', None)
                xfail_reason = 'Needs fixing on GPU' + (': ' + how if how else '')
                metafunc.parametrize('task_type', [pytest.mark.xfail('GPU', reason=xfail_reason)])
            else:
                metafunc.parametrize('task_type', ['GPU'])
        else:
            metafunc.parametrize('task_type', [pytest.mark.skip('GPU', reason='No GPU devices available')])
    else:
        metafunc.function = pytest.mark.skip(reason='CPU test on GPU host')(metafunc.function)
