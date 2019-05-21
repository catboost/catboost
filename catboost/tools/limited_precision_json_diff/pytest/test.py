import pytest
import yatest.common

import os.path

PROJECT_REPOSITORY_PATH = os.path.join(
    "catboost", "tools", "limited_precision_json_diff")

PROGRAM_BINARY_PATH = yatest.common.binary_path(
    os.path.join(PROJECT_REPOSITORY_PATH, "limited_precision_json_diff")
)


def data_file(*path):
    return yatest.common.source_path(
        os.path.join(PROJECT_REPOSITORY_PATH, "pytest", "data", *path)
    )


def local_canonical_file(*args, **kwargs):
    return yatest.common.canonical_file(*args, local=True, **kwargs)


def exec_test_diff_with_stdout(*args):
    cmd = (PROGRAM_BINARY_PATH,) + args
    stdout_file = yatest.common.test_output_path('stdout')
    with pytest.raises(yatest.common.ExecutionError):
        yatest.common.execute(cmd, stdout=open(stdout_file, 'w'))

    return [local_canonical_file(stdout_file)]


def test_equal():
    cmd = (PROGRAM_BINARY_PATH, data_file('f1.json'), data_file('f1.json'))
    yatest.common.execute(cmd)


def test_totally_different():
    return exec_test_diff_with_stdout(data_file('f1.json'), data_file('f2.json'))


def test_cut():
    return exec_test_diff_with_stdout(data_file('f1.json'), data_file('f1_cut.json'))


def test_with_small_noise():
    return exec_test_diff_with_stdout(data_file('f1.json'), data_file('f1_with_small_noise.json'))


def test_with_small_noise_within_precision():
    cmd = (
        PROGRAM_BINARY_PATH,
        data_file('f1.json'),
        data_file('f1_with_small_noise.json'),
        '--diff-limit', '1e-5'
    )
    yatest.common.execute(cmd)


def test_with_small_noise_outside_precision():
    return exec_test_diff_with_stdout(
        data_file('f2.json'),
        data_file('f2_with_small_noise.json'),
        '--diff-limit', '1e-10'
    )
