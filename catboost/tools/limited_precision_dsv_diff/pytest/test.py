import pytest
import yatest.common

import os.path

PROJECT_REPOSITORY_PATH = os.path.join(
    "catboost", "tools", "limited_precision_dsv_diff")

PROGRAM_BINARY_PATH = yatest.common.binary_path(
    os.path.join(PROJECT_REPOSITORY_PATH, "limited_precision_dsv_diff")
)


def data_file(*path):
    return yatest.common.source_path(
        os.path.join(PROJECT_REPOSITORY_PATH, "pytest", "data", *path)
    )


def make_data_file(contents):
    filename = yatest.common.output_path(hex(hash(contents))[-8:])
    with open(filename, 'w') as f:
        f.write(contents)
    return filename


def local_canonical_file(*args, **kwargs):
    return yatest.common.canonical_file(*args, local=True, **kwargs)


def exec_test_diff_with_stdout(*args):
    cmd = (PROGRAM_BINARY_PATH,) + args
    stdout_file = yatest.common.test_output_path('stdout')
    with pytest.raises(yatest.common.ExecutionError):
        with open(stdout_file, 'w') as f:
            yatest.common.execute(cmd, stdout=f)

    return [local_canonical_file(stdout_file)]


def test_equal():
    cmd = (PROGRAM_BINARY_PATH, data_file('f1.tsv'), data_file('f1.tsv'), '--have-header')
    yatest.common.execute(cmd)


def test_totally_different():
    return exec_test_diff_with_stdout(data_file('f1.tsv'), data_file('f2.tsv'))


def test_cut():
    return exec_test_diff_with_stdout(data_file('f1.tsv'), data_file('f1_cut.tsv'), '--have-header')


def test_with_vs_without_header():
    return exec_test_diff_with_stdout(data_file('f2.tsv'), data_file('f2_without_header.tsv'))


def test_with_small_noise():
    return exec_test_diff_with_stdout(
        data_file('f1.tsv'),
        data_file('f1_with_small_noise.tsv'),
        '--have-header'
    )


def test_with_small_noise_within_precision():
    cmd = (
        PROGRAM_BINARY_PATH,
        data_file('f1.tsv'),
        data_file('f1_with_small_noise.tsv'),
        '--have-header',
        '--diff-limit', '1e-5'
    )
    yatest.common.execute(cmd)


def test_with_small_noise_outside_precision():
    return exec_test_diff_with_stdout(
        data_file('f2.tsv'),
        data_file('f2_with_small_noise.tsv'),
        '--have-header',
        '--diff-limit', '1e-10'
    )


def test_finite_vs_finite():
    return exec_test_diff_with_stdout(
        make_data_file('1\n'),
        make_data_file('1.0\n')
    )


def test_tiny_vs_tiny():
    return exec_test_diff_with_stdout(
        make_data_file('1e-400\n'),
        make_data_file('1e-500\n')
    )


def test_finite_vs_inf():
    return exec_test_diff_with_stdout(
        make_data_file('1\n'),
        make_data_file('1e400\n')
    )


def test_inf_vs_inf():
    return exec_test_diff_with_stdout(
        make_data_file('1e400\n'),
        make_data_file('1e500\n')
    )


def test_no_diff_limit_1():
    return exec_test_diff_with_stdout(
        make_data_file('1e400\n'),
        make_data_file('1e500\n'),
        '--diff-limit', '1e500'
    )


def test_no_diff_limit_2():
    return exec_test_diff_with_stdout(
        make_data_file('-1e400\n'),
        make_data_file('1e500\n'),
        '--diff-limit', '1e500'
    )
