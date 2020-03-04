# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division

import os
import re

import pytest

import yatest.common as yc


def clean_traceback(traceback):
    traceback = re.sub(r'\033\[(\d|;)+?m', '', traceback)  # strip ANSI codes
    traceback = re.sub(r' at 0x[0-9a-fA-F]+', '', traceback)  # remove object ids
    traceback = re.sub(r'line \d+', 'line XXX', traceback)  # remove line numbers
    return traceback


@pytest.mark.parametrize('mode', [
    'default',
    'ultratb_color',
    'ultratb_verbose',
])
@pytest.mark.parametrize('entry_point', [
    'main',
    'custom',
])
def test_traceback(mode, entry_point):
    tb_tool = yc.build_path('library/python/runtime/test/traceback/traceback')
    stdout_path = yc.test_output_path('stdout_raw.txt')
    stderr_path = yc.test_output_path('stderr_raw.txt')
    filtered_stdout_path = yc.test_output_path('stdout.txt')
    filtered_stderr_path = yc.test_output_path('stderr.txt')

    env = os.environ.copy()
    env.pop('PYTHONPATH', None)  # Do not let program peek into its sources on filesystem
    if entry_point == 'custom':
        env['Y_PYTHON_ENTRY_POINT'] = 'library.python.runtime.test.traceback.crash:main'

    proc = yc.execute(
        command=[tb_tool, mode],
        env=env,
        stdout=stdout_path,
        stderr=stderr_path,
        check_exit_code=False,
    )

    with open(filtered_stdout_path, 'w') as f:
        f.write(clean_traceback(proc.std_out))

    with open(filtered_stderr_path, 'w') as f:
        f.write(clean_traceback(proc.std_err))

    return {
        'stdout': yc.canonical_file(
            filtered_stdout_path,
            local=True,
        ),
        'stderr': yc.canonical_file(
            filtered_stderr_path,
            local=True,
        ),
    }
