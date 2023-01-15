#!/usr/bin/env python
# -*- coding: utf-8 -*-
import minidump2core
import sys
import os


def test_minidump2core(tests_dir=None):
    if tests_dir is None:
        import yatest.common
        tests_dir = yatest.common.source_path('library/python/coredump_filter/tests')
    minidump_path = os.path.join(tests_dir, 'md3.txt')
    core_text = minidump2core.minidump_file_to_core(
        file_name=minidump_path,
    )
    assert len(core_text) >= 300, "Test failed: too small output in bytes: {}".format(len(core_text))
    assert 'Program terminated' in core_text, "Test failed: no signal message detected"
    threads = minidump2core.minidump_text_to_threads(
        minidump_text=open(minidump_path).read(),
    )
    assert len(threads) == 148, "Test failed: invaid thread count parsed: {}".format(len(threads))


if __name__ == '__main__':
    try:
        test_minidump2core(
            tests_dir='tests',
        )
    except AssertionError as e:
        print e
        sys.exit(1)
    print "Module minidump2core test passed"
