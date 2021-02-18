#!/usr/bin/env python
# coding: utf-8

import os
import sys
import unittest

import yatest.common

from library.python.coredump_filter import core_proc


class TestStackAuxiliary(unittest.TestCase):
    def setUp(self):
        self.data_dir = yatest.common.source_path('library/python/coredump_filter/tests/data')
        with open(os.path.join(self.data_dir, 'stack1.txt')) as f:
            stack_lines = f.readlines()

        self.stack = core_proc.Stack(
            lines=stack_lines,
            stream=sys.stdout,
        )

        self.stack.parse()

    def test_count_frames_in_stack(self):
        self.assertEqual(len(self.stack.frames), 65, 'Invalid frame count.')

    def test_cropped_source(self):
        source = self.stack.frames[10].cropped_source()
        self.assertEqual(source, '$S/search/cache/reqcacher.cpp', 'Invalid cropped source.')
