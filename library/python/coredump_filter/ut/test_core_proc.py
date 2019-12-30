#!/usr/bin/env python
# coding: utf-8

import sys
from library.python.coredump_filter import core_proc

from yatest import common as ytc


def test_stack():
    with open(ytc.test_source_path('data/stack1.txt')) as f:
        stack_lines = f.readlines()

    stack = core_proc.Stack(
        lines=stack_lines,
        stream=sys.stdout,
    )
    stack.parse()
    assert len(stack.frames) == 65, "Invalid frame count. "
    assert stack.frames[10].cropped_source() == "$S/search/cache/reqcacher.cpp", "Invalid cropped source. "
