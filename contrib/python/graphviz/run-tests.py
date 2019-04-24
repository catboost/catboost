#!/usr/bin/env python
# run-tests.py

import sys

import pytest

ARGS = [
    #'--exitfirst',
    #'--pdb',
]

if 'idlelib' in sys.modules or 'thonny' in sys.modules:
    ARGS.extend(['--capture=sys', '--color=no'])
elif sys.version_info[0] == 2 and 'win_unicode_console' in sys.modules:
    ARGS.append('--capture=sys')

pytest.main(ARGS + sys.argv[1:])
