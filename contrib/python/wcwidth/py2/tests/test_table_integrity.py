"""
Executes verify-table-integrity.py as a unit test.
"""
import os
import sys
import subprocess

import pytest

@pytest.mark.skipif(sys.version_info[:2] != (3, 12), reason='Test only with a single version of python')
def test_verify_table_integrity():
    subprocess.check_output([sys.executable, os.path.join(os.path.dirname(__file__),
                                                          os.path.pardir,
                                                          'bin',
                                                          'verify-table-integrity.py')])