import os
import sys

try:
    from catboost_pytest_lib import compressed_data  # noqa
except ImportError:
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from lib import compressed_data  # noqa
    pytest_plugins = ["lib.common.pytest_plugin"]
