import os  # noqa
import sys  # noqa

try:
    from catboost_pytest_lib import compressed_data  # noqa
except ImportError:
    sys.path.append(os.path.join(os.environ['CMAKE_SOURCE_DIR'], 'catboost', 'pytest'))
    from lib import compressed_data  # noqa
