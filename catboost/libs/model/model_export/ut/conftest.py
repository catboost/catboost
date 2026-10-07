import os
import pathlib
import sys


try:
    import catboost_pytest_lib  # noqa
except ImportError:
    if not (repo_root := next((p for p in pathlib.Path(__file__).parents if (p / '.git').exists()), None)):
        raise RuntimeError("Git repository root not found")
    sys.path.append(os.path.join(str(repo_root.absolute()), 'catboost', 'pytest'))
    pytest_plugins = ["lib.common.pytest_plugin"]
