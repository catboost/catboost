try:
    from catboost_pytest_lib import compressed_data  # noqa
except ImportError:
    from lib import compressed_data  # noqa
    pytest_plugins = ["lib.common.pytest_plugin"]
