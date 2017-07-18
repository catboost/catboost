import os
import yatest.common


def data_file(*path):
    return yatest.common.source_path(os.path.join("catboost", "pytest", "data", *path))


def local_canonical_file(*args, **kwargs):
    return yatest.common.canonical_file(*args, local=True, **kwargs)
