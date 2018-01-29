import os
import yatest.common
import json


def data_file(*path):
    return yatest.common.source_path(os.path.join("catboost", "pytest", "data", *path))


def local_canonical_file(*args, **kwargs):
    return yatest.common.canonical_file(*args, local=True, **kwargs)


def remove_time_from_json(filename):
    with open(filename) as f:
        log = json.load(f)
    iterations = log['iterations']
    for i, iter_info in enumerate(iterations):
        del iter_info['remaining_time']
        del iter_info['passed_time']
    with open(filename, 'w') as f:
        json.dump(log, f)
    return filename
