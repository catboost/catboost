import os
import json


def data_file(*path):
    return os.path.join(os.environ["DATA_PATH"], *path)


def local_canonical_file(path, diff_tool=None):
    with open("canonize", "a") as f:
        f.write(path)
        if diff_tool:
            f.write(" " + diff_tool)
        f.write("\n")


# TODO(exprmntr): this function is duplicated, remove it
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


def binary_path(*path):
    return os.path.join(os.environ["BINARY_PATH"], *path)


def test_output_path(*path):
    return os.path.join(os.getcwd(), *path)
