import json
import os
from common_helpers import *

def data_file(*path):
    return os.path.join(os.environ["DATA_PATH"], *path)


def local_canonical_file(path, diff_tool=None):
    with open("canonize", "a") as f:
        f.write(path)
        if diff_tool:
            f.write(" " + (diff_tool if isinstance(diff_tool, str) else ' '.join(diff_tool)))
        f.write("\n")
