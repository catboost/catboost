#!/usr/bin/env python
import os
import sys


def mkdir_p(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    for directory in sys.argv[1:]:
        mkdir_p(directory)
