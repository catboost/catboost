import sys
import os
import errno
from os import listdir
from os.path import isfile, join

def ensure_dir_exists(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def headers_set(directory):
    ensure_dir_exists(join('.', directory))    
    return set([f for f in listdir(directory) if isfile(join('.', directory, f)) and f.endswith('.h')])

if __name__ == "__main__":

    python2_path = sys.argv[1]
    python3_path = sys.argv[2]
    output_path = sys.argv[3]

    ensure_dir_exists(join('.', python2_path))
    ensure_dir_exists(join('.', python3_path))
    make_dir(output_path)

    only_headers2 = headers_set(python2_path)
    only_headers3 = headers_set(python3_path)
    all_headers = only_headers2 | only_headers3

    for header in all_headers:
        f = open(join(output_path, header), 'w')
        f.write('#pragma once\n\n')
        f.write('#ifdef USE_PYTHON3\n')
        if (header in only_headers3):
            f.write('#include <' + join(python3_path, header) + '>\n')
        else:
            f.write('#error "No <' + header + '> in Python3"\n')
        f.write('#else\n')
        if (header in only_headers2):
            f.write('#include <' + join(python2_path, header) + '>\n')
        else:
            f.write('#error "No <' + header + '> in Python2"\n')
        f.write('#endif\n')

    