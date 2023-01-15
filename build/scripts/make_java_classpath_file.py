import os
import sys

import process_command_files as pcf


def make_cp_file(args):
    source = args[0]
    destination = args[1]
    with open(source) as src:
        lines = [l.strip() for l in src if l.strip()]
    with open(destination, 'w') as dst:
        dst.write(os.pathsep.join(lines))

def make_cp_file_from_args(args):
    destination = args[0]
    with open(destination, 'w') as dst:
        dst.write(os.pathsep.join(args[1:]))


if __name__ == '__main__':
    args = pcf.get_args(sys.argv[1:])
    if sys.argv[1] != '--from-args':
        make_cp_file(args)
    else:
        make_cp_file_from_args(args[1:])
