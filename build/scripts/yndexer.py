import sys
import subprocess
import os
import re

rx_resource_dir = re.compile(r'libraries: =([^:]*)')

if __name__ == '__main__':
    args = sys.argv

    yndexer = args[1]
    arc_root = args[2]
    build_root = args[3]
    input_file = args[4]
    output_file = args[5]
    tail_args = args[6:]

    subprocess.check_call(tail_args)

    clang = tail_args[0]
    out = subprocess.check_output([clang, '-print-search-dirs'])
    resource_dir = rx_resource_dir.search(out).group(1)

    yndexer_args = [
        yndexer, input_file,
        '-pb2',
        '-i', 'arc::{}'.format(arc_root),
        '-i', 'build::{}'.format(build_root),
        '-i', '.IGNORE::/',
        '-o', os.path.dirname(output_file),
        '-n', os.path.basename(output_file).rsplit('.ydx.pb2', 1)[0],
        '--'
    ] + tail_args + [
        '-resource-dir', resource_dir,
    ]

    os.execv(yndexer, yndexer_args)
