import sys
import subprocess
import os
import re

rx_resource_dir = re.compile(r'libraries: =([^:]*)')

if __name__ == '__main__':
    args = sys.argv

    yndexer = args[1]
    arc_root = args[2]
    input_file = args[3]
    output_file = args[4]
    tail_args = args[5:]

    subprocess.check_call(tail_args)

    clang = tail_args[0]
    out = subprocess.check_output([clang, '-print-search-dirs'])
    resource_dir = rx_resource_dir.search(out).group(1)

    optional_build_root = ['-i', 'build::' + os.path.dirname(input_file)] if not input_file.startswith(arc_root) else []

    yndexer_args = [
        yndexer, input_file,
        '-pb2',
        '-i', 'arc::{}'.format(arc_root),
        '-i', '.IGNORE::/',
    ] + optional_build_root + [
        '-o', os.path.dirname(output_file),
        '-n', os.path.basename(output_file).rsplit('.ydx.pb2', 1)[0],
        '--'
    ] + tail_args + [
        '-resource-dir', resource_dir,
    ]

    os.execv(yndexer, yndexer_args)
