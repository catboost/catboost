import argparse
import os
import subprocess
import sys


CGO1_SUFFIX='.cgo1.go'
FIXED_CGO1_SUFFIX='.fixed.cgo1.go'


def call(cmd, cwd, env=None):
    # print >>sys.stderr, ' '.join(cmd)
    return subprocess.call(cmd, stdin=None, stderr=sys.stderr, stdout=sys.stdout, cwd=cwd, env=env)


def process_cgo1_go(source_root, src_path, dst_path):
    with open(src_path, 'r') as src_file, open(dst_path, 'w') as dst_file:
        for line in src_file:
            if line.startswith('//'):
                dst_file.write(line.replace(source_root, '__ARCADIA_SOURCE_ROOT_PREFIX__'))
            else:
                dst_file.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cgo1-files', nargs='+', required=True)
    parser.add_argument('--process-cgo1-go', action='store_true')
    parser.add_argument('--source-root', required=True)
    parser.add_argument('cgo1_cmd', nargs='*')
    args = parser.parse_args()

    exit_code = call(args.cgo1_cmd, args.source_root)
    if exit_code != 0:
        sys.exit(exit_code)

    if args.process_cgo1_go:
        for dst_path in args.cgo1_files:
            assert dst_path.endswith(FIXED_CGO1_SUFFIX)
            src_path = '{}{}'.format(dst_path[:-len(FIXED_CGO1_SUFFIX)], CGO1_SUFFIX)
            process_cgo1_go(args.source_root, src_path, dst_path)
