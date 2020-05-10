import argparse
import os
import subprocess


def call(cmd, cwd, env=None):
    # print >>sys.stderr, ' '.join(cmd)
    return subprocess.check_output(cmd, stdin=None, stderr=subprocess.STDOUT, cwd=cwd, env=env)


def postprocess(source_root, path):
    with open(path, 'r') as f:
        content = f.read()
        content = content.replace(source_root, '__ARCADIA_SOURCE_ROOT_PREFIX__/')
    with open(path, 'w') as f:
        f.write(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cgo1-files', nargs='+', required=True)
    parser.add_argument('--process-cgo1-go', action='store_true')
    parser.add_argument('--source-root', required=True)
    parser.add_argument('cgo1_cmd', nargs='*')
    args = parser.parse_args()

    call(args.cgo1_cmd, args.source_root)

    if args.process_cgo1_go:
        source_root = args.source_root + os.path.sep
        for path in args.cgo1_files:
            postprocess(source_root, path)
