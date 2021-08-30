import argparse
import shutil
import subprocess
import sys


CGO1_SUFFIX='.cgo1.go'


def call(cmd, cwd, env=None):
    # sys.stderr.write('{}\n'.format(' '.join(cmd)))
    return subprocess.call(cmd, stdin=None, stderr=sys.stderr, stdout=sys.stdout, cwd=cwd, env=env)


def process_file(source_root, source_prefix, build_root, build_prefix, src_path, comment_prefix):
    dst_path = '{}.tmp'.format(src_path)
    with open(src_path, 'r') as src_file, open(dst_path, 'w') as dst_file:
        for line in src_file:
            if line.startswith(comment_prefix):
                dst_file.write(line.replace(source_root, source_prefix).replace(build_root, build_prefix))
            else:
                dst_file.write(line)
    shutil.move(dst_path, src_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-prefix', default='__ARCADIA_BUILD_ROOT_PREFIX__')
    parser.add_argument('--build-root', required=True)
    parser.add_argument('--cgo1-files', nargs='+', required=True)
    parser.add_argument('--cgo2-files', nargs='+', required=True)
    parser.add_argument('--source-prefix', default='__ARCADIA_SOURCE_ROOT_PREFIX__')
    parser.add_argument('--source-root', required=True)
    parser.add_argument('cgo1_cmd', nargs='*')
    args = parser.parse_args()

    exit_code = call(args.cgo1_cmd, args.source_root)
    if exit_code != 0:
        sys.exit(exit_code)

    for src_path in args.cgo1_files:
        process_file(args.source_root, args.source_prefix, args.build_root, args.build_prefix, src_path, '//')

    for src_path in args.cgo2_files:
        process_file(args.source_root, args.source_prefix, args.build_root, args.build_prefix, src_path, '#line')
