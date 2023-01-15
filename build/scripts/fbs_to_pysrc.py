import argparse
import os
import subprocess
import tarfile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flags', nargs='*')
    parser.add_argument('--flatc', required=True)
    parser.add_argument('--input', nargs='*', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--work-dir', required=True)

    return parser.parse_args()


def call(cmd, cwd=None, env=None):
    # sys.stderr.write('{}\n'.format(' '.join(cmd)))
    return subprocess.check_output(cmd, stdin=None, stderr=subprocess.STDOUT, cwd=cwd, env=env)


def main():
    args = parse_args()

    cmd = [args.flatc, '--python', '-o', args.work_dir] + args.flags + args.input
    call(cmd)

    py_srcs = []
    for root, _, files in os.walk(args.work_dir):
        for f in files:
            if f.endswith('.py'):
                py_srcs.append(os.path.join(root, f))

    with tarfile.open(args.output, 'w') as out:
        for f in py_srcs:
            out.add(f, arcname=os.path.relpath(f, args.work_dir))


if __name__ == '__main__':
    main()
