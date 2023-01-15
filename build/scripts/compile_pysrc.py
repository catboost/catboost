import argparse
import os
import shutil
import subprocess
import tarfile


LIMIT = 6000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rescompiler', required=True)
    subparsers = parser.add_subparsers(dest='mode')

    parser_py2 = subparsers.add_parser('py2')
    parser_py2.add_argument('--py_compile', required=True)
    parser_py2.add_argument('--python', required=True)

    parser_py3 = subparsers.add_parser('py3')
    parser_py3.add_argument('--pycc', required=True)

    return parser.parse_args()


def call(cmd, cwd=None, env=None):
    return subprocess.check_output(cmd, stdin=None, stderr=subprocess.STDOUT, cwd=cwd, env=env)


def iterate_py2_resource_params(py_files):
    for py in py_files:
        mod = py[:-3].replace('/', '.')
        key = '/py_modules/{}'.format(mod)
        yield py, key
        yield '-', 'resfs/src/{}={}'.format(key, py)
        yield '{}.yapyc'.format(py), '/py_code/{}'.format(mod)


def iterate_py3_resource_params(py_files):
    for py in py_files:
        for ext in ('', '.yapyc3'):
            path = '{}{}'.format(py, ext)
            dest = 'py/{}'.format(path)
            key = 'resfs/file/{}'.format(dest)
            src = 'resfs/src/{}={}'.format(key, os.path.basename(path))
            yield '-', src
            yield path, key


def main():
    args = parse_args()

    names = []
    with tarfile.open(args.input, 'r') as tar:
        names = tar.getnames()
        tar.extractall()

    if args.mode == 'py3':
        pycc_cmd = [args.pycc]
        pycc_ext = '.yapyc3'
        iterate_resource_params = iterate_py3_resource_params
    else:
        pycc_cmd = [args.python, args.py_compile]
        pycc_ext = '.yapyc'
        iterate_resource_params = iterate_py2_resource_params

    py_files = sorted(names)

    for py in py_files:
        cmd = pycc_cmd + ['{}-'.format(os.path.basename(py)), py, '{}{}'.format(py, pycc_ext)]
        call(cmd)

    outputs = []
    cmd = [args.rescompiler, '{}.0'.format(args.output)]
    size = 0
    for path, key in iterate_resource_params(py_files):
        addendum = len(path) + len(key)
        if size + addendum > LIMIT and len(cmd) > 2:
            call(cmd)
            outputs.append(cmd[1])
            cmd[1] = '{}.{}'.format(args.output, len(outputs))
            cmd = cmd[0:2]
            size = 0
        cmd.extend([path, key])
        size += addendum
    if len(outputs) == 0:
        cmd[1] = args.output
        call(cmd)
    else:
        call(cmd)
        outputs.append(cmd[1])
        with open(args.output, 'w') as fout:
            for fname in outputs:
                with open(fname, 'r') as fin:
                    shutil.copyfileobj(fin, fout)


if __name__ == '__main__':
    main()
