import ast
import importlib.util
import marshal
import sys
import warnings

import _pytest
from _pytest.assertion.rewrite import rewrite_asserts

HEADER = importlib.util.MAGIC_NUMBER + b'pytest-' + _pytest.__version__.encode() + b'\n'


def expand_args(argv):
    args = []
    for arg in argv:
        if arg in ('--ya-start-command-file', '--ya-end-command-file'):
            continue
        if arg.startswith('@'):
            with open(arg[1:], encoding='utf-8') as f:
                args += [line for line in f.read().splitlines() if line]
        else:
            args.append(arg)
    return args


def rewrite(source, path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            tree = ast.parse(source.decode('utf-8'), filename=path)
            rewrite_asserts(tree, source, path, None)
            co = compile(tree, path, 'exec', dont_inherit=True)
        except Exception:
            return b''
    if caught:
        return b''
    return HEADER + marshal.dumps(co)


def main():
    args = expand_args(sys.argv[1:])
    n = len(args) // 3
    for srcx, src, out in zip(args[:n], args[n : 2 * n], args[2 * n :]):
        with open(src, 'rb') as f:
            data = rewrite(f.read(), srcx[:-1])
        with open(out, 'wb') as f:
            f.write(data)


if __name__ == '__main__':
    main()
