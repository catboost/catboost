from __future__ import print_function

import os
import sys
import runpy


def _clean_env():
    for k, v in os.environ.items():
        if k.startswith('PYTHON'):
            del os.environ[k]


def _remove_user_site(paths):
    site_paths = ('site-packages', 'site-python')

    def is_site_path(path):
        for p in site_paths:
            if path.find(p) != -1:
                return True
        return False

    new_paths = list(paths)
    for p in paths:
        if is_site_path(p):
            new_paths.remove(p)

    return new_paths


def run():
    if len(sys.argv) >= 2:

        python_args = set()
        for i, arg in enumerate(sys.argv[1:]):
            if arg.startswith('-'):
                python_args.add(arg)
            else:
                script = arg
                sys.argv = sys.argv[i+1:]
                break

        possible_args = ['-B', '-E', '-s', '-S', '-c', '-m']  # order is important

        args_diff = python_args - set(possible_args)
        if args_diff:
            print('{arg} is not implemented in huge_python'.format(arg=args_diff), file=sys.stderr)

        for p_arg in possible_args:
            if p_arg not in python_args:
                continue

            if p_arg == '-B':
                os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
            elif p_arg == '-E':
                _clean_env()
            elif p_arg == '-s':
                pass
            elif p_arg == '-S':
                sys.path = _remove_user_site(sys.path)
            elif p_arg == '-c':
                exec(sys.argv[0])
                sys.exit(0)
            elif p_arg == '-m':
                runpy.run_module(sys.argv[0], run_name="__main__")
                sys.exit(0)

        sys.path.insert(0, os.path.dirname(script))
        runpy.run_path(script, run_name='__main__')
    else:
        raise NotImplementedError


if __name__ == '__main__':
    run()
