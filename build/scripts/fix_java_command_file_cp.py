import sys
import os
import argparse
import subprocess
import platform


def fix_files(args):
    args = args[:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-root', default=None)
    args, tail = parser.parse_known_args(args)
    for idx, arg in list(enumerate(tail)):
        if arg.startswith('@') and os.path.isfile(arg[1:]):
            with open(arg[1:]) as f:
                fixed = [i.strip() for i in f]
                if args.build_root:
                    fixed = [os.path.join(args.build_root, i) for i in fixed]
                fixed = os.pathsep.join([i.strip() for i in fixed])
            fixed_name = list(os.path.splitext(arg))
            fixed_name[0] += '_fixed'
            fixed_name = ''.join(fixed_name)
            with open(fixed_name[1:], 'w') as f:
                f.write(fixed)
            tail[idx:idx + 1] = [fixed_name]
    return tail


if __name__ == '__main__':
    args = fix_files(sys.argv[1:])
    if platform.system() == 'Windows':
        sys.exit(subprocess.Popen(args).wait())
    else:
        os.execv(args[0], args)
