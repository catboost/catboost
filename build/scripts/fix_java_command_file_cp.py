import sys
import os


def fix_files(args):
    args = args[:]
    for idx, arg in list(enumerate(args)):
        if arg.startswith('@') and os.path.isfile(arg[1:]):
            with open(arg[1:]) as f:
                fixed = os.pathsep.join([i.strip() for i in f])
            fixed_name = list(os.path.splitext(arg))
            fixed_name[0] += '_fixed'
            fixed_name = ''.join(fixed_name)
            with open(fixed_name[1:], 'w') as f:
                f.write(fixed)
            args[idx:idx + 1] = [fixed_name]
    return args


if __name__ == '__main__':
    args = fix_files(sys.argv[1:])
    os.execv(args[0], args)
