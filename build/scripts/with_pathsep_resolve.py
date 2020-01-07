import sys
import os
import subprocess
import platform


def fix_args(args):
    just_replace_it = False
    for arg in args:
        if arg == '--fix-path-sep':
            just_replace_it = True
            continue
        if just_replace_it:
            arg = arg.replace('::', os.pathsep)
            just_replace_it = False
        yield arg

if __name__ == '__main__':
    res = list(fix_args(sys.argv[1:]))
    if platform.system() == 'Windows':
        sys.exit(subprocess.Popen(res).wait())
    else:
        os.execv(res[0], res)
