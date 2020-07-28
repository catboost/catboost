import os
import sys
import platform
import subprocess


def fix_tmpdir(cmd):
    if not cmd:
        return cmd
    java = cmd[0]
    if not java.endswith('java') and not java.endswith('java.exe'):
        return cmd
    for arg in cmd:
        if arg.startswith('-Djava.io.tmpdir'):
            return cmd
    tmpdir = os.environ.get('TMPDIR') or os.environ.get('TEMPDIR')
    if not tmpdir:
        return cmd
    return cmd[0:1] + ['-Djava.io.tmpdir={}'.format(tmpdir)] + cmd[1:]


def just_do_it():
    args = fix_tmpdir(sys.argv[1:])
    if platform.system() == 'Windows':
        sys.exit(subprocess.Popen(args).wait())
    else:
        os.execv(args[0], args)


if __name__ == '__main__':
    just_do_it()
