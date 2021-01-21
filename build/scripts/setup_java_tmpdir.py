import os
import sys
import platform
import subprocess


def fix_tmpdir(cmd):
    if not cmd:
        return cmd
    java_id, option_name = None, None
    for i, java in enumerate(cmd):
        if java.endswith('java') or java.endswith('java.exe'):
            java_id = i
            option_name = '-Djava.io.tmpdir='
            break
        if java.endswith('javac') or java.endswith('javac.exe'):
            java_id = i
            option_name = '-J-Djava.io.tmpdir='
            break
    if java_id is None:
        return cmd
    for arg in cmd[java_id:]:
        if arg.startswith(option_name):
            return cmd
    tmpdir = os.environ.get('TMPDIR') or os.environ.get('TEMPDIR')
    if not tmpdir:
        return cmd
    return cmd[:java_id + 1] + ['{}{}'.format(option_name, tmpdir)] + cmd[java_id + 1:]


def just_do_it():
    args = fix_tmpdir(sys.argv[1:])
    if platform.system() == 'Windows':
        sys.exit(subprocess.Popen(args).wait())
    else:
        os.execv(args[0], args)


if __name__ == '__main__':
    just_do_it()
