import platform
import sys
import os
import subprocess


def fix_windows(args):
    for arg in args:
        if os.path.basename(arg) == 'groovyc' and os.path.basename(os.path.dirname(arg)) == 'bin':
            yield arg + '.bat'
        else:
            yield arg


if __name__ == '__main__':
    env = os.environ.copy()
    jdk = sys.argv[1]
    env['JAVA_HOME'] = jdk
    args = sys.argv[2:]
    if platform.system() == 'Windows':
        sys.exit(subprocess.Popen(list(fix_windows(args)), env=env).wait())
    else:
        os.execve(args[0], args, env)
