import os
import subprocess
import sys


def main():
    cmd = sys.argv[1:]
    try:
        index = cmd.index('-o')
        cmd[index+1] = os.path.dirname(cmd[index+1])
    except (ValueError, IndexError):
        pass
    p = subprocess.Popen(cmd, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode:
        if out:
            sys.stderr.write('stdout:\n{}\n'.format(out))
        if err:
            sys.stderr.write('stderr:\n{}\n'.format(err))
    sys.exit(p.returncode)


if __name__ == '__main__':
    main()
