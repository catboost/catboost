import sys
import os
import subprocess

if __name__ == '__main__':
    pos = sys.argv.index('--')
    fname = sys.argv[pos + 1]
    cmd = sys.argv[pos + 2 :]

    with open(fname, 'r') as f:
        args = [x.strip() for x in f]

    os.remove(fname)

    p = subprocess.Popen(cmd + args, shell=False, stderr=sys.stderr, stdout=sys.stdout)
    p.communicate()

    sys.exit(p.returncode)
