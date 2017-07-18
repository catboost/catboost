import os
import sys
import subprocess


if __name__ == '__main__':
    with open(os.devnull, 'w') as fnull:
        p = subprocess.Popen(sys.argv[1:], shell=False, stderr=fnull, stdout=sys.stdout)

    p.communicate()
    sys.exit(p.returncode)
