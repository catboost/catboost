import sys
import subprocess


if __name__ == '__main__':
    rc = subprocess.call(sys.argv[1:], shell=False, stderr=sys.stderr, stdout=sys.stdout)
    sys.exit(rc)
