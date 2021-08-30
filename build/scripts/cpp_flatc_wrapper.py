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
    sys.exit(subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr).wait())


if __name__ == '__main__':
    main()
