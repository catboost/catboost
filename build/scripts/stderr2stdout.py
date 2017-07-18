import subprocess
import sys

if __name__ == '__main__':
    assert len(sys.argv) > 1
    sys.exit(subprocess.Popen(sys.argv[1:], stderr=sys.stdout).wait())
