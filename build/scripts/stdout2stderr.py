import subprocess
import sys

FILE_PARAM = '--file='

if __name__ == '__main__':
    i = 1
    stdout = sys.stderr
    if len(sys.argv) > i and sys.argv[i].startswith(FILE_PARAM):
        file_name = sys.argv[i][len(FILE_PARAM) :]
        stdout = open(file_name, "w")
        i += 1
    assert len(sys.argv) > i and not sys.argv[i].startswith(FILE_PARAM)
    sys.exit(subprocess.Popen(sys.argv[i:], stdout=stdout).wait())
