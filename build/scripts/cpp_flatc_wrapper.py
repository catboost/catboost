import os
import subprocess
import sys


def main():
    cmd = sys.argv[1:]
    h_file = None
    try:
        index = cmd.index('-o')
        h_file = cmd[index + 1]
        cmd[index + 1] = os.path.dirname(h_file)
    except (ValueError, IndexError):
        pass
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        if p.stdout:
            sys.stderr.write('stdout:\n{}\n'.format(p.stdout))
        if p.stderr:
            sys.stderr.write('stderr:\n{}\n'.format(p.stderr))
        sys.exit(p.returncode)
    if h_file and h_file.endswith(('.fbs.h', '.fbs64.h')):
        cpp_file = '{}.cpp'.format(h_file[:-2])
        with open(cpp_file, 'w') as f:
            f.write('#include "{}"\n'.format(os.path.basename(h_file)))
    sys.exit(0)


if __name__ == '__main__':
    main()
