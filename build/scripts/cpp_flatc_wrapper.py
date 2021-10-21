import os
import subprocess
import sys


def main():
    cmd = sys.argv[1:]
    h_file = None
    try:
        index = cmd.index('-o')
        h_file = cmd[index+1]
        cmd[index+1] = os.path.dirname(h_file)
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
    if h_file and h_file.endswith(('.fbs.h', '.fbs64.h')):
        cpp_file = '{}.cpp'.format(h_file[:-2])
        with open(cpp_file, 'w') as f:
            f.write('#include "{}"\n'.format(os.path.basename(h_file)))
    sys.exit(0)


if __name__ == '__main__':
    main()
