import subprocess
import sys


def out2err(cmd):
    return subprocess.Popen(cmd, stdout=sys.stderr).wait()


def out2err_cut_first_line(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    first_line = True
    while True:
        line = p.stdout.readline()
        if not line:
            break
        if first_line:
            sys.stdout.write(line)
            first_line = False
        else:
            sys.stderr.write(line)
    return p.wait()


if __name__ == '__main__':
    mode, cmd = sys.argv[1], sys.argv[2:]
    run = out2err
    if mode in ('cl', 'ml'):
        # First line of cl.exe and ml64.exe stdout is useless: it prints input file
        run = out2err_cut_first_line
    sys.exit(run(cmd))
