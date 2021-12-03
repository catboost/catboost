import os
import subprocess
import sys


def main():
    cmd = []
    build_root = sys.argv[1]
    length = len(build_root)
    is_dep = False
    for arg in sys.argv[2:]:
        if is_dep:
            is_dep = False
            if not arg.endswith('.tar.gz'):
                continue
            basename = os.path.basename(arg)
            assert arg.startswith(build_root) and len(arg) > length + len(basename) and arg[length] in ('/', '\\')
            cmd.extend(['--dep', '{}:{}:{}'.format(build_root, os.path.dirname(arg[length+1:]), basename)])
        elif arg == '--dep':
            is_dep = True
        else:
            cmd.append(arg)
    assert not is_dep
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
