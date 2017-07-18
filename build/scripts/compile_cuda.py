import sys
import subprocess
import os


def is_clang(command):
    for word in command:
        if '--compiler-bindir' in word and 'clang' in word:
            return True

    return False

if __name__ == '__main__':
    spl = sys.argv.index('--cflags')
    command = sys.argv[1: spl]
    cflags = sys.argv[spl + 1:]

    executable = command[0]
    if not os.path.exists(executable):
        print >> sys.stderr, '{} not found'.format(executable)
        sys.exit(1)

    if not is_clang(command) and '-fopenmp=libomp' in cflags:
        cflags.append('-fopenmp')
        cflags.remove('-fopenmp=libomp')

    # CUDA uses system STL library.
    for flag in ('-nostdinc++', '-gline-tables-only'):
        if flag in cflags:
            cflags.remove(flag)

    command += ['--compiler-options', ','.join(cflags)]
    proc = subprocess.Popen(command, shell=False, stderr=sys.stderr, stdout=sys.stdout)
    proc.communicate()
    sys.exit(proc.returncode)
