import sys
import subprocess
import os
import collections


def is_clang(command):
    for word in command:
        if '--compiler-bindir' in word and 'clang' in word:
            return True

    return False


def main():
    spl = sys.argv.index('--cflags')
    command = sys.argv[1: spl]
    cflags = sys.argv[spl + 1:]

    dump_args = False
    if '--y_dump_args' in command:
        command.remove('--y_dump_args')
        dump_args = True

    executable = command[0]
    if not os.path.exists(executable):
        print >> sys.stderr, '{} not found'.format(executable)
        sys.exit(1)

    if is_clang(command):
        cflags.append('-Wno-unused-parameter')

    if not is_clang(command) and '-fopenmp=libomp' in cflags:
        cflags.append('-fopenmp')
        cflags.remove('-fopenmp=libomp')

    skip_list = [
        '-nostdinc++',  # CUDA uses system STL library
        '-gline-tables-only',
        # clang coverage
        '-fprofile-instr-generate',
        '-fcoverage-mapping',
        '/Zc:inline',  # disable unreferenced functions (kernel registrators) remove
        '-Wno-c++17-extensions',
    ]

    for flag in skip_list:
        if flag in cflags:
            cflags.remove(flag)

    skip_prefix_list = [
        '-fsanitize=',
        '-fsanitize-coverage=',
        '-fsanitize-blacklist=',
        '--system-header-prefix',
    ]
    for prefix in skip_prefix_list:
        cflags = [i for i in cflags if not i.startswith(prefix)]

    include_args = []
    compiler_args = []

    cflags_queue = collections.deque(cflags)
    while cflags_queue:
        arg = cflags_queue.popleft()
        if arg[:2].upper() in ('-I', '/I', '-B'):
            value = arg[2:]
            if not value:
                value = cflags_queue.popleft()
            if arg[1] == 'I':
                include_args.append('-I{}'.format(value))
            elif arg[1] == 'B':  # todo: delete "B" flag check when cuda stop to use gcc
                pass
        else:
            compiler_args.append(arg)

    command += include_args
    command += ['--compiler-options', ','.join(compiler_args)]

    if dump_args:
        sys.stdout.write('\n'.join(command))
    else:
        sys.exit(subprocess.Popen(command).wait())


if __name__ == '__main__':
    main()
