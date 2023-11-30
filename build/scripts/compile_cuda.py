import sys
import subprocess
import os
import collections
import re
import tempfile


def is_clang(command):
    for word in command:
        if '--compiler-bindir' in word and 'clang' in word:
            return True

    return False


def main():
    try:
        sys.argv.remove('--y_skip_nocxxinc')
        skip_nocxxinc = True
    except ValueError:
        skip_nocxxinc = False

    spl = sys.argv.index('--cflags')
    cmd = 1
    mtime0 = None
    if sys.argv[1] == '--mtime':
        mtime0 = sys.argv[2]
        cmd = 3
    command = sys.argv[cmd:spl]
    cflags = sys.argv[spl + 1 :]

    dump_args = False
    if '--y_dump_args' in command:
        command.remove('--y_dump_args')
        dump_args = True

    executable = command[0]
    if not os.path.exists(executable):
        print >> sys.stderr, '{} not found'.format(executable)
        sys.exit(1)

    if is_clang(command):
        # nvcc concatenates the sources for clang, and clang reports unused
        # things from .h files as if they they were defined in a .cpp file.
        cflags += ['-Wno-unused-function', '-Wno-unused-parameter']

    if not is_clang(command) and '-fopenmp=libomp' in cflags:
        cflags.append('-fopenmp')
        cflags.remove('-fopenmp=libomp')

    skip_list = [
        '-gline-tables-only',
        # clang coverage
        '-fprofile-instr-generate',
        '-fcoverage-mapping',
        '/Zc:inline',  # disable unreferenced functions (kernel registrators) remove
        '-Wno-c++17-extensions',
        '-flto',
        '-faligned-allocation',
        '-fsized-deallocation',
        # While it might be reasonable to compile host part of .cu sources with these optimizations enabled,
        # nvcc passes these options down towards cicc which lacks x86_64 extensions support.
        '-msse2',
        '-msse3',
        '-mssse3',
        '-msse4.1',
        '-msse4.2',
    ]

    if skip_nocxxinc:
        skip_list.append('-nostdinc++')

    for flag in skip_list:
        if flag in cflags:
            cflags.remove(flag)

    skip_prefix_list = [
        '-fsanitize=',
        '-fsanitize-coverage=',
        '-fsanitize-blacklist=',
        '--system-header-prefix',
    ]
    new_cflags = []
    for flag in cflags:
        if all(not flag.startswith(skip_prefix) for skip_prefix in skip_prefix_list):
            if flag.startswith('-fopenmp-version='):
                new_cflags.append(
                    '-fopenmp-version=45'
                )  # Clang 11 only supports OpenMP 4.5, but the default is 5.0, so we need to forcefully redefine it.
            else:
                new_cflags.append(flag)
    cflags = new_cflags

    if not is_clang(command):

        def good(arg):
            if arg.startswith('--target='):
                return False
            return True

        cflags = filter(good, cflags)

    cpp_args = []
    compiler_args = []

    # NVCC requires particular MSVC versions which may differ from the version
    # used to compile regular C++ code. We have a separate MSVC in Arcadia for
    # the CUDA builds and pass it's root in $Y_VC_Root.
    # The separate MSVC for CUDA may absent in Yandex Open Source builds.
    vc_root = os.environ.get('Y_VC_Root')

    cflags_queue = collections.deque(cflags)
    while cflags_queue:
        arg = cflags_queue.popleft()
        if arg == '-mllvm':
            compiler_args.append(arg)
            compiler_args.append(cflags_queue.popleft())
            continue
        if arg[:2].upper() in ('-I', '/I', '-B'):
            value = arg[2:]
            if not value:
                value = cflags_queue.popleft()
            if arg[1] == 'I':
                cpp_args.append('-I{}'.format(value))
            elif arg[1] == 'B':  # todo: delete "B" flag check when cuda stop to use gcc
                pass
            continue

        match = re.match(r'[-/]D(.*)', arg)
        if match:
            define = match.group(1)
            # We have C++ flags configured for the regular C++ build.
            # There is Y_MSVC_INCLUDE define with a path to the VC header files.
            # We need to change the path accordingly when using a separate MSVC for CUDA.
            if vc_root and define.startswith('Y_MSVC_INCLUDE'):
                define = os.path.expandvars('Y_MSVC_INCLUDE={}/include'.format(vc_root))
            cpp_args.append('-D' + define.replace('\\', '/'))
            continue

        compiler_args.append(arg)

    command += cpp_args
    if compiler_args:
        command += ['--compiler-options', ','.join(compiler_args)]

    # --keep is necessary to prevent nvcc from embedding nvcc pid in generated
    # symbols.  It makes nvcc use the original file name as the prefix in the
    # generated files (otherwise it also prepends tmpxft_{pid}_00000000-5), and
    # cicc derives the module name from its {input}.cpp1.ii file name.
    command += ['--keep', '--keep-dir', tempfile.mkdtemp(prefix='compile_cuda.py.')]
    # nvcc generates symbols like __fatbinwrap_{len}_{basename}_{hash} where
    # {basename} is {input}.cpp1.ii with non-C chars translated to _, {len} is
    # {basename} length, and {hash} is the hash of first exported symbol in
    # {input}.cpp1.ii if there is one, otherwise it is based on its modification
    # time (converted to string in the local timezone) and the current working
    # directory.  To stabilize the names of these symbols we need to fix mtime,
    # timezone, and cwd.
    if mtime0:
        os.environ['LD_PRELOAD'] = mtime0
    os.environ['TZ'] = 'UTC0'  # POSIX fixed offset format.
    os.environ['TZDIR'] = '/var/empty'  # Against counterfeit /usr/share/zoneinfo/$TZ.

    if dump_args:
        sys.stdout.write('\n'.join(command))
    else:
        sys.exit(subprocess.Popen(command, stdout=sys.stderr, stderr=sys.stderr, cwd='/').wait())


if __name__ == '__main__':
    main()
