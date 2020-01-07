import os
import shutil
import subprocess
import sys
import tempfile


SYMBOLS_TO_PATCH = (
    'connect',
    'poll',
    'recv',
    'recvfrom',
    'send',
    'sendto',
)

class Error(Exception):
    pass


def find_compiler(args):
    for arg in args:
        if os.path.basename(arg) in ('clang', 'clang++'):
            return arg
    raise Error('No known compiler found in the command line')


def find_libraries(project, args):
    if not project.endswith('/'):
        project = project + '/'

    for arg in args:
        if arg.startswith(project):
            yield arg


def rename_symbol(symbol):
    return 'green_{}'.format(symbol)


def patch_object(object_path, objcopy):
    args = [objcopy]
    for symbol in SYMBOLS_TO_PATCH:
        args.extend(('--redefine-sym', '{}={}'.format(symbol, rename_symbol(symbol))))
    args.append(object_path)
    subprocess.check_call(args)


def patch_library(library_path, ar, objcopy):
    tmpdir = tempfile.mkdtemp(dir=os.path.dirname(library_path))
    try:
        subprocess.check_call((ar, 'x', library_path), cwd=tmpdir)
        names = os.listdir(tmpdir)
        for name in names:
            patch_object(os.path.join(tmpdir, name), objcopy=objcopy)

        new_library_path = os.path.join(tmpdir, 'library.a')
        subprocess.check_call([ar, 'rcs', new_library_path] + names, cwd=tmpdir)

        os.rename(new_library_path, library_path)

    finally:
        shutil.rmtree(tmpdir)


def main():
    try:
        args = sys.argv[1:]
        compiler = find_compiler(args)
        compiler_dir = os.path.dirname(compiler)

        def get_tool(name):
            path = os.path.join(compiler_dir, name)
            if not os.path.exists(path):
                raise Error('No {} found alongside the compiler'.format(name))
            return path

        ar = get_tool('llvm-ar')
        objcopy = get_tool('llvm-objcopy')

        libraries = tuple(find_libraries('contrib/libs/libmysql_r', args))
        for library in libraries:
            library_path = os.path.abspath(library)
            if not os.path.exists(library_path):
                raise Error('No {} file exists'.format(library))

            patch_library(library_path, ar=ar, objcopy=objcopy)

    except Exception as error:
        name = os.path.basename(sys.argv[0])
        command = ' '.join(args)
        message = '{name} failed: {error}\nCommand line: {command}'
        print >> sys.stderr, message.format(**locals())


if __name__ == '__main__':
    main()
