import os.path
import sys

from cffi.setuptools_ext import execfile

usage = '''Usage: {} INPUT:VAR OUTPUT
Generate CFFI C module source code.

INPUT is a source .py file.
VAR is a cffi.FFI() object defined in the source file.
OUTPUT is a .c or .cpp output file.
'''


def main():
    if len(sys.argv) != 3 or ':' not in sys.argv[1]:
        sys.stdout.write(usage.format(sys.argv[0]))
        sys.exit(1)

    mod_spec, c_file = sys.argv[1:3]
    build_file_name, ffi_var_name = mod_spec.rsplit(':', 1)

    source_dir = os.path.dirname(os.path.abspath(build_file_name))
    sys._MEIPASS = source_dir  # For pygit2.
    sys.dont_write_bytecode = True
    sys.path = [source_dir]
    mod_vars = {'__name__': '__cffi__', '__file__': build_file_name}
    execfile(build_file_name, mod_vars)

    ffi = mod_vars[ffi_var_name]
    if callable(ffi):
        ffi = ffi()
    ffi.emit_c_code(c_file)


if __name__ == '__main__':
    main()
