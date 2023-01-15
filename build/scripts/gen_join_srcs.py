import sys

import process_command_files as pcf


with open(sys.argv[1], 'w') as f:
    f.write('#if defined(__GNUC__)\n')
    f.write('#pragma GCC diagnostic ignored "-Wunknown-pragmas"\n')
    f.write('#if defined(__clang__)\n')
    f.write('#pragma GCC diagnostic ignored "-Wunknown-warning-option"\n')
    f.write('#endif\n')
    f.write('#pragma GCC diagnostic ignored "-Wsubobject-linkage"\n')
    f.write('#endif\n\n')

    for arg in pcf.iter_args(sys.argv[2:]):
        f.write('#include "' + arg + '"\n')
