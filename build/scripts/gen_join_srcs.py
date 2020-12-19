import sys

import process_command_files as pcf


# Support @response-file notation for windows to reduce cmd length
if pcf.is_cmdfile_arg(sys.argv[1]):
    args = pcf.read_from_command_file(pcf.cmdfile_path(sys.argv[1]))
    sys.argv[:] = [sys.argv[0]] + args + sys.argv[2:]

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
