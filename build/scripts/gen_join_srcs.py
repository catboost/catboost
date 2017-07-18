import sys


with open(sys.argv[1], 'w') as f:
    f.write('#if defined(__GNUC__)\n')
    f.write('#pragma GCC diagnostic ignored "-Wunknown-pragmas"\n')
    f.write('#pragma GCC diagnostic ignored "-Wsubobject-linkage"\n')
    f.write('#endif\n\n')

    for i in sys.argv[2:]:
        f.write('#include "' + i + '"\n')
