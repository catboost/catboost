import os

lz4 = '''
#define LZ4_MEMORY_USAGE {i}
#define LZ4_NAMESPACE lz4_{i}
#include "lz4_ns.h"
'''.lstrip()

lz4methods = '''
#include "iface.h"

%s

extern "C" {

struct TLZ4Methods* LZ4Methods(int memory) {
    switch (memory) {
%s
    }

    return 0;
}

}
'''.lstrip()

lz4_namespace = 'namespace lz4_{i} {{ extern struct TLZ4Methods ytbl; }}'
lz4_case = '        case {i}: return &lz4_{i}::ytbl;'

namespaces = []
cases = []

os.chdir(os.path.dirname(__file__))

for i in range(10, 21):
    name = 'lz4_{}.cpp'.format(i)
    namespaces.append(lz4_namespace.format(i=i))
    cases.append(lz4_case.format(i=i))
    print '    ' + name

    with open(name, 'w') as f:
        f.write(lz4.format(i=i))

with open('lz4methods.cpp', 'w') as f:
    f.write(lz4methods % ('\n'.join(namespaces), '\n'.join(cases)))
