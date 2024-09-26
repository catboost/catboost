from __future__ import print_function
import sys

template = '''
struct PyObject;
extern "C" int PyImport_AppendInittab(const char* name, PyObject* (*initfunc)());
extern "C" PyObject* {1}();

namespace {
    struct TRegistrar {
        inline TRegistrar() {
            // TODO Collect all modules and call PyImport_ExtendInittab once
            PyImport_AppendInittab("{0}", {1});
        }
    } REG;
}
'''


def mangle(name):
    if '.' not in name:
        return name
    return ''.join('{}{}'.format(len(s), s) for s in name.split('.'))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: <path/to/gen_py_reg.py> <python_module_name> <output_file>', file=sys.stderr)
        print('Passed: ' + ' '.join(sys.argv), file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[2], 'w') as f:
        modname = sys.argv[1]
        initname = 'PyInit_' + mangle(modname)
        code = template.replace('{0}', modname).replace('{1}', initname)
        f.write(code)
