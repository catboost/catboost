import sys

template = '''
extern "C" void PyImport_AppendInittab(const char* name, void (*fn)(void));
extern "C" void {1}();

namespace {
    struct TRegistrar {
        inline TRegistrar() {
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
        print >> sys.stderr, 'Usage: <path/to/gen_py_reg.py> <python_module_name> <output_file>'
        print >> sys.stderr, 'Passed: ' + ' '.join(sys.argv)
        sys.exit(1)

    with open(sys.argv[2], 'w') as f:
        modname = sys.argv[1]
        initname = 'init' + mangle(modname)
        code = template.replace('{0}', modname).replace('{1}', initname)
        f.write(code)
