import ymake
from _common import stripext, rootrel_arc_src


def is_arc_src(src, unit):
    return (
        src.startswith('${ARCADIA_ROOT}/') or
        src.startswith('${CURDIR}/') or
        unit.resolve_arc_path(src).startswith('$S/')
    )


def to_build_root(path, unit):
    if is_arc_src(path, unit):
        return '${ARCADIA_BUILD_ROOT}/' + rootrel_arc_src(path, unit)
    return path


def pb2_arg(path, mod, unit):
    return '{}_pb2.py={}_pb2'.format(stripext(to_build_root(path, unit)), mod)


def pb2_grpc_arg(path, mod, unit):
    return '{}_pb2_grpc.py={}_pb2_grpc'.format(stripext(to_build_root(path, unit)), mod)


def ev_arg(path, mod, unit):
    return '{}_ev.proto={}_ev'.format(stripext(to_build_root(path, unit)), mod)


def mangle(name):
    if '.' not in name:
        return name
    return ''.join('{}{}'.format(len(s), s) for s in name.split('.'))


def add_python_lint_checks(unit, files):
    if files and unit.get('LINT_LEVEL_VALUE') != "none":
        resolved_files = []
        for path in files:
            resolved = unit.resolve_arc_path([path])
            if resolved != path:  # path was resolved
                resolved_files.append(resolved)
        unit.onadd_check(["PEP8"] + resolved_files)
        unit.onadd_check(["PYFLAKES"] + resolved_files)


def py_program(unit):
    unit.onpeerdir(['library/python/runtime/main'])
    unit.onadd_check_py_imports()


def onpy_srcs(unit, *args):
    # Each file arg must either be a path, or "${...}/buildpath=modname", where
    # "${...}/buildpath" part will be used as a file source in a future macro,
    # and "modname" will be used as a module name.
    unit.onuse_python([])

    if '/library/python/runtime' not in unit.path():
        unit.onpeerdir(['library/python/runtime'])

    if unit.get('MODULE_TYPE') == 'PROGRAM':
        py_program(unit)

    ns = (unit.get('PY_NAMESPACE_VALUE') or unit.path()[3:].replace('/', '.')) + '.'
    cython_directives = []

    pyxs_c = []
    pyxs_cpp = []
    pyxs = pyxs_cpp
    pys = []
    protos = []
    evs = []
    swigs = []

    args = iter(args)
    for arg in args:
        # Namespace directives.
        if arg == 'TOP_LEVEL':
            ns = ''
        elif arg == 'NAMESPACE':
            ns = next(args) + '.'
        # Cython directives.
        elif arg == 'CYTHON_C':
            pyxs = pyxs_c
        elif arg == 'CYTHON_CPP':
            pyxs = pyxs_cpp
        elif arg == 'CYTHON_DIRECTIVE':
            cython_directives += ['-X', next(args)]
        # Unsupported but legal PROTO_LIBRARY arguments.
        elif arg == 'GLOBAL' or arg.endswith('.gztproto'):
            pass
        # Sources.
        else:
            if '=' in arg:
                path, mod = arg.split('=', 1)
            else:
                path = arg
                if arg == '__main__.py' or arg.endswith('/__main__.py'):
                    mod = '__main__'
                else:
                    mod = ns + stripext(arg).replace('/', '.')

            pathmod = (path, mod)

            if path.endswith('.py'):
                pys.append(pathmod)
            elif path.endswith('.pyx'):
                pyxs.append(pathmod)
            elif path.endswith('.proto'):
                protos.append(pathmod)
            elif path.endswith('.ev'):
                evs.append(pathmod)
            elif path.endswith('.swg'):
                swigs.append(path)  # ignore mod, use last (and only) ns
            else:
                ymake.report_configure_error('in PY_SRCS: unrecognized arg {!r}'.format(path))

    if pyxs:
        for pyxs, cython in [
            (pyxs_c, unit.onbuildwith_cython_c),
            (pyxs_cpp, unit.onbuildwith_cython_cpp),
        ]:
            for path, mod in pyxs:
                cython([
                    path,
                    '--module-name', mod,
                    '--init-name', 'init' + mangle(mod),
                ] + cython_directives)
                unit.onpy_register([mod])

    if pys:
        res = []

        for path, mod in pys:
            root_rel_path = rootrel_arc_src(path, unit)
            unit.onpy_compile_bytecode([root_rel_path + '-', path])

            source_key = '/py_modules/' + mod
            res += [
                path, source_key,
                path + '.yapyc', '/py_code/' + mod,
                # build map modname-filename via resource macro
                '-', "{}={}".format('/py_fs/' + mod, root_rel_path)
            ]

        unit.onresource(res)
        add_python_lint_checks(unit, [path for path, mod in pys])

    if protos:
        if '/contrib/libs/protobuf/python/google_lib' not in unit.path():
            unit.onpeerdir(['contrib/libs/protobuf/python/google_lib'])

        grpc = unit.get('GRPC_FLAG') == 'yes'

        if grpc:
            unit.onpeerdir(['contrib/libs/grpc/python'])

        unit.ongenerate_py_protos([path for path, mod in protos])
        unit.onpy_srcs([pb2_arg(path, mod, unit) for path, mod in protos])

        if grpc:
            unit.onpy_srcs([pb2_grpc_arg(path, mod, unit) for path, mod in protos])

    if evs:
        unit.onpy_srcs([ev_arg(path, mod, unit) for path, mod in evs])

    if swigs:
        unit.onsrcs(swigs)
        prefix = unit.get('MODULE_PREFIX')
        project = unit.get('REALPRJNAME')
        unit.onpy_register([prefix + project])
        path = '${ARCADIA_BUILD_ROOT}/' + '{}/{}.py'.format(unit.path()[3:], project)
        arg = '{}={}'.format(path, ns + project.replace('/', '.'))
        unit.onpy_srcs([arg])


def ontest_srcs(unit, *args):
    if unit.get('PYTEST_BIN') != 'no':
        unit.onpy_srcs(["NAMESPACE", "__tests__"] + list(args))


def onpy_register(unit, *args):
    for name in args:
        if '=' in name:
            fullname, shortname = name.split('=', 1)
            assert '.' not in shortname, shortname
            assert fullname == shortname or fullname.endswith('.' + shortname), fullname
            unit.on_py_register([fullname])
            unit.oncflags(['-Dinit{}=init{}'.format(shortname, mangle(fullname))])
        else:
            unit.on_py_register([name])


def onpy_main(unit, arg):
    if ':' not in arg:
        arg += ':main'

    py_program(unit)
    unit.onresource(['-', 'PY_MAIN={}'.format(arg)])
