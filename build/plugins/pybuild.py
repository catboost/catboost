import os
import ymake
from _common import stripext, rootrel_arc_src, listid, resolve_to_ymake_path, generate_chunks
from pyx import PyxParser


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


def pb_cc_arg(path, unit):
    return '{}.pb.cc'.format(stripext(to_build_root(path, unit)))

def pb_grpc_arg(path, unit):
    return '{}.grpc.pb.cc'.format(stripext(to_build_root(path, unit)))

def ev_cc_arg(path, unit):
    return '{}.ev.pb.cc'.format(stripext(to_build_root(path, unit)))


def pb2_grpc_arg(path, mod, unit):
    return '{}_pb2_grpc.py={}_pb2_grpc'.format(stripext(to_build_root(path, unit)), mod)


def ev_arg(path, mod, unit):
    return '{}_ev_pb2.py={}_ev_pb2'.format(stripext(to_build_root(path, unit)), mod)


def mangle(name):
    if '.' not in name:
        return name
    return ''.join('{}{}'.format(len(s), s) for s in name.split('.'))


def parse_pyx_includes(filename, path, source_root, seen=None):
    normpath = lambda *x: os.path.normpath(os.path.join(*x))

    abs_path = normpath(source_root, filename)
    seen = seen or set()
    if abs_path in seen:
        return
    seen.add(abs_path)

    if not os.path.exists(abs_path):
        # File might be missing, because it might be generated
        return

    with open(abs_path, 'rb') as f:
        # Don't parse cimports and etc - irrelevant for cython, it's linker work
        includes, _, _ = PyxParser.parse_includes(f.readlines(), perm_includes=False, direct_includes_only=True)

    abs_dirname = os.path.dirname(abs_path)
    # All includes are relative to the file which include
    path_dirname = os.path.dirname(path)
    file_dirname = os.path.dirname(filename)

    for incfile in includes:
        abs_path = normpath(abs_dirname, incfile)
        if os.path.exists(abs_path):
            incname, incpath = normpath(file_dirname, incfile), normpath(path_dirname, incfile)
            yield (incname, incpath)
            # search for includes in the included files
            for e in parse_pyx_includes(incname, incpath, source_root, seen):
                yield e
        else:
            # There might be arcadia root relative include.
            # Don't treat such file as missing, because there must be PEERDIR on py_library
            # which contains it.
            if not os.path.exists(normpath(source_root, incfile)):
                ymake.report_configure_error("'{}' includes missing file: {} ({})".format(path, incfile, abs_path))

def has_pyx(args):
    return any(arg.endswith('.pyx') for arg in args)

def get_srcdir(path, unit):
    return rootrel_arc_src(path, unit)[:-len(path)].rstrip('/')


def is_generated(path, unit):
    return not unit.resolve(path).startswith('$S/')


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
    """
    Documentation: https://wiki.yandex-team.ru/devtools/commandsandvars/py_srcs/#modulpyprogramimakrospymain
    """
    unit.onpeerdir(['library/python/runtime/main'])
    unit.onadd_check_py_imports()


def py3_program(unit):
    unit.onpeerdir(['library/python/runtime_py3/main'])
    #unit.onadd_check_py_imports()


def onpy_srcs(unit, *args):
    """
        PY_SRCS() - is rule to build extended versions of Python interpreters and containing all application code in its executable file. It can be used to collect only the executables but not shared libraries, and, in particular, not to collect the modules that are imported using import directive.
        The main disadvantage is the lack of IDE support; There is also no readline yet.
        The application can be collect from any of the sources from which the C library, and with the help of PY_SRCS .py , .pyx,.proto and .swg files.
        At the same time extensions for Python on C language generating from .pyx and .swg, will be registered in Python's as built-in modules, and sources on .py are stored as static data: when the interpreter starts, the initialization code will add a custom loader of these modules to sys.meta_path.
        By default .pyx files are collected as C++-extensions. To collect them as C (similar to BUILDWITH_CYTHON_C, but with the ability to specify namespace), you must specify the Directive CYTHON_C.
        Building with pyx automatically registers modules, you do not need to call PY_REGISTER for them
        __init__.py never required, but if present (and specified in PY_SRCS), it will be imported when you import package modules with __init__.py Oh.

        Example of library declaration with PY_SRCS():
        PY_LIBRARY(mymodule)
        PY_SRCS({| CYTHON_C} { | TOP_LEVEL | NAMESPACE ns} a.py sub/dir/b.py e.proto sub/dir/f.proto c.pyx sub/dir/d.pyx g.swg sub/dir/h.swg)
        END()

        Documentation: https://wiki.yandex-team.ru/devtools/commandsandvars/py_srcs/
    """
    # Each file arg must either be a path, or "${...}/buildpath=modname", where
    # "${...}/buildpath" part will be used as a file source in a future macro,
    # and "modname" will be used as a module name.

    if '/contrib/tools/python/src/Lib' not in unit.path():
        unit.onpeerdir(['contrib/libs/python'])

    if '/library/python/runtime' not in unit.path():
        unit.onpeerdir(['library/python/runtime'])

    is_program = unit.get('MODULE_TYPE') == 'PROGRAM'
    if is_program:
        py_program(unit)

    py_namespace_value = unit.get('PY_NAMESPACE_VALUE')
    if py_namespace_value == ".":
        ns = ""
    else:
        ns = (unit.get('PY_NAMESPACE_VALUE') or unit.path()[3:].replace('/', '.')) + '.'

    cython_coverage = unit.get('CYTHON_COVERAGE') == 'yes'
    optimize_proto = unit.get('OPTIMIZE_PY_PROTOS_FLAG') == 'yes'

    cython_includes = []
    for path in unit.includes():
        cython_includes += ['-I', resolve_to_ymake_path(path)]

    cython_directives = []
    if cython_coverage:
        cython_directives += ['-X', 'linetrace=True']

    pyxs_c = []
    pyxs_cpp = []
    pyxs = pyxs_cpp
    pys = []
    protos = []
    evs = []
    swigs = []

    dump_dir = unit.get('PYTHON_BUILD_DUMP_DIR')
    dump_output = None
    if dump_dir:
        import thread
        pid = os.getpid()
        tid = thread.get_ident()
        dump_name = '{}-{}.dump'.format(pid, tid)
        dump_output = open(os.path.join(dump_dir, dump_name), 'a')

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
            main_mod = arg == 'MAIN'
            if main_mod:
                arg = next(args)

            if '=' in arg:
                path, mod = arg.split('=', 1)
            else:
                path = arg
                if arg == '__main__.py' or arg.endswith('/__main__.py'):
                    mod = '__main__'
                else:
                    if arg.startswith('../'):
                        ymake.report_configure_error('PY_SRCS item starts with "../": {!r}'.format(arg))
                    mod = ns + stripext(arg).replace('/', '.')

            if main_mod:
                unit.onpy_main(mod)

            pathmod = (path, mod)

            if dump_output is not None:
                dump_output.write('{path}\t{module}\n'.format(path=rootrel_arc_src(path, unit), module=mod))

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

    if dump_output is not None:
        dump_output.close()

    if pyxs:
        files2res = set()

        if cython_coverage:
            def process_pyx(filename, path, out_suffix):
                # skip generated files
                if not is_arc_src(path, unit):
                    return
                # source file
                files2res.add((filename, path))
                # generated
                files2res.add((filename + out_suffix, path + out_suffix))
                # used includes
                for entry in parse_pyx_includes(filename, path, unit.resolve('$S')):
                    files2res.add(entry)
        else:
            def process_pyx(filename, path, out_suffix):
                pass

        for pyxs, cython, out_suffix in [
            (pyxs_c, unit.onbuildwith_cython_c, ".c"),
            (pyxs_cpp, unit.onbuildwith_cython_cpp, ".cpp"),
        ]:
            for path, mod in pyxs:
                filename = rootrel_arc_src(path, unit)
                cython([
                    path,
                    '--module-name', mod,
                    '--init-suffix', mangle(mod),
                    '--source-root', '${ARCADIA_ROOT}',
                    # set arcadia root relative __file__ for generated modules
                    '-X', 'set_initial_path={}'.format(filename),
                ] + cython_includes + cython_directives)
                unit.onpy_register([mod])
                process_pyx(filename, path, out_suffix)

        if files2res:
            # Compile original and generated sources into target for proper cython coverage calculation
            unit.onresource_files([x for name, path in files2res for x in ('DEST', name, path)])

    if pys:
        pys_seen = set()
        pys_dups = {m for _, m in pys if (m in pys_seen or pys_seen.add(m))}
        if pys_dups:
            ymake.report_configure_error('Duplicate(s) is found in the PY_SRCS macro: {}'.format(pys_dups))

        res = []

        for path, mod in pys:
            root_rel_path = rootrel_arc_src(path, unit)
            unit.onpy_compile_bytecode([root_rel_path + '-', path])
            key = '/py_modules/' + mod
            res += [
                path, key,
                '-', 'resfs/src/{}={}'.format(key, root_rel_path),
                path + '.yapyc', '/py_code/' + mod,
            ]

        unit.onresource(res)
        add_python_lint_checks(unit, [path for path, mod in pys])

    if protos:
        if '/contrib/libs/protobuf/python/google_lib' not in unit.path():
            unit.onpeerdir(['contrib/libs/protobuf/python/google_lib'])

        grpc = unit.get('GRPC_FLAG') == 'yes'

        if grpc:
            unit.onpeerdir(['contrib/libs/grpc/python', 'contrib/libs/grpc'])

        proto_paths = [path for path, mod in protos]
        unit.ongenerate_py_protos(proto_paths)
        unit.onpy_srcs([pb2_arg(path, mod, unit) for path, mod in protos])

        if grpc:
            unit.onpy_srcs([pb2_grpc_arg(path, mod, unit) for path, mod in protos])

        if optimize_proto:
            unit.onsrcs(proto_paths)

            pb_cc_outs = [pb_cc_arg(path, unit) for path in proto_paths]
            if grpc:
                pb_cc_outs += [pb_grpc_arg(path, unit) for path in proto_paths]
            for pb_cc_outs_chunk in generate_chunks(pb_cc_outs, 10):
                if is_program:
                    unit.onjoin_srcs(['join_' + listid(pb_cc_outs_chunk) + '.cpp'] + pb_cc_outs_chunk)
                else:
                    unit.onjoin_srcs_global(['join_' + listid(pb_cc_outs_chunk) + '.cpp'] + pb_cc_outs_chunk)


    if evs:
        if '/contrib/libs/protobuf/python/google_lib' not in unit.path():
            unit.onpeerdir(['contrib/libs/protobuf/python/google_lib'])

        unit.ongenerate_py_evs([path for path, mod in evs])
        unit.onpy_srcs([ev_arg(path, mod, unit) for path, mod in evs])

        if optimize_proto:
            unit.onsrcs([path for path, mod in evs])

            pb_cc_outs = [ev_cc_arg(path, unit) for path, _ in evs]
            for pb_cc_outs_chunk in generate_chunks(pb_cc_outs, 10):
                if is_program:
                    unit.onjoin_srcs(['join_' + listid(pb_cc_outs_chunk) + '.cpp'] + pb_cc_outs_chunk)
                else:
                    unit.onjoin_srcs_global(['join_' + listid(pb_cc_outs_chunk) + '.cpp'] + pb_cc_outs_chunk)

    if swigs:
        unit.onsrcs(swigs)
        prefix = unit.get('MODULE_PREFIX')
        project = unit.get('REALPRJNAME')
        unit.onpy_register([prefix + project])
        path = '${ARCADIA_BUILD_ROOT}/' + '{}/{}.py'.format(unit.path()[3:], project)
        arg = '{}={}'.format(path, ns + project.replace('/', '.'))
        unit.onpy_srcs([arg])


def onpy3_srcs(unit, *args):
    # Each file arg must either be a path, or "${...}/buildpath=modname", where
    # "${...}/buildpath" part will be used as a file source in a future macro,
    # and "modname" will be used as a module name.

    if '/contrib/tools/python3/src/Lib' not in unit.path():
        unit.onpeerdir(['contrib/libs/python'])

        if '/library/python/runtime_py3' not in unit.path():
            unit.onpeerdir(['library/python/runtime_py3'])

    is_program = unit.get('MODULE_TYPE') == 'PROGRAM'
    if is_program:
        py3_program(unit)

    py_namespace_value = unit.get('PY_NAMESPACE_VALUE')
    if py_namespace_value == ".":
        ns = ""
    else:
        ns = (unit.get('PY_NAMESPACE_VALUE') or unit.path()[3:].replace('/', '.')) + '.'

    optimize_proto = unit.get('OPTIMIZE_PY_PROTOS_FLAG') == 'yes'
    cython_coverage = unit.get('CYTHON_COVERAGE') == 'yes'

    cython_directives = []
    if cython_coverage:
        cython_directives += ['-X', 'linetrace=True']

    pyxs_c = []
    pyxs_cpp = []
    pyxs = pyxs_cpp
    pys = []
    protos = []
    evs = []

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
            main_mod = arg == 'MAIN'
            if main_mod:
                arg = next(args)

            if '=' in arg:
                path, mod = arg.split('=', 1)
            else:
                path = arg
                if arg == '__main__.py' or arg.endswith('/__main__.py'):
                    mod = '__main__'
                else:
                    if arg.startswith('../'):
                        ymake.report_configure_error('PY3_SRCS item starts with "../": {!r}'.format(arg))
                    mod = ns + stripext(arg).replace('/', '.')

            if main_mod:
                unit.onpy3_main(mod)

            pathmod = (path, mod)

            if path.endswith('.py'):
                pys.append(pathmod)
            elif path.endswith('.pyx'):
                pyxs.append(pathmod)
            elif path.endswith('.proto'):
                protos.append(pathmod)
            elif path.endswith('.ev'):
                evs.append(pathmod)
            else:
                ymake.report_configure_error('in PY3_SRCS: unrecognized arg {!r}'.format(path))

    if pyxs:
        files2res = set()

        if cython_coverage:
            def process_pyx(filename, path, out_suffix):
                # skip generated files
                if not is_arc_src(path, unit):
                    return
                # source file
                files2res.add((filename, path))
                # generated
                files2res.add((filename + out_suffix, path + out_suffix))
                # used includes
                for entry in parse_pyx_includes(filename, path, unit.resolve('$S')):
                    files2res.add(entry)
        else:
            def process_pyx(filename, path, out_suffix):
                pass

        for pyxs, cython, out_suffix in [
            (pyxs_c, unit.onbuildwith_cython_c, ".c"),
            (pyxs_cpp, unit.onbuildwith_cython_cpp, ".cpp"),
        ]:
            for path, mod in pyxs:
                filename = rootrel_arc_src(path, unit)
                cython([
                    path,
                    '--module-name', mod,
                    '--init-suffix', mangle(mod),
                    '--source-root', '${ARCADIA_ROOT}',
                    # set arcadia root relative __file__ for generated modules
                    '-X', 'set_initial_path={}'.format(filename),
                ] + cython_directives)
                unit.onpy3_register([mod])
                process_pyx(filename, path, out_suffix)

        if files2res:
            # Compile original and generated sources into target for proper cython coverage calculation
            unit.onresource_files([x for name, path in files2res for x in ('DEST', name, path)])

    if pys:
        res = []

        for path, mod in pys:
            root_rel_path = rootrel_arc_src(path, unit)
            unit.onpy3_compile_bytecode([root_rel_path + '-', path])
            dest = 'py/' + mod.replace('.', '/') + '.py'
            res += [
                'DEST', dest, path,
                'DEST', dest + '.yapyc3', path + '.yapyc3'
            ]

        unit.onresource_files(res)
        #add_python_lint_checks(unit, [path for path, mod in pys])

    if protos:
        if '/contrib/libs/protobuf/python/google_lib' not in unit.path():
            unit.onpeerdir(['contrib/libs/protobuf/python/google_lib'])

        grpc = unit.get('GRPC_FLAG') == 'yes'

        if grpc:
            unit.onpeerdir(['contrib/libs/grpc/python', 'contrib/libs/grpc'])

        proto_paths = [path for path, mod in protos]
        unit.ongenerate_py_protos(proto_paths)
        unit.onpy3_srcs([pb2_arg(path, mod, unit) for path, mod in protos])

        if grpc:
            unit.onpy3_srcs([pb2_grpc_arg(path, mod, unit) for path, mod in protos])

        if optimize_proto:
            unit.onsrcs(proto_paths)

            pb_cc_outs = [pb_cc_arg(path, unit) for path in proto_paths]
            if grpc:
                pb_cc_outs += [pb_grpc_arg(path, unit) for path in proto_paths]
            for pb_cc_outs_chunk in generate_chunks(pb_cc_outs, 10):
                if is_program:
                    unit.onjoin_srcs(['join_py3' + listid(pb_cc_outs_chunk) + '.cpp'] + pb_cc_outs_chunk)
                else:
                    unit.onjoin_srcs_global(['join_py3' + listid(pb_cc_outs_chunk) + '.cpp'] + pb_cc_outs_chunk)

    if evs:
        if '/contrib/libs/protobuf/python/google_lib' not in unit.path():
            unit.onpeerdir(['contrib/libs/protobuf/python/google_lib'])

        unit.ongenerate_py_evs([path for path, mod in evs])
        unit.onpy3_srcs([ev_arg(path, mod, unit) for path, mod in evs])

        if optimize_proto:
            unit.onsrcs([path for path, mod in evs])

            pb_cc_outs = [ev_cc_arg(path, unit) for path, _ in evs]
            for pb_cc_outs_chunk in generate_chunks(pb_cc_outs, 10):
                if is_program:
                    unit.onjoin_srcs(['join_py3' + listid(pb_cc_outs_chunk) + '.cpp'] + pb_cc_outs_chunk)
                else:
                    unit.onjoin_srcs_global(['join_py3' + listid(pb_cc_outs_chunk) + '.cpp'] + pb_cc_outs_chunk)


def _check_test_srcs(*args):
    used = set(args) & {"NAMESPACE", "TOP_LEVEL", "__main__.py"}
    if used:
        param = list(used)[0]
        ymake.report_configure_error('in TEST_SRCS: you cannot use {} here - it would broke testing machinery'.format(param))


def ontest_srcs(unit, *args):
    _check_test_srcs(*args)
    if unit.get('PYTEST_BIN') != 'no':
        unit.onpy_srcs(["NAMESPACE", "__tests__"] + list(args))


def on_test3_srcs(unit, *args):
    _check_test_srcs(*args)
    if unit.get('PY3TEST_BIN') != 'no':
        unit.onpy3_srcs(["NAMESPACE", "__tests__"] + list(args))


def onpy_register(unit, *args):
    """
    Python knows about which built-ins can be imported, due to their registration in the Assembly or at the start of the interpreter.

    All modules from the sources listed in PY_SRCS() are registered automatically.
    To register the modules from the sources in the SRCS(), you need to use PY_REGISTER().
    """
    for name in args:
        if '=' in name:
            fullname, shortname = name.split('=', 1)
            assert '.' not in shortname, shortname
            assert fullname == shortname or fullname.endswith('.' + shortname), fullname
            unit.on_py_register([fullname])
            unit.oncflags(['-Dinit{}=init{}'.format(shortname, mangle(fullname))])
        else:
            unit.on_py_register([name])


def onpy3_register(unit, *args):
    for name in args:
        if '=' in name:
            fullname, shortname = name.split('=', 1)
            assert '.' not in shortname, shortname
            assert fullname == shortname or fullname.endswith('.' + shortname), fullname
            unit.on_py3_register([fullname])
            unit.oncflags(['-DPyInit_{}=PyInit_{}'.format(shortname, mangle(fullname))])
        else:
            unit.on_py3_register([name])


def onpy_main(unit, arg):
    """
        @usage: PY_MAIN(pkg.mod[:func])

        Specifies the function from which to start executing a python program

        Documentation: https://wiki.yandex-team.ru/devtools/commandsandvars/py_srcs/
    """
    if ':' not in arg:
        arg += ':main'

    py_program(unit)
    unit.onresource(['-', 'PY_MAIN={}'.format(arg)])

def onpy3_main(unit, arg):
    if ':' not in arg:
        arg += ':main'

    py3_program(unit)
    unit.onresource(['-', 'PY_MAIN={}'.format(arg)])
