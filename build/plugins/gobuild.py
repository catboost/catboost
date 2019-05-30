import os
from _common import rootrel_arc_src
import ymake


runtime_cgo_path = os.path.join('runtime', 'cgo')
runtime_msan_path = os.path.join('runtime', 'msan')
runtime_race_path = os.path.join('runtime', 'race')


def get_appended_values(unit, key):
    value = []
    raw_value = unit.get(key)
    if raw_value:
        value = filter(lambda x: len(x) > 0, raw_value.split(' '))
        assert len(value) == 0 or value[0] == '$' + key
    return value[1:] if len(value) > 0 else value


def compare_versions(version1, version2):
    v1 = tuple(str(int(x)).zfill(8) for x in version1.split('.'))
    v2 = tuple(str(int(x)).zfill(8) for x in version2.split('.'))
    if v1 == v2:
        return 0
    return 1 if v1 < v2 else -1


def go_package_name(unit):
    # TODO(prime@): this seems wrong. This code detects wrong package name when used from GO_TEST module.
    return unit.get('GO_PACKAGE_VALUE') or unit.get('MODULE_TYPE') == 'PROGRAM' and 'main' or unit.get('REALPRJNAME')


def on_go_process_srcs(unit):
    """
        _GO_PROCESS_SRCS() macro processes only 'CGO' files. All remaining *.go files
        and other input files are currently processed by a link command of the
        GO module (GO_LIBRARY, GO_PROGRAM)
    """

    go_files = get_appended_values(unit, 'GO_SRCS_VALUE')
    for f in go_files:
        if f.endswith('_test.go'):
            ymake.report_configure_error('file {} must be listed in GO_TEST_SRCS() or GO_XTEST_SRCS() macros'.format(f))
    go_test_files = get_appended_values(unit, 'GO_TEST_SRCS_VALUE')
    go_xtest_files = get_appended_values(unit, 'GO_XTEST_SRCS_VALUE')
    for f in go_test_files + go_xtest_files:
        if not f.endswith('_test.go'):
            ymake.report_configure_error('file {} should not be listed in GO_TEST_SRCS() or GO_XTEST_SRCS() macros'.format(f))

    resolved_go_files = []
    for path in go_files + go_test_files + go_xtest_files:
        if path.endswith(".go"):
            resolved = unit.resolve_arc_path([path])
            if resolved != path and not resolved.startswith("$S/vendor") and not resolved.startswith("$S/contrib"):
                resolved_go_files.append(resolved)
    if resolved_go_files:
        basedirs = {}
        for f in resolved_go_files:
            basedir = os.path.dirname(f)
            if basedir not in basedirs:
                basedirs[basedir] = []
            basedirs[basedir].append(f)
        for basedir in basedirs:
            unit.onadd_check(["gofmt"] + basedirs[basedir])

    go_std_root = unit.get('GOSTD') + os.path.sep

    proto_files = filter(lambda x: x.endswith('.proto'), go_files)
    if len(proto_files) > 0:
        for f in proto_files:
            unit.on_go_proto_cmd(f)

    in_files = filter(lambda x: x.endswith('.in'), go_files)
    if len(in_files) > 0:
        for f in in_files:
            unit.onsrc(f)

    if compare_versions('1.12', unit.get('GOSTD_VERSION')) >= 0:
        asm_files = filter(lambda x: x.endswith('.s'), go_files)
        if len(asm_files) > 0:
            unit.on_go_compile_symabis(asm_files)

    s_files = filter(lambda x: x.endswith('.S'), go_files)
    c_files = filter(lambda x: x.endswith('.c'), go_files)
    if len(c_files) + len(s_files) > 0:
        cgo_flags = get_appended_values(unit, 'CGO_CFLAGS_VALUE')
        for f in c_files + s_files:
            unit.onsrc([f] + cgo_flags)

    cgo_files = get_appended_values(unit, 'CGO_SRCS_VALUE')
    if len(cgo_files) > 0:
        import_path = rootrel_arc_src(unit.path(), unit)
        if import_path.startswith(go_std_root):
            import_path = import_path[len(go_std_root):]
        if import_path != runtime_cgo_path:
            unit.onpeerdir(os.path.join(go_std_root, runtime_cgo_path))
        import_runtime_cgo = 'false' if import_path in [runtime_cgo_path, runtime_msan_path, runtime_race_path] else 'true'
        import_syscall = 'false' if import_path == runtime_cgo_path else 'true'
        args = [import_path] + cgo_files + ['FLAGS', '-import_runtime_cgo=' + import_runtime_cgo, '-import_syscall=' + import_syscall]
        unit.on_go_compile_cgo1(args)
        args = [go_package_name(unit)] + cgo_files
        if len(c_files) > 0:
            args += ['C_FILES'] + c_files
        if len(s_files) > 0:
            args += ['S_FILES'] + s_files
        unit.on_go_compile_cgo2(args)


def on_go_resource(unit, *args):
    args = list(args)
    files = args[::2]
    keys = args[1::2]
    resource_go = os.path.join("resource.res.go")

    unit.onpeerdir(["library/go/core/resource"])

    if len(files) != len(keys):
        ymake.report_configure_error("last file {} is missing resource key".format(files[-1]))

    for i, (key, filename) in enumerate(zip(keys, files)):
        if not key:
            ymake.report_configure_error("file key must be non empty")
            return

        if filename == "-" and "=" not in key:
            ymake.report_configure_error("key \"{}\" must contain = sign".format(key))
            return

        # quote key, to avoid automatic substitution of filename by absolute
        # path in RUN_PROGRAM
        args[2*i+1] = "notafile" + args[2*i+1]

    files = [file for file in files if file != "-"]
    unit.onrun_program([
        "library/go/core/resource/cc",
        "-package", go_package_name(unit),
        "-o", resource_go] + list(args) + [
        "IN"] + files + [
        "OUT", resource_go])
