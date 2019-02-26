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

    go_std_root = unit.get('GOSTD') + os.path.sep

    proto_files = filter(lambda x: x.endswith('.proto'), go_files)
    if len(proto_files) > 0:
        for f in proto_files:
            unit.ongo_proto_cmd(f)

    in_files = filter(lambda x: x.endswith('.in'), go_files)
    if len(in_files) > 0:
        for f in in_files:
            unit.onsrc(f)

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
        unit.ongo_compile_cgo1(args)
        args = [unit.get('GO_PACKAGE_VALUE') or unit.get('REALPRJNAME')] + cgo_files
        if len(c_files) > 0:
            args += ['C_FILES'] + c_files
        if len(s_files) > 0:
            args += ['S_FILES'] + s_files
        unit.ongo_compile_cgo2(args)
