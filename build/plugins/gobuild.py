import os
from _common import rootrel_arc_src


go_root = os.path.join('contrib', 'go', 'src') + os.path.sep
runtime_cgo_path = os.path.join('runtime', 'cgo')
runtime_msan_path = os.path.join('runtime', 'msan')
runtime_race_path = os.path.join('runtime', 'race')
cxxsupp_path = os.path.join('contrib', 'libs', 'cxxsupp')


def get_appended_values(unit, key):
    value = []
    raw_value = unit.get(key)
    if raw_value:
        value = raw_value.split(' ')
        assert len(value) == 0 or value[0] == '$' + key
    return value[1:] if len(value) > 0 else value


def on_go_process_srcs(unit):
    """
        _GO_PROCESS_SRCS() macro processes only 'CGO' files. All remaining *.go files
        and other input files are currently processed by a link command of the
        GO module (GO_LIBRARY, GO_PROGRAM)
    """

    go_files = get_appended_values(unit, 'GO_FILES_VALUE')
    if len(go_files) > 0:
        files = filter(lambda x: x.endswith('.c') or x.endswith('.S'), go_files)
        if len(files) > 0:
            cgo_flags = get_appended_values(unit, 'CGO_CFLAGS_VALUE')
            for f in files:
                flags = cgo_flags if f.endswith('c') else []
                unit.onsrc([f] + flags)

    cgo_files = get_appended_values(unit, 'CGO_FILES_VALUE')
    if len(cgo_files) > 0:
        unit.onpeerdir(cxxsupp_path)
        import_path = rootrel_arc_src(unit.path(), unit)
        if import_path.startswith(go_root):
            import_path = import_path[len(go_root):]
        if import_path != runtime_cgo_path:
            unit.onpeerdir(os.path.join(go_root, runtime_cgo_path))
        import_runtime_cgo = 'false' if import_path in [runtime_cgo_path, runtime_msan_path, runtime_race_path] else 'true'
        import_syscall = 'false' if import_path == runtime_cgo_path else 'true'
        unit.ongo_compile_cgo1([import_path] + cgo_files + ['FLAGS', '-import_runtime_cgo=' + import_runtime_cgo, '-import_syscall=' + import_syscall])
        unit.ongo_compile_cgo2([os.path.basename(import_path)] + cgo_files)
