import base64
import itertools
import md5
import os
from _common import rootrel_arc_src, tobuilddir
import ymake


runtime_cgo_path = os.path.join('runtime', 'cgo')
runtime_msan_path = os.path.join('runtime', 'msan')
runtime_race_path = os.path.join('runtime', 'race')
arc_project_prefix = 'a.yandex-team.ru/'
import_runtime_cgo_false = {
    'norace': (runtime_cgo_path, runtime_msan_path, runtime_race_path),
    'race': (runtime_cgo_path, runtime_msan_path),
}
import_syscall_false = {
    'norace': (runtime_cgo_path),
    'race': (runtime_cgo_path, runtime_race_path),
}


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
    name = unit.get('GO_PACKAGE_VALUE')
    if not name:
        name = unit.get('GO_TEST_IMPORT_PATH')
        if name:
            name = os.path.basename(os.path.normpath(name))
        elif unit.get('MODULE_TYPE') == 'PROGRAM':
            name = 'main'
        else:
            name = unit.get('REALPRJNAME')
    return name


def need_lint(path):
    return not path.startswith('$S/vendor/') and not path.startswith('$S/contrib/')


def on_go_process_srcs(unit):
    """
        _GO_PROCESS_SRCS() macro processes only 'CGO' files. All remaining *.go files
        and other input files are currently processed by a link command of the
        GO module (GO_LIBRARY, GO_PROGRAM)
    """

    srcs_files = get_appended_values(unit, 'GO_SRCS_VALUE')

    asm_files = []
    c_files = []
    cxx_files = []
    go_files = []
    in_files = []
    proto_files = []
    s_files = []
    syso_files = []

    classifed_files = {
        '.c': c_files,
        '.cc': cxx_files,
        '.cpp': cxx_files,
        '.cxx': cxx_files,
        '.go': go_files,
        '.in': in_files,
        '.proto': proto_files,
        '.s': asm_files,
        '.syso': syso_files,
        '.C': cxx_files,
        '.S': s_files,
    }

    # Classify files specifed in _GO_SRCS() macro by extension and process CGO_EXPORT keyword
    # which can preceed C/C++ files only
    is_cgo_export = False
    for f in srcs_files:
        _, ext = os.path.splitext(f)
        ext_files = classifed_files.get(ext)
        if ext_files is not None:
            if is_cgo_export:
                is_cgo_export = False
                if ext in ('.c', '.cc', '.cpp', '.cxx', '.C'):
                    unit.oncopy_file_with_context([f, f, 'OUTPUT_INCLUDES', '${BINDIR}/_cgo_export.h'])
                    f = '${BINDIR}/' + f
                else:
                    ymake.report_configure_error('Unmatched CGO_EXPORT keyword in SRCS()/_GO_SRCS() macro')
            ext_files.append(f)
        elif f == 'CGO_EXPORT':
            is_cgo_export = True
        else:
            # FIXME(snermolaev): We can report an unsupported files for _GO_SRCS here
            pass
    if is_cgo_export:
        ymake.report_configure_error('Unmatched CGO_EXPORT keyword in SRCS()/_GO_SRCS() macro')

    for f in go_files:
        if f.endswith('_test.go'):
            ymake.report_configure_error('file {} must be listed in GO_TEST_SRCS() or GO_XTEST_SRCS() macros'.format(f))
    go_test_files = get_appended_values(unit, 'GO_TEST_SRCS_VALUE')
    go_xtest_files = get_appended_values(unit, 'GO_XTEST_SRCS_VALUE')
    for f in go_test_files + go_xtest_files:
        if not f.endswith('_test.go'):
            ymake.report_configure_error('file {} should not be listed in GO_TEST_SRCS() or GO_XTEST_SRCS() macros'.format(f))

    is_test_module = unit.enabled('GO_TEST_MODULE')

    # Add gofmt style checks
    if unit.enabled('_GO_FMT_ADD_CHECK'):
        resolved_go_files = []
        go_source_files = [] if is_test_module and unit.get(['GO_TEST_FOR_DIR']) else go_files
        for path in itertools.chain(go_source_files, go_test_files, go_xtest_files):
            if path.endswith('.go'):
                resolved = unit.resolve_arc_path([path])
                if resolved != path and need_lint(resolved):
                    resolved_go_files.append(resolved)
        if resolved_go_files:
            basedirs = {}
            for f in resolved_go_files:
                basedir = os.path.dirname(f)
                if basedir not in basedirs:
                    basedirs[basedir] = []
                basedirs[basedir].append(f)
            for basedir in basedirs:
                unit.onadd_check(['gofmt'] + basedirs[basedir])

    # Go coverage instrumentation (NOTE! go_files list is modified here)
    if is_test_module and unit.enabled('GO_TEST_COVER'):
        cover_info = []

        for f in go_files:
            if f.endswith('_test.go'):
                continue
            cover_var = 'GoCover' + base64.b32encode(f).rstrip('=')
            cover_file = unit.resolve_arc_path(f)
            unit.on_go_gen_cover_go([cover_file, cover_var])
            if cover_file.startswith('$S/'):
                cover_file = arc_project_prefix + cover_file[3:]
            cover_info.append('{}:{}'.format(cover_var, cover_file))

        # go_files should be empty now since the initial list shouldn't contain
        # any non-go or go test file. The value of go_files list will be used later
        # to update the value of GO_SRCS_VALUE
        go_files = []
        unit.set(['GO_COVER_INFO_VALUE', ' '.join(cover_info)])

    # We have cleaned up the list of files from GO_SRCS_VALUE var and we have to update
    # the value since it is used in module command line
    unit.set(['GO_SRCS_VALUE', ' '.join(itertools.chain(go_files, asm_files, syso_files))])

    unit_path = unit.path()

    # Add go vet check
    if unit.enabled('_GO_VET_ADD_CHECK') and need_lint(unit_path):
        vet_report_file_name = os.path.join(unit_path, '{}{}'.format(unit.filename(), unit.get('GO_VET_REPORT_EXT')))
        unit.onadd_check(["govet", '$(BUILD_ROOT)/' + tobuilddir(vet_report_file_name)[3:]])

    # Process .proto files
    for f in proto_files:
        unit.on_go_proto_cmd(f)

    # Process .in files
    for f in in_files:
        unit.onsrc(f)

    # Generate .symabis for .s files (starting from 1.12 version)
    if compare_versions('1.12', unit.get('GOSTD_VERSION')) >= 0 and len(asm_files) > 0:
        unit.on_go_compile_symabis(asm_files)

    # Process cgo files
    cgo_files = get_appended_values(unit, 'CGO_SRCS_VALUE')

    cgo_cflags = []
    if len(c_files) + len(cxx_files) + len(s_files) + len(cgo_files) > 0:
        if is_test_module:
            go_test_for_dir = unit.get('GO_TEST_FOR_DIR')
            if go_test_for_dir and go_test_for_dir.startswith('$S/'):
                cgo_cflags.append(os.path.join('-I${ARCADIA_ROOT}', go_test_for_dir[3:]))
        cgo_cflags.append('-I$CURDIR')
        unit.oncgo_cflags(cgo_cflags)
        cgo_cflags = get_appended_values(unit, 'CGO_CFLAGS_VALUE')

    for f in itertools.chain(c_files, cxx_files, s_files):
        unit.onsrc([f] + cgo_cflags)

    if len(cgo_files) > 0:
        if not unit.enabled('CGO_ENABLED'):
            ymake.report_configure_error('trying to build with CGO (CGO_SRCS is non-empty) when CGO is disabled')
        import_path = rootrel_arc_src(unit_path, unit)
        go_std_root = unit.get('GOSTD') + os.path.sep
        if import_path.startswith(go_std_root):
            import_path = import_path[len(go_std_root):]
        if import_path != runtime_cgo_path:
            unit.onpeerdir(os.path.join(go_std_root, runtime_cgo_path))
        race_mode = 'race' if unit.enabled('RACE') else 'norace'
        import_runtime_cgo = 'false' if import_path in import_runtime_cgo_false[race_mode] else 'true'
        import_syscall = 'false' if import_path in import_syscall_false[race_mode] else 'true'
        args = [import_path] + cgo_files + ['FLAGS', '-import_runtime_cgo=' + import_runtime_cgo, '-import_syscall=' + import_syscall]
        unit.on_go_compile_cgo1(args)
        cgo2_cflags = get_appended_values(unit, 'CGO2_CFLAGS_VALUE')
        for f in cgo_files:
            if f.endswith('.go'):
                unit.onsrc([f[:-2] + 'cgo2.c'] + cgo_cflags + cgo2_cflags)
            else:
                ymake.report_configure_error('file {} should not be listed in CGO_SRCS() macros'.format(f))
        args = [go_package_name(unit)] + cgo_files
        if len(c_files) > 0:
            args += ['C_FILES'] + c_files
        if len(s_files) > 0:
            args += ['S_FILES'] + s_files
        if len(syso_files) > 0:
            args += ['OBJ_FILES'] + syso_files
        unit.on_go_compile_cgo2(args)


def on_go_resource(unit, *args):
    args = list(args)
    files = args[::2]
    keys = args[1::2]
    suffix_md5 = md5.new('@'.join(args)).hexdigest()
    resource_go = os.path.join("resource.{}.res.go".format(suffix_md5))

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
