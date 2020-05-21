import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading

arc_project_prefix = 'a.yandex-team.ru/'
std_lib_prefix = 'contrib/go/_std/src/'
vendor_prefix = 'vendor/'
vet_info_ext = '.vet.out'
vet_report_ext = '.vet.txt'

FIXED_CGO1_SUFFIX='.fixed.cgo1.go'

COMPILE_OPTIMIZATION_FLAGS=('-N',)


def get_trimpath_args(args):
    return ['-trimpath', args.trimpath] if args.trimpath else []


def preprocess_cgo1(src_path, dst_path, source_root):
    with open(src_path, 'r') as f:
        content = f.read()
        content = content.replace('__ARCADIA_SOURCE_ROOT_PREFIX__', source_root)
    with open(dst_path, 'w') as f:
        f.write(content)


def preprocess_args(args):
    # Temporary work around for noauto
    if args.cgo_srcs and len(args.cgo_srcs) > 0:
        cgo_srcs_set = set(args.cgo_srcs)
        args.srcs = list(filter(lambda x: x not in cgo_srcs_set, args.srcs))

    args.pkg_root = os.path.join(args.toolchain_root, 'pkg')
    toolchain_tool_root = os.path.join(args.pkg_root, 'tool', '{}_{}'.format(args.host_os, args.host_arch))
    args.go_compile = os.path.join(toolchain_tool_root, 'compile')
    args.go_cgo = os.path.join(toolchain_tool_root, 'cgo')
    args.go_link = os.path.join(toolchain_tool_root, 'link')
    args.go_asm = os.path.join(toolchain_tool_root, 'asm')
    args.go_pack = os.path.join(toolchain_tool_root, 'pack')
    args.go_vet = os.path.join(toolchain_tool_root, 'vet') if args.vet is True else args.vet
    args.output = os.path.normpath(args.output)
    args.vet_report_output = vet_report_output_name(args.output, args.vet_report_ext)
    args.trimpath = None
    if args.debug_root_map:
        roots = {'build': args.build_root, 'source': args.source_root, 'tools': args.tools_root}
        replaces = []
        for root in args.debug_root_map.split(';'):
            src, dst = root.split('=', 1)
            assert src in roots
            replaces.append('{}=>{}'.format(roots[src], dst))
            del roots[src]
        assert len(replaces) > 0
        args.trimpath = ';'.join(replaces)
    args.build_root = os.path.normpath(args.build_root)
    args.build_root_dir = args.build_root + os.path.sep
    args.source_root = os.path.normpath(args.source_root)
    args.source_root_dir = args.source_root + os.path.sep
    args.output_root = os.path.normpath(args.output_root)
    args.import_map = {}
    args.module_map = {}
    if args.cgo_peers:
        args.cgo_peers = [x for x in args.cgo_peers if not x.endswith('.fake.pkg')]

    assert args.mode == 'test' or args.test_srcs is None and args.xtest_srcs is None
    # add lexical oreder by basename for go sources
    args.srcs.sort(key=lambda x: os.path.basename(x))
    if args.test_srcs:
        args.srcs += sorted(args.test_srcs, key=lambda x: os.path.basename(x))
        del args.test_srcs
    if args.xtest_srcs:
        args.xtest_srcs.sort(key=lambda x: os.path.basename(x))

    # compute root relative module dir path
    assert args.output is None or args.output_root == os.path.dirname(args.output)
    assert args.output_root.startswith(args.build_root_dir)
    args.module_path = args.output_root[len(args.build_root_dir):]
    assert len(args.module_path) > 0
    args.import_path, args.is_std = get_import_path(args.module_path)

    assert args.asmhdr is None or args.word == 'go'

    srcs = []
    for f in args.srcs:
        if f.endswith(FIXED_CGO1_SUFFIX) and f.startswith(args.build_root_dir):
            path = os.path.join(args.output_root, '{}.cgo1.go'.format(os.path.basename(f[:-len(FIXED_CGO1_SUFFIX)])))
            srcs.append(path)
            preprocess_cgo1(f, path, args.source_root)
        else:
            srcs.append(f)
    args.srcs = srcs

    classify_srcs(args.srcs, args)


def compare_versions(version1, version2):
    v1 = tuple(str(int(x)).zfill(8) for x in version1.split('.'))
    v2 = tuple(str(int(x)).zfill(8) for x in version2.split('.'))
    if v1 == v2:
        return 0
    return 1 if v1 < v2 else -1


def get_symlink_or_copyfile():
    os_symlink = getattr(os, 'symlink', None)
    if os_symlink is None:
        os_symlink = shutil.copyfile
    return os_symlink


def copy_args(args):
    return copy.copy(args)


def get_vendor_index(import_path):
    index = import_path.rfind('/' + vendor_prefix)
    if index < 0:
        index = 0 if import_path.startswith(vendor_prefix) else index
    else:
        index = index + 1
    return index


def get_import_path(module_path):
    assert len(module_path) > 0
    import_path = module_path.replace('\\', '/')
    is_std_module = import_path.startswith(std_lib_prefix)
    if is_std_module:
        import_path = import_path[len(std_lib_prefix):]
    elif import_path.startswith(vendor_prefix):
        import_path = import_path[len(vendor_prefix):]
    else:
        import_path = arc_project_prefix + import_path
    assert len(import_path) > 0
    return import_path, is_std_module


def call(cmd, cwd, env=None):
    # print >>sys.stderr, ' '.join(cmd)
    return subprocess.check_output(cmd, stdin=None, stderr=subprocess.STDOUT, cwd=cwd, env=env)


def classify_srcs(srcs, args):
    args.go_srcs = list(filter(lambda x: x.endswith('.go'), srcs))
    args.asm_srcs = list(filter(lambda x: x.endswith('.s'), srcs))
    args.objects = list(filter(lambda x: x.endswith('.o') or x.endswith('.obj'), srcs))
    args.symabis = list(filter(lambda x: x.endswith('.symabis'), srcs))
    args.sysos = list(filter(lambda x: x.endswith('.syso'), srcs))


def get_import_config_info(peers, gen_importmap, import_map={}, module_map={}):
    info = {'importmap': [], 'packagefile': [], 'standard': {}}
    if gen_importmap:
        for key, value in import_map.items():
            info['importmap'].append((key, value))
    for peer in peers:
        peer_import_path, is_std = get_import_path(os.path.dirname(peer))
        if gen_importmap:
            index = get_vendor_index(peer_import_path)
            if index >= 0:
                index += len(vendor_prefix)
                info['importmap'].append((peer_import_path[index:], peer_import_path))
        info['packagefile'].append((peer_import_path, os.path.join(args.build_root, peer)))
        if is_std:
            info['standard'][peer_import_path] = True
    for key, value in module_map.items():
        info['packagefile'].append((key, value))
    return info


def create_import_config(peers, gen_importmap, import_map={}, module_map={}):
    lines = []
    info = get_import_config_info(peers, gen_importmap, import_map, module_map)
    for key in ('importmap', 'packagefile'):
        for item in info[key]:
            lines.append('{} {}={}'.format(key, *item))
    if len(lines) > 0:
        lines.append('')
        content = '\n'.join(lines)
        # print >>sys.stderr, content
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            return f.name
    return None


def vet_info_output_name(path, ext=None):
    return '{}{}'.format(path, ext or vet_info_ext)


def vet_report_output_name(path, ext=None):
    return '{}{}'.format(path, ext or vet_report_ext)


def get_source_path(args):
    return args.test_import_path or args.module_path


def gen_vet_info(args):
    import_path = args.real_import_path if hasattr(args, 'real_import_path') else args.import_path
    info = get_import_config_info(args.peers, True, args.import_map, args.module_map)

    import_map = dict(info['importmap'])
    # FIXME(snermolaev): it seems that adding import map for 'fake' package
    #                    does't make any harm (it needs to be revised later)
    import_map['unsafe'] = 'unsafe'

    for (key, _) in info['packagefile']:
        if key not in import_map:
            import_map[key] = key

    data = {
        'ID': import_path,
        'Compiler': 'gc',
        'Dir': os.path.join(args.source_root, get_source_path(args)),
        'ImportPath': import_path,
        'GoFiles': list(filter(lambda x: x.endswith('.go'), args.go_srcs)),
        'NonGoFiles': list(filter(lambda x: not x.endswith('.go'), args.go_srcs)),
        'ImportMap': import_map,
        'PackageFile': dict(info['packagefile']),
        'Standard': dict(info['standard']),
        'PackageVetx': dict((key, vet_info_output_name(value)) for key, value in info['packagefile']),
        'VetxOnly': False,
        'VetxOutput': vet_info_output_name(args.output),
        'SucceedOnTypecheckFailure': False
    }
    # print >>sys.stderr, json.dumps(data, indent=4)
    return data


def create_vet_config(args, info):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.cfg') as f:
        f.write(json.dumps(info))
        return f.name


def decode_vet_report(json_report):
    report = ''
    if json_report:
        try:
            full_diags = json.JSONDecoder(encoding='UTF-8').decode(json_report)
        except ValueError:
            report = json_report
        else:
            messages = []
            for _, module_diags in full_diags.iteritems():
                for _, type_diags in module_diags.iteritems():
                     for diag in type_diags:
                         messages.append(u'{}: {}'.format(diag['posn'], diag['message']))
            report = '\n'.join(sorted(messages)).encode('UTF-8')

    return report


def dump_vet_report(args, report):
    if report:
        report = report.replace(args.build_root, '$B')
        report = report.replace(args.source_root, '$S')
    with open(args.vet_report_output, 'w') as f:
        f.write(report)


def read_vet_report(args):
    assert args
    report = ''
    if os.path.exists(args.vet_report_output):
        with open(args.vet_report_output, 'r') as f:
            report += f.read()
    return report


def dump_vet_report_for_tests(args, *test_args_list):
    dump_vet_report(args, reduce(lambda x, y: x + read_vet_report(y), filter(None, test_args_list), ''))


def do_vet(args):
    assert args.vet
    info = gen_vet_info(args)
    vet_config = create_vet_config(args, info)
    cmd = [args.go_vet, '-json']
    if args.vet_flags:
        cmd.extend(args.vet_flags)
    cmd.append(vet_config)
    # print >>sys.stderr, '>>>> [{}]'.format(' '.join(cmd))
    p_vet = subprocess.Popen(cmd, stdin=None, stderr=subprocess.PIPE, stdout=subprocess.PIPE, cwd=args.source_root)
    vet_out, vet_err = p_vet.communicate()
    report = decode_vet_report(vet_out) if vet_out else ''
    dump_vet_report(args, report)
    if p_vet.returncode:
        raise subprocess.CalledProcessError(returncode=p_vet.returncode, cmd=cmd, output=vet_err)


def _do_compile_go(args):
    import_path, is_std_module = args.import_path, args.is_std
    cmd = [
        args.go_compile,
        '-o',
        args.output,
        '-p',
        import_path,
        '-D',
        '""',
        '-goversion',
        'go{}'.format(args.goversion)
    ]
    cmd.extend(get_trimpath_args(args))
    if is_std_module:
        cmd.append('-std')
        if import_path == 'runtime' or import_path.startswith('runtime/internal/'):
            cmd.append('-+')
    import_config_name = create_import_config(args.peers, True, args.import_map, args.module_map)
    if import_config_name:
        cmd += ['-importcfg', import_config_name]
    else:
        if import_path == 'unsafe' or len(args.objects) > 0 or args.asmhdr:
            pass
        else:
            cmd.append('-complete')
    if args.asmhdr:
        cmd += ['-asmhdr', args.asmhdr]
    if compare_versions('1.12', args.goversion) >= 0:
        if args.symabis:
            cmd += ['-symabis'] + args.symabis
        if compare_versions('1.13', args.goversion) >= 0:
            pass
        elif import_path in ('runtime', 'runtime/internal/atomic'):
            cmd.append('-allabis')
    compile_workers = '4'
    if args.compile_flags:
        if import_path == 'runtime' or import_path.startswith('runtime/'):
            cmd.extend(x for x in args.compile_flags if x not in COMPILE_OPTIMIZATION_FLAGS)
        else:
            cmd.extend(args.compile_flags)
        if '-race' in args.compile_flags:
            compile_workers = '1'
    cmd += ['-pack', '-c={}'.format(compile_workers)]
    cmd += args.go_srcs
    call(cmd, args.build_root)


class VetThread(threading.Thread):

    def __init__(self, target, args):
        super(VetThread, self).__init__(target=target, args=args)
        self.exc_info = None

    def run(self):
        try:
            super(VetThread, self).run()
        except:
            self.exc_info = sys.exc_info()

    def join_with_exception(self, reraise_exception):
        self.join()
        if reraise_exception and self.exc_info:
            raise self.exc_info[0], self.exc_info[1], self.exc_info[2]


def do_compile_go(args):
    raise_exception_from_vet = False
    if args.vet:
        run_vet = VetThread(target=do_vet, args=(args,))
        run_vet.start()
    try:
        _do_compile_go(args)
        raise_exception_from_vet = True
    finally:
        if args.vet:
            run_vet.join_with_exception(raise_exception_from_vet)


def do_compile_asm(args):
    assert(len(args.srcs) == 1 and len(args.asm_srcs) == 1)
    cmd = [args.go_asm]
    cmd += get_trimpath_args(args)
    cmd += ['-I', args.output_root, '-I', os.path.join(args.pkg_root, 'include')]
    cmd += ['-D', 'GOOS_' + args.targ_os, '-D', 'GOARCH_' + args.targ_arch, '-o', args.output]
    if args.asm_flags:
        cmd += args.asm_flags
    cmd += args.asm_srcs
    call(cmd, args.build_root)


def do_link_lib(args):
    if len(args.asm_srcs) > 0:
        asmargs = copy_args(args)
        asmargs.asmhdr = os.path.join(asmargs.output_root, 'go_asm.h')
        do_compile_go(asmargs)
        for src in asmargs.asm_srcs:
            asmargs.srcs = [src]
            asmargs.asm_srcs = [src]
            asmargs.output = os.path.join(asmargs.output_root, os.path.basename(src) + '.o')
            do_compile_asm(asmargs)
            args.objects.append(asmargs.output)
    else:
        do_compile_go(args)
    if args.objects:
        cmd = [args.go_pack, 'r', args.output] + args.objects + args.sysos
        call(cmd, args.build_root)


def do_link_exe(args):
    assert args.extld is not None
    assert args.non_local_peers is not None
    compile_args = copy_args(args)
    compile_args.output = os.path.join(args.output_root, 'main.a')
    compile_args.real_import_path = compile_args.import_path
    compile_args.import_path = 'main'

    if args.vcs and os.path.isfile(compile_args.vcs):
        build_info = os.path.join('library', 'go', 'core', 'buildinfo')
        if any(map(lambda x: x.startswith(build_info), compile_args.peers)):
            compile_args.go_srcs.append(compile_args.vcs)

    do_link_lib(compile_args)
    cmd = [args.go_link, '-o', args.output]
    import_config_name = create_import_config(args.peers + args.non_local_peers, False, args.import_map, args.module_map)
    if import_config_name:
        cmd += ['-importcfg', import_config_name]
    if args.link_flags:
        cmd += args.link_flags

    if args.mode in ('exe', 'test'):
        cmd.append('-buildmode=exe')
    elif args.mode == 'dll':
        cmd.append('-buildmode=c-shared')
    else:
        assert False, 'Unexpected mode: {}'.format(args.mode)
    cmd.append('-extld={}'.format(args.extld))

    extldflags = []
    if args.extldflags is not None:
        filter_musl = None
        if args.musl:
            cmd.append('-linkmode=external')
            extldflags.append('-static')
            filter_musl = lambda x: not x in ('-lc', '-ldl', '-lm', '-lpthread', '-lrt')
        extldflags += list(filter(filter_musl, args.extldflags))
    if args.cgo_peers is not None and len(args.cgo_peers) > 0:
        is_group = args.targ_os == 'linux'
        if is_group:
            extldflags.append('-Wl,--start-group')
        extldflags.extend(os.path.join(args.build_root, x) for x in args.cgo_peers)
        if is_group:
            extldflags.append('-Wl,--end-group')
    if len(extldflags) > 0:
        cmd.append('-extldflags=' + ' '.join(extldflags))
    cmd.append(compile_args.output)
    call(cmd, args.build_root)


def gen_cover_info(args):
    lines = []
    lines.extend([
        """
var (
    coverCounters = make(map[string][]uint32)
    coverBlocks = make(map[string][]testing.CoverBlock)
)
        """,
        'func init() {',
    ])
    for var, file in (x.split(':') for x in args.cover_info):
        lines.append('    coverRegisterFile("{file}", _cover0.{var}.Count[:], _cover0.{var}.Pos[:], _cover0.{var}.NumStmt[:])'.format(file=file, var=var))
    lines.extend([
        '}',
        """
func coverRegisterFile(fileName string, counter []uint32, pos []uint32, numStmts []uint16) {
    if 3*len(counter) != len(pos) || len(counter) != len(numStmts) {
        panic("coverage: mismatched sizes")
    }
    if coverCounters[fileName] != nil {
        // Already registered.
        return
    }
    coverCounters[fileName] = counter
    block := make([]testing.CoverBlock, len(counter))
    for i := range counter {
        block[i] = testing.CoverBlock{
            Line0: pos[3*i+0],
            Col0: uint16(pos[3*i+2]),
            Line1: pos[3*i+1],
            Col1: uint16(pos[3*i+2]>>16),
            Stmts: numStmts[i],
        }
    }
    coverBlocks[fileName] = block
}
        """,
    ])
    return lines


def filter_out_skip_tests(tests, skip_tests):
    skip_set = set(skip_tests)
    return filter(lambda x: x not in skip_set, tests)


def gen_test_main(args, test_lib_args, xtest_lib_args):
    assert args and (test_lib_args or xtest_lib_args)
    test_miner = args.test_miner
    test_module_path = test_lib_args.import_path if test_lib_args else xtest_lib_args.import_path
    is_cover = args.cover_info and len(args.cover_info) > 0

    # Prepare GOPATH
    # $BINDIR
    #    |- __go__
    #        |- src
    #        |- pkg
    #            |- ${TARGET_OS}_${TARGET_ARCH}
    go_path_root = os.path.join(args.output_root, '__go__')
    test_src_dir = os.path.join(go_path_root, 'src')
    target_os_arch = '_'.join([args.targ_os, args.targ_arch])
    test_pkg_dir = os.path.join(go_path_root, 'pkg', target_os_arch, os.path.dirname(test_module_path))
    os.makedirs(test_pkg_dir)

    my_env = os.environ.copy()
    my_env['GOROOT'] = ''
    my_env['GOPATH'] = go_path_root
    my_env['GOARCH'] = args.targ_arch
    my_env['GOOS'] = args.targ_os

    tests = []
    xtests = []
    os_symlink = get_symlink_or_copyfile()

    # Get the list of "internal" tests
    if test_lib_args:
        os.makedirs(os.path.join(test_src_dir, test_module_path))
        os_symlink(test_lib_args.output, os.path.join(test_pkg_dir, os.path.basename(test_module_path) + '.a'))
        cmd = [test_miner, '-benchmarks', '-tests', test_module_path]
        tests = filter(lambda x: len(x) > 0, (call(cmd, test_lib_args.output_root, my_env) or '').strip().split('\n'))
        if args.skip_tests:
            tests = filter_out_skip_tests(tests, args.skip_tests)
    test_main_found = '#TestMain' in tests

    # Get the list of "external" tests
    if xtest_lib_args:
        xtest_module_path = xtest_lib_args.import_path
        os.makedirs(os.path.join(test_src_dir, xtest_module_path))
        os_symlink(xtest_lib_args.output, os.path.join(test_pkg_dir, os.path.basename(xtest_module_path) + '.a'))
        cmd = [test_miner, '-benchmarks', '-tests', xtest_module_path]
        xtests = filter(lambda x: len(x) > 0, (call(cmd, xtest_lib_args.output_root, my_env) or '').strip().split('\n'))
        if args.skip_tests:
            xtests = filter_out_skip_tests(xtests, args.skip_tests)
    xtest_main_found = '#TestMain' in xtests

    test_main_package = None
    if test_main_found and xtest_main_found:
        assert False, 'multiple definition of TestMain'
    elif test_main_found:
        test_main_package = '_test'
    elif xtest_main_found:
        test_main_package = '_xtest'

    shutil.rmtree(go_path_root)

    lines = ['package main', '', 'import (']
    if test_main_package is None:
        lines.append('    "os"')
    lines.extend(['    "testing"', '    "testing/internal/testdeps"'])

    if len(tests) > 0:
        lines.append('    _test "{}"'.format(test_module_path))
    elif test_lib_args:
        lines.append('    _ "{}"'.format(test_module_path))

    if len(xtests) > 0:
        lines.append('    _xtest "{}"'.format(xtest_module_path))
    elif xtest_lib_args:
        lines.append('    _ "{}"'.format(xtest_module_path))

    if is_cover:
        lines.append('    _cover0 "{}"'.format(test_module_path))
    lines.extend([')', ''])

    for kind in ['Test', 'Benchmark', 'Example']:
        lines.append('var {}s = []testing.Internal{}{{'.format(kind.lower(), kind))
        for test in list(filter(lambda x: x.startswith(kind), tests)):
            lines.append('    {{"{test}", _test.{test}}},'.format(test=test))
        for test in list(filter(lambda x: x.startswith(kind), xtests)):
            lines.append('    {{"{test}", _xtest.{test}}},'.format(test=test))
        lines.extend(['}', ''])

    if is_cover:
        lines.extend(gen_cover_info(args))

    lines.append('func main() {')
    if is_cover:
        lines.extend([
            '    testing.RegisterCover(testing.Cover{',
            '        Mode: "set",',
            '        Counters: coverCounters,',
            '        Blocks: coverBlocks,',
            '        CoveredPackages: "",',
            '    })',
        ])
    lines.extend([
        '    m := testing.MainStart(testdeps.TestDeps{}, tests, benchmarks, examples)'
        '',
    ])

    if test_main_package:
        lines.append('    {}.TestMain(m)'.format(test_main_package))
    else:
        lines.append('    os.Exit(m.Run())')
    lines.extend(['}', ''])

    content = '\n'.join(lines)
    # print >>sys.stderr, content
    return content


def do_link_test(args):
    assert args.srcs or args.xtest_srcs
    assert args.test_miner is not None

    test_module_path = get_source_path(args)
    test_import_path, _ = get_import_path(test_module_path)

    test_lib_args = copy_args(args) if args.srcs else None
    xtest_lib_args = copy_args(args) if args.xtest_srcs else None

    ydx_file_name = None
    xtest_ydx_file_name = None
    need_append_ydx = test_lib_args and xtest_lib_args and args.ydx_file and args.vet_flags
    if need_append_ydx:
        def find_ydx_file_name(name, flags):
            for i, elem in enumerate(flags):
                if elem.endswith(name):
                    return (i, elem)
            assert False, 'Unreachable code'

        idx, ydx_file_name = find_ydx_file_name(xtest_lib_args.ydx_file, xtest_lib_args.vet_flags)
        xtest_ydx_file_name = '{}_xtest'.format(ydx_file_name)
        xtest_lib_args.vet_flags = copy.copy(xtest_lib_args.vet_flags)
        xtest_lib_args.vet_flags[idx] = xtest_ydx_file_name

    if test_lib_args:
        test_lib_args.output = os.path.join(args.output_root, 'test.a')
        test_lib_args.vet_report_output = vet_report_output_name(test_lib_args.output)
        test_lib_args.module_path = test_module_path
        test_lib_args.import_path = test_import_path
        do_link_lib(test_lib_args)

    if xtest_lib_args:
        xtest_lib_args.srcs = xtest_lib_args.xtest_srcs
        classify_srcs(xtest_lib_args.srcs, xtest_lib_args)
        xtest_lib_args.output = os.path.join(args.output_root, 'xtest.a')
        xtest_lib_args.vet_report_output = vet_report_output_name(xtest_lib_args.output)
        xtest_lib_args.module_path = test_module_path + '_test'
        xtest_lib_args.import_path = test_import_path + '_test'
        if test_lib_args:
            xtest_lib_args.module_map[test_import_path] = test_lib_args.output
        need_append_ydx = args.ydx_file and args.srcs and args.vet_flags
        do_link_lib(xtest_lib_args)

    if need_append_ydx:
        with open(os.path.join(args.build_root, ydx_file_name), 'ab') as dst_file:
            with open(os.path.join(args.build_root, xtest_ydx_file_name), 'rb') as src_file:
                dst_file.write(src_file.read())

    test_main_content = gen_test_main(args, test_lib_args, xtest_lib_args)
    test_main_name = os.path.join(args.output_root, '_test_main.go')
    with open(test_main_name, "w") as f:
        f.write(test_main_content)
    test_args = copy_args(args)
    test_args.srcs = [test_main_name]
    if test_args.test_import_path is None:
        # it seems that we can do it unconditionally, but this kind
        # of mangling doesn't really looks good to me and we leave it
        # for pure GO_TEST module
        test_args.module_path = test_args.module_path + '___test_main__'
        test_args.import_path = test_args.import_path + '___test_main__'
    classify_srcs(test_args.srcs, test_args)
    if test_lib_args:
        test_args.module_map[test_lib_args.import_path] = test_lib_args.output
    if xtest_lib_args:
        test_args.module_map[xtest_lib_args.import_path] = xtest_lib_args.output

    if args.vet:
        dump_vet_report_for_tests(test_args, test_lib_args, xtest_lib_args)
    test_args.vet = False

    do_link_exe(test_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prefix_chars='+')
    parser.add_argument('++mode', choices=['dll', 'exe', 'lib', 'test'], required=True)
    parser.add_argument('++srcs', nargs='*', required=True)
    parser.add_argument('++cgo-srcs', nargs='*')
    parser.add_argument('++test_srcs', nargs='*')
    parser.add_argument('++xtest_srcs', nargs='*')
    parser.add_argument('++cover_info', nargs='*')
    parser.add_argument('++output', nargs='?', default=None)
    parser.add_argument('++source-root', default=None)
    parser.add_argument('++build-root', required=True)
    parser.add_argument('++tools-root', default=None)
    parser.add_argument('++output-root', required=True)
    parser.add_argument('++toolchain-root', required=True)
    parser.add_argument('++host-os', choices=['linux', 'darwin', 'windows'], required=True)
    parser.add_argument('++host-arch', choices=['amd64'], required=True)
    parser.add_argument('++targ-os', choices=['linux', 'darwin', 'windows'], required=True)
    parser.add_argument('++targ-arch', choices=['amd64', 'x86'], required=True)
    parser.add_argument('++peers', nargs='*')
    parser.add_argument('++non-local-peers', nargs='*')
    parser.add_argument('++cgo-peers', nargs='*')
    parser.add_argument('++asmhdr', nargs='?', default=None)
    parser.add_argument('++test-import-path', nargs='?')
    parser.add_argument('++test-miner', nargs='?')
    parser.add_argument('++arc-project-prefix', nargs='?', default=arc_project_prefix)
    parser.add_argument('++std-lib-prefix', nargs='?', default=std_lib_prefix)
    parser.add_argument('++extld', nargs='?', default=None)
    parser.add_argument('++extldflags', nargs='+', default=None)
    parser.add_argument('++goversion', required=True)
    parser.add_argument('++asm-flags', nargs='*')
    parser.add_argument('++compile-flags', nargs='*')
    parser.add_argument('++link-flags', nargs='*')
    parser.add_argument('++vcs', nargs='?', default=None)
    parser.add_argument('++vet', nargs='?', const=True, default=False)
    parser.add_argument('++vet-flags', nargs='*', default=None)
    parser.add_argument('++vet-info-ext', default=vet_info_ext)
    parser.add_argument('++vet-report-ext', default=vet_report_ext)
    parser.add_argument('++musl', action='store_true')
    parser.add_argument('++skip-tests', nargs='*', default=None)
    parser.add_argument('++ydx-file', default='')
    parser.add_argument('++debug-root-map', default=None)
    args = parser.parse_args()

    preprocess_args(args)

    arc_project_prefix = args.arc_project_prefix
    std_lib_prefix = args.std_lib_prefix
    vet_info_ext = args.vet_info_ext
    vet_report_ext = args.vet_report_ext

    try:
        os.unlink(args.output)
    except OSError:
        pass

    # We are going to support only 'lib', 'exe' and 'cgo' build modes currently
    # and as a result we are going to generate only one build node per module
    # (or program)
    dispatch = {
        'exe': do_link_exe,
        'dll': do_link_exe,
        'lib': do_link_lib,
        'test': do_link_test
    }

    exit_code = 1
    try:
        dispatch[args.mode](args)
        exit_code = 0
    except KeyError:
        print >>sys.stderr, 'Unknown build mode [{}]...'.format(args.mode)
    except subprocess.CalledProcessError as e:
        print >>sys.stderr, '{} returned non-zero exit code {}. stop.'.format(' '.join(e.cmd), e.returncode)
        print >>sys.stderr, e.output
        exit_code = e.returncode
    except Exception as e:
       print >>sys.stderr, "Unhandled exception [{}]...".format(str(e))
    sys.exit(exit_code)
