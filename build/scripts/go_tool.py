import argparse
import copy
import os
import shutil
import subprocess
import sys
import tempfile


arc_project_prefix = 'a.yandex-team.ru/'
std_lib_prefix = 'contrib/go/_std/src/'
vendor_prefix = 'vendor/'


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
    index = get_vendor_index(import_path)
    if index >= 0:
        index += len(vendor_prefix)
        import_path = import_path[index:]
    elif not is_std_module:
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


def create_import_config(peers, import_map={}, module_map={}):
    content = ''
    for key, value in import_map.items():
        content += 'importmap {}={}\n'.format(key, value)
    for peer in peers:
        peer_import_path, _ = get_import_path(os.path.dirname(peer))
        index = get_vendor_index(peer_import_path)
        if index >= 0:
            index += len(vendor_prefix)
            content += 'importmap {}={}\n'.format(peer_import_path[index:], peer_import_path)
        content += 'packagefile {}={}\n'.format(peer_import_path, os.path.join(args.build_root, peer))
    for key, value in module_map.items():
        content += 'packagefile {}={}\n'.format(key, value)
    if len(content) > 0:
        # print >>sys.stderr, content
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            return f.name
    return None


def do_compile_go(args):
    import_path, is_std_module = get_import_path(args.module_path)
    cmd = [args.go_compile, '-o', args.output, '-trimpath', args.build_root, '-p', import_path, '-D', '""']
    cmd += ['-goversion', 'go' + args.goversion]
    if is_std_module:
        cmd.append('-std')
        if import_path == 'runtime':
            cmd.append('-+')
    import_config_name = create_import_config(args.peers, args.import_map, args.module_map)
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
        if import_path in ('runtime', 'runtime/internal/atomic'):
            cmd.append('-allabis')
    cmd += ['-pack', '-c=4'] + args.go_srcs
    call(cmd, args.build_root)


def do_compile_asm(args):
    assert(len(args.srcs) == 1 and len(args.asm_srcs) == 1)
    cmd = [args.go_asm, '-trimpath', args.build_root]
    cmd += ['-I', args.output_root, '-I', os.path.join(args.pkg_root, 'include')]
    cmd += ['-D', 'GOOS_' + args.targ_os, '-D', 'GOARCH_' + args.targ_arch, '-o', args.output] + args.asm_srcs
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
        cmd = [args.go_pack, 'r', args.output] + args.objects
        call(cmd, args.build_root)


def do_link_exe(args):
    assert args.extld is not None
    compile_args = copy_args(args)
    compile_args.output = os.path.join(args.output_root, 'main.a')
    do_link_lib(compile_args)
    cmd = [args.go_link, '-o', args.output]
    import_config_name = create_import_config(args.peers, args.import_map, args.module_map)
    if import_config_name:
        cmd += ['-importcfg', import_config_name]
    cmd += ['-buildmode=exe', '-extld={}'.format(args.extld)]
    extldflags = ''
    if args.extldflags is not None and len(args.extldflags) > 0:
        extldflags = ' '.join(args.extldflags)
    if args.cgo_peers is not None and len(args.cgo_peers) > 0:
        peer_libs = ' '.join(os.path.join(args.build_root, x) for x in args.cgo_peers)
        extldflags += ' ' if len(extldflags) > 0 else ''
        if args.targ_os == 'linux':
            extldflags += '-Wl,--start-group {} -Wl,--end-group'.format(peer_libs)
        else:
            extldflags += peer_libs
    if extldflags:
        cmd.append('-extldflags=' + extldflags)
    cmd.append(compile_args.output)
    call(cmd, args.build_root)


def gen_test_main(args, test_lib_args, xtest_lib_args):
    assert args and (test_lib_args or xtest_lib_args)
    test_miner = args.test_miner
    test_module_path = test_lib_args.module_path if test_lib_args else xtest_lib_args.module_path

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
        cmd = [test_miner, '-tests', test_module_path]
        tests = filter(lambda x: len(x) > 0, (call(cmd, test_lib_args.output_root, my_env) or '').strip().split('\n'))
    test_main_found = '#TestMain' in tests

    # Get the list of "external" tests
    if xtest_lib_args:
        xtest_module_path = xtest_lib_args.module_path
        os.makedirs(os.path.join(test_src_dir, xtest_module_path))
        os_symlink(xtest_lib_args.output, os.path.join(test_pkg_dir, os.path.basename(xtest_module_path) + '.a'))
        cmd = [test_miner, '-tests', xtest_module_path]
        xtests = filter(lambda x: len(x) > 0, (call(cmd, xtest_lib_args.output_root, my_env) or '').strip().split('\n'))
    xtest_main_found = '#TestMain' in xtests

    test_main_package = None
    if test_main_found and xtest_main_found:
        assert False, 'multiple definition of TestMain'
    elif test_main_found:
        test_main_package = '_test'
    elif xtest_main_found:
        test_main_package = '_xtest'

    shutil.rmtree(go_path_root)

    content = """package main

import (
"""
    if test_main_package is None:
        content += """    "os"
"""
    content += """    "testing"
    "testing/internal/testdeps"
"""
    if len(tests) > 0:
        content += '    _test "{}"\n'.format(test_module_path)
    if len(xtests) > 0:
        content += '    _xtest "{}"\n'.format(xtest_module_path)
    content += ')\n\n'

    for kind in ['Test', 'Benchmark', 'Example']:
        content += 'var {}s = []testing.Internal{}{{\n'.format(kind.lower(), kind)
        for test in list(filter(lambda x: x.startswith(kind), tests)):
            content += '    {{"{test}", _test.{test}}},\n'.format(test=test)
        for test in list(filter(lambda x: x.startswith(kind), xtests)):
            content += '    {{"{test}", _xtest.{test}}},\n'.format(test=test)
        content += '}\n\n'

    content += """func main() {
    m := testing.MainStart(testdeps.TestDeps{}, tests, benchmarks, examples)
"""
    if test_main_package:
        content += '    {}.TestMain(m)'.format(test_main_package)
    else:
        content += '    os.Exit(m.Run())'
    content += """
}
"""
    # print >>sys.stderr, content
    return content


def do_link_test(args):
    assert args.srcs or args.xtest_srcs
    assert args.test_miner is not None

    test_import_path, _ = get_import_path(args.test_import_path or args.module_path)

    test_lib_args = None
    xtest_lib_args = None

    if args.srcs:
        test_lib_args = copy_args(args)
        test_lib_args.output = os.path.join(args.output_root, 'test.a')
        test_lib_args.module_path = test_import_path
        do_link_lib(test_lib_args)

    if args.xtest_srcs:
        xtest_lib_args = copy_args(args)
        xtest_lib_args.srcs = xtest_lib_args.xtest_srcs
        classify_srcs(xtest_lib_args.srcs, xtest_lib_args)
        xtest_lib_args.output = os.path.join(args.output_root, 'xtest.a')
        xtest_lib_args.module_path = test_import_path + '_test'
        if test_lib_args:
            xtest_lib_args.module_map[test_import_path] = test_lib_args.output
        do_link_lib(xtest_lib_args)

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
    classify_srcs(test_args.srcs, test_args)
    if test_lib_args:
        test_args.module_map[test_import_path] = test_lib_args.output
    if xtest_lib_args:
        test_args.module_map[test_import_path + '_test'] = xtest_lib_args.output
    do_link_exe(test_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prefix_chars='+')
    parser.add_argument('++mode', choices=['lib', 'exe', 'test'], required=True)
    parser.add_argument('++srcs', nargs='*', required=True)
    parser.add_argument('++test_srcs', nargs='*')
    parser.add_argument('++xtest_srcs', nargs='*')
    parser.add_argument('++output', nargs='?', default=None)
    parser.add_argument('++build-root', required=True)
    parser.add_argument('++output-root', required=True)
    parser.add_argument('++tools-root', required=True)
    parser.add_argument('++host-os', choices=['linux', 'darwin', 'windows'], required=True)
    parser.add_argument('++host-arch', choices=['amd64'], required=True)
    parser.add_argument('++targ-os', choices=['linux', 'darwin', 'windows'], required=True)
    parser.add_argument('++targ-arch', choices=['amd64', 'x86'], required=True)
    parser.add_argument('++peers', nargs='*')
    parser.add_argument('++cgo-peers', nargs='*')
    parser.add_argument('++asmhdr', nargs='?', default=None)
    parser.add_argument('++test-import-path', nargs='?')
    parser.add_argument('++test-miner', nargs='?')
    parser.add_argument('++arc-project-prefix', nargs='?', default=arc_project_prefix)
    parser.add_argument('++std-lib-prefix', nargs='?', default=std_lib_prefix)
    parser.add_argument('++extld', nargs='?', default=None)
    parser.add_argument('++extldflags', nargs='+', default=None)
    parser.add_argument('++goversion', required=True)
    args = parser.parse_args()

    args.pkg_root = os.path.join(str(args.tools_root), 'pkg')
    args.tool_root = os.path.join(args.pkg_root, 'tool', '{}_{}'.format(args.host_os, args.host_arch))
    args.go_compile = os.path.join(args.tool_root, 'compile')
    args.go_cgo = os.path.join(args.tool_root, 'cgo')
    args.go_link = os.path.join(args.tool_root, 'link')
    args.go_asm = os.path.join(args.tool_root, 'asm')
    args.go_pack = os.path.join(args.tool_root, 'pack')
    args.output = os.path.normpath(args.output)
    args.build_root = os.path.normpath(args.build_root) + os.path.sep
    args.output_root = os.path.normpath(args.output_root)
    args.import_map = {}
    args.module_map = {}

    assert args.mode == 'test' or args.test_srcs is None and args.xtest_srcs is None
    # add lexical oreder by basename for go sources
    args.srcs.sort(key=lambda x: os.path.basename(x))
    if args.test_srcs:
        args.srcs += sorted(args.test_srcs, key=lambda x: os.path.basename(x))
        del args.test_srcs
    if args.xtest_srcs:
        args.xtest_srcs.sort(key=lambda x: os.path.basename(x))

    arc_project_prefix = args.arc_project_prefix
    std_lib_prefix = args.std_lib_prefix

    # compute root relative module dir path
    assert args.output is None or args.output_root == os.path.dirname(args.output)
    assert args.output_root.startswith(args.build_root)
    args.module_path = args.output_root[len(args.build_root):]
    assert len(args.module_path) > 0

    classify_srcs(args.srcs, args)

    assert args.asmhdr is None or args.word == 'go'

    try:
        os.unlink(args.output)
    except OSError:
        pass

    # We are going to support only 'lib', 'exe' and 'cgo' build modes currently
    # and as a result we are going to generate only one build node per module
    # (or program)
    dispatch = {
        'lib': do_link_lib,
        'exe': do_link_exe,
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
