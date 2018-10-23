import argparse
import copy
import os
import subprocess
import sys
import tempfile


contrib_go_src_prefix = 'contrib/go/src/'


def get_import_path(module_path):
    assert len(module_path) > 0
    import_path = module_path
    is_std_module = import_path.startswith(contrib_go_src_prefix)
    if is_std_module:
        import_path = import_path[len(contrib_go_src_prefix):]
    assert len(import_path) > 0
    return import_path, is_std_module


def call(cmd, cwd):
    try:
        subprocess.check_output(cmd, stdin=None, stderr=subprocess.STDOUT, cwd=cwd)
    except OSError as e:
        raise Exception('while running %s: %s' % (' '.join(cmd), e))


def create_import_config(peers, need_importmap):
    if peers and len(peers) > 0:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            for peer in peers:
                peer_import_path, _ = get_import_path(os.path.dirname(peer))
                index = peer_import_path.find('vendor/') if need_importmap else -1
                if index == 0 or index > 0 and peer_import_path[index-1] == '/':
                    index += len('vendor/')
                    f.write('importmap {}={}\n'.format(peer_import_path[index:], peer_import_path))
                f.write('packagefile {}={}\n'.format(peer_import_path, os.path.join(args.build_root, peer)))
                # print >>sys.stderr, 'packagefile {}={}'.format(peer_import_path, os.path.join(args.build_root, peer))
            return f.name
    return None


def do_compile_go(args):
    import_path, is_std_module = get_import_path(args.module_path)
    cmd = [args.go_compile, '-o', args.output, '-trimpath', args.build_root, '-p', import_path, '-D', '""']
    if is_std_module:
        cmd.append('-std')
        if import_path == 'runtime':
            cmd.append('-+')
    import_config_name = create_import_config(args.peers, True)
    if import_config_name:
        cmd += ['-importcfg', import_config_name]
    else:
        if import_path == 'contrib/go/src/unsafe' or len(args.objects) > 0 or args.asmhdr:
            pass
        else:
            cmd.append('-complete')
    if args.asmhdr:
        cmd += ['-asmhdr', args.asmhdr]
    cmd += ['-pack', '-c=4'] + args.go_srcs
    call(cmd, args.build_root)


def do_compile_asm(args):
    assert(len(args.srcs) == 1 and len(args.asm_srcs) == 1)
    cmd = [args.go_asm, '-trimpath', args.build_root]
    cmd += ['-I', args.output_root, '-I', os.path.join(args.pkg_root, 'include')]
    cmd += ['-D', 'GOOS_' + args.host_os, '-D', 'GOARCH_' + args.host_arch, '-o', args.output] + args.asm_srcs
    call(cmd, args.build_root)


def do_link_lib(args):
    # assert len(args.objects) == 0
    if len(args.asm_srcs) > 0:
        asmargs = copy.deepcopy(args)
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


def do_link_exe(arg):
    assert args.cc
    compile_args = copy.deepcopy(args)
    compile_args.output = os.path.join(args.output_root, 'main.a')
    do_link_lib(compile_args)
    cmd = [args.go_link, '-o', args.output]
    import_config_name = create_import_config(compile_args.peers, False)
    if import_config_name:
        cmd += ['-importcfg', import_config_name]
    cmd += ['-buildmode=exe', '-extld=' + args.cc, '-linkmode=external', '-extldflags=-lpthread', compile_args.output]
    call(cmd, args.build_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['lib', 'exe'], required=True)
    parser.add_argument('--srcs', nargs='+', required=True)
    parser.add_argument('--output', nargs='?', default=None)
    parser.add_argument('--build-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--tools-root', required=True)
    parser.add_argument('--host-os', choices=['linux', 'darwin', 'windows'], required=True)
    parser.add_argument('--host-arch', choices=['amd64'], required=True)
    parser.add_argument('--peers', nargs='*')
    parser.add_argument('--asmhdr', nargs='?', default=None)
    parser.add_argument('--cc', nargs='?', default=None)
    args = parser.parse_args()

    args.pkg_root = os.path.join(str(args.tools_root), 'pkg')
    args.tool_root = os.path.join(args.pkg_root, 'tool', '{}_{}'.format(args.host_os, args.host_arch))
    args.go_compile = os.path.join(args.tool_root, 'compile')
    args.go_cgo = os.path.join(args.tool_root, 'cgo')
    args.go_link = os.path.join(args.tool_root, 'link')
    args.go_asm = os.path.join(args.tool_root, 'asm')
    args.go_pack = os.path.join(args.tool_root, 'pack')
    args.build_root = os.path.normpath(args.build_root) + os.path.sep
    args.output_root = os.path.dirname(args.output)

    # compute root relative module dir path
    assert args.output is None or args.output_root == os.path.dirname(args.output)
    assert args.output_root.startswith(args.build_root)
    args.module_path = args.output_root[len(args.build_root):]
    assert len(args.module_path) > 0

    args.go_srcs = list(filter(lambda x: x.endswith('.go'), args.srcs))
    args.c_srcs = list(filter(lambda x: x.endswith('.c'), args.srcs))
    args.cxx_srcs = list(filter(lambda x: x.endswith('.cc'), args.srcs))
    args.asm_srcs = list(filter(lambda x: x.endswith('.s'), args.srcs))
    args.objects = list(filter(lambda x: x.endswith('.o') or x.endswith('.obj'), args.srcs))
    args.packages = list(filter(lambda x: x.endswith('.a'), args.srcs))

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
    sys.exit(exit_code)
