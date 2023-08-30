import sys
import subprocess
import optparse

import process_command_files as pcf

from process_whole_archive_option import ProcessWholeArchiveOption


def get_leaks_suppressions(cmd):
    supp, newcmd = [], []
    for arg in cmd:
        if arg.endswith(".supp"):
            supp.append(arg)
        else:
            newcmd.append(arg)
    return supp, newcmd


MUSL_LIBS = '-lc', '-lcrypt', '-ldl', '-lm', '-lpthread', '-lrt', '-lutil'


CUDA_LIBRARIES = {
    '-lcublas_static': '-lcublas',
    '-lcublasLt_static': '-lcublasLt',
    '-lcudart_static': '-lcudart',
    '-lcudnn_static': '-lcudnn',
    '-lcufft_static_nocallback': '-lcufft',
    '-lcurand_static': '-lcurand',
    '-lcusolver_static': '-lcusolver',
    '-lcusparse_static': '-lcusparse',
    '-lmyelin_compiler_static': '-lmyelin',
    '-lmyelin_executor_static': '-lnvcaffe_parser',
    '-lmyelin_pattern_library_static': '',
    '-lmyelin_pattern_runtime_static': '',
    '-lnvinfer_static': '-lnvinfer',
    '-lnvinfer_plugin_static': '-lnvinfer_plugin',
    '-lnvonnxparser_static': '-lnvonnxparser',
    '-lnvparsers_static': '-lnvparsers'
}


def remove_excessive_flags(cmd):
    flags = []
    for flag in cmd:
        if not flag.endswith('.ios.interface') and not flag.endswith('.pkg.fake'):
            flags.append(flag)
    return flags


def fix_sanitize_flag(cmd):
    """
    Remove -fsanitize=address flag if sanitazers are linked explicitly for linux target.
    """
    for flag in cmd:
        if flag.startswith('--target') and 'linux' not in flag.lower():
            # use toolchained sanitize libraries
            return cmd
    if 'CLANG16_YES_PLEASE' in str(cmd):
        CLANG_RT = 'contrib/libs/clang16-rt/lib/'
    else:
        CLANG_RT = 'contrib/libs/clang14-rt/lib/'
    sanitize_flags = {
        '-fsanitize=address': CLANG_RT + 'asan',
        '-fsanitize=memory': CLANG_RT + 'msan',
        '-fsanitize=leak': CLANG_RT + 'lsan',
        '-fsanitize=undefined': CLANG_RT + 'ubsan',
        '-fsanitize=thread': CLANG_RT + 'tsan'
    }

    used_sanitize_libs = []
    aux = []
    for flag in cmd:
        if flag.startswith('-fsanitize-coverage='):
            # do not link sanitizer libraries from clang
            aux.append('-fno-sanitize-link-runtime')
        if flag in sanitize_flags and any(s.startswith(sanitize_flags[flag]) for s in cmd):
            # exclude '-fsanitize=' if appropriate library is linked explicitly
            continue
        if any(flag.startswith(lib) for lib in sanitize_flags.values()):
            used_sanitize_libs.append(flag)
            continue
        aux.append(flag)

    # move sanitize libraries out of the repeatedly searched group of archives
    flags = []
    for flag in aux:
        if flag == '-Wl,--start-group':
            flags += ['-Wl,--whole-archive'] + used_sanitize_libs + ['-Wl,--no-whole-archive']
        flags.append(flag)

    return flags


def fix_cmd_for_musl(cmd):
    flags = []
    for flag in cmd:
        if flag not in MUSL_LIBS:
            flags.append(flag)
    return flags


def fix_cmd_for_dynamic_cuda(cmd):
    flags = []
    for flag in cmd:
        if flag in CUDA_LIBRARIES:
            flags.append(CUDA_LIBRARIES[flag])
        else:
            flags.append(flag)
    return flags


def gen_default_suppressions(inputs, output, source_root):
    import collections
    import os

    supp_map = collections.defaultdict(set)
    for filename in inputs:
        sanitizer = os.path.basename(filename).split('.', 1)[0]
        with open(os.path.join(source_root, filename)) as src:
            for line in src:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                supp_map[sanitizer].add(line)

    with open(output, "wb") as dst:
        for supp_type, supps in supp_map.items():
            dst.write('extern "C" const char *__%s_default_suppressions() {\n' % supp_type)
            dst.write('    return "{}";\n'.format('\\n'.join(sorted(supps))))
            dst.write('}\n')


def fix_blas_resolving(cmd):
    # Intel mkl comes as a precompiled static library and thus can not be recompiled with sanitizer runtime instrumentation.
    # That's why we prefer to use cblas instead of Intel mkl as a drop-in replacement under sanitizers.
    # But if the library has dependencies on mkl and cblas simultaneously, it will get a linking error.
    # Hence we assume that it's probably compiling without sanitizers and we can easily remove cblas to prevent multiple definitions of the same symbol at link time.
    for arg in cmd:
        if arg.startswith('contrib/libs') and arg.endswith('mkl-lp64.a'):
            return [arg for arg in cmd if not arg.endswith('libcontrib-libs-cblas.a')]
    return cmd


def parse_args():
    parser = optparse.OptionParser()
    parser.disable_interspersed_args()
    parser.add_option('--musl', action='store_true')
    parser.add_option('--custom-step')
    parser.add_option('--python')
    parser.add_option('--source-root')
    parser.add_option('--dynamic-cuda', action='store_true')
    parser.add_option('--arch')
    parser.add_option('--linker-output')
    parser.add_option('--whole-archive-peers', action='append')
    parser.add_option('--whole-archive-libs', action='append')
    return parser.parse_args()


if __name__ == '__main__':
    opts, args = parse_args()
    args = pcf.skip_markers(args)

    cmd = fix_blas_resolving(args)
    cmd = remove_excessive_flags(cmd)
    if opts.musl:
        cmd = fix_cmd_for_musl(cmd)

    cmd = fix_sanitize_flag(cmd)

    if 'ld.lld' in str(cmd):
        if '-fPIE' in str(cmd) or '-fPIC' in str(cmd):
            # support explicit PIE
            pass
        else:
            cmd.append('-Wl,-no-pie')

    if opts.dynamic_cuda:
        cmd = fix_cmd_for_dynamic_cuda(cmd)
    cmd = ProcessWholeArchiveOption(opts.arch, opts.whole_archive_peers, opts.whole_archive_libs).construct_cmd(cmd)

    if opts.custom_step:
        assert opts.python
        subprocess.check_call([opts.python] + [opts.custom_step] + args)

    supp, cmd = get_leaks_suppressions(cmd)
    if supp:
        src_file = "default_suppressions.cpp"
        gen_default_suppressions(supp, src_file, opts.source_root)
        cmd += [src_file]

    if opts.linker_output:
        stdout = open(opts.linker_output, 'w')
    else:
        stdout = sys.stdout

    rc = subprocess.call(cmd, shell=False, stderr=sys.stderr, stdout=stdout)
    sys.exit(rc)
