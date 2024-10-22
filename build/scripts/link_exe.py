import itertools
import os
import os.path
import sys
import subprocess
import optparse
import textwrap

import process_command_files as pcf
import thinlto_cache

from process_whole_archive_option import ProcessWholeArchiveOption
from fix_py2_protobuf import fix_py2


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
    '-lcupti_static': '-lcupti',
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
    '-lnvparsers_static': '-lnvparsers',
    '-lnvrtc_static': '-lnvrtc',
    '-lnvrtc-builtins_static': '-lnvrtc-builtins',
    '-lnvptxcompiler_static': '',
    '-lnppc_static': '-lnppc',
    '-lnppial_static': '-lnppial',
    '-lnppicc_static': '-lnppicc',
    '-lnppicom_static': '-lnppicom',
    '-lnppidei_static': '-lnppidei',
    '-lnppif_static': '-lnppif',
    '-lnppig_static': '-lnppig',
    '-lnppim_static': '-lnppim',
    '-lnppist_static': '-lnppist',
    '-lnppisu_static': '-lnppisu',
    '-lnppitc_static': '-lnppitc',
    '-lnpps_static': '-lnpps',
}


class CUDAManager:
    def __init__(self, known_arches, nvprune_exe):
        self.fatbin_libs = self._known_fatbin_libs(set(CUDA_LIBRARIES))

        self.prune_args = []
        if known_arches:
            for arch in known_arches.split(':'):
                self.prune_args.append('-gencode')
                self.prune_args.append(self._arch_flag(arch))

        self.nvprune_exe = nvprune_exe

    def has_cuda_fatbins(self, cmd):
        return bool(set(cmd) & self.fatbin_libs)

    @property
    def can_prune_libs(self):
        return self.prune_args and self.nvprune_exe

    def _known_fatbin_libs(self, libs):
        libs_wo_device_code = {
            '-lcudart_static',
            '-lcupti_static',
            '-lnppc_static',
        }
        return set(libs) - libs_wo_device_code

    def _arch_flag(self, arch):
        _, ver = arch.split('_', 1)
        return 'arch=compute_{},code={}'.format(ver, arch)

    def prune_lib(self, inp_fname, out_fname):
        if self.prune_args:
            prune_command = [self.nvprune_exe] + self.prune_args + ['--output-file', out_fname, inp_fname]
            subprocess.check_call(prune_command)

    def write_linker_script(self, f):
        # This script simply says:
        # * Place all `.nv_fatbin` input sections from all input files into one `.nv_fatbin` output section of output file
        # * Place it after `.bss` section
        #
        # Motivation can be found here: https://maskray.me/blog/2021-07-04-sections-and-overwrite-sections#insert-before-and-insert-after
        # TL;DR - we put section with a lot of GPU code directly after the last meaningful section in the binary
        # (which turns out to be .bss)
        # In that case, we decrease chances of relocation overflows from .text to .bss,
        # because now these sections are close to each other
        script = textwrap.dedent("""
            SECTIONS {
                .nv_fatbin : { *(.nv_fatbin) }
            } INSERT AFTER .bss
        """).strip()

        f.write(script)


def tmpdir_generator(base_path, prefix):
    for idx in itertools.count():
        path = os.path.abspath(os.path.join(base_path, prefix + '_' + str(idx)))
        os.makedirs(path)
        yield path


def process_cuda_library_by_external_tool(cmd, build_root, tool_name, callable_tool_executor, allowed_cuda_libs):
    tmpdir_gen = tmpdir_generator(build_root, 'cuda_' + tool_name + '_libs')

    new_flags = []
    cuda_deps = set()

    # Because each directory flag only affects flags that follow it,
    # for correct pruning we need to process that in reversed order
    for flag in reversed(cmd):
        if flag in allowed_cuda_libs:
            cuda_deps.add('lib' + flag[2:] + '.a')
            flag += '_' + tool_name
        elif flag.startswith('-L') and os.path.exists(flag[2:]) and os.path.isdir(flag[2:]) and any(f in cuda_deps for f in os.listdir(flag[2:])):
            from_dirpath = flag[2:]
            from_deps = list(cuda_deps & set(os.listdir(from_dirpath)))

            if from_deps:
                to_dirpath = next(tmpdir_gen)

                for f in from_deps:
                    from_path = os.path.join(from_dirpath, f)
                    to_path = os.path.join(to_dirpath, f[:-2] + '_' + tool_name +'.a')
                    callable_tool_executor(from_path, to_path)
                    cuda_deps.remove(f)

                # do not remove current directory
                # because it can contain other libraries we want link to
                # instead we just add new directory with processed by tool libs
                new_flags.append('-L' + to_dirpath)

        new_flags.append(flag)

    assert not cuda_deps, ('Unresolved CUDA deps: ' + ','.join(cuda_deps))
    return reversed(new_flags)


def process_cuda_libraries_by_objcopy(cmd, build_root, objcopy_exe):
    if not objcopy_exe:
        return cmd

    def run_objcopy(from_path, to_path):
        rename_section_command = [objcopy_exe, "--rename-section", ".ctors=.init_array", from_path, to_path]
        subprocess.check_call(rename_section_command)

    possible_libraries = set(CUDA_LIBRARIES.keys())
    possible_libraries.update([
        '-lcudadevrt',
        '-lcufilt',
        '-lculibos',
    ])
    possible_libraries.update([
        lib_name + "_pruner" for lib_name in possible_libraries
    ])

    return process_cuda_library_by_external_tool(list(cmd), build_root, 'objcopy', run_objcopy, possible_libraries)


def process_cuda_libraries_by_nvprune(cmd, cuda_manager, build_root):
    if not cuda_manager.has_cuda_fatbins(cmd):
        return cmd

    # add custom linker script
    to_dirpath = next(tmpdir_generator(build_root, 'cuda_linker_script'))
    script_path = os.path.join(to_dirpath, 'script')
    with open(script_path, 'w') as f:
        cuda_manager.write_linker_script(f)
    flags_with_linker = list(cmd) + ['-Wl,--script={}'.format(script_path)]

    if not cuda_manager.can_prune_libs:
        return flags_with_linker

    return process_cuda_library_by_external_tool(flags_with_linker, build_root, 'pruner', cuda_manager.prune_lib, cuda_manager.fatbin_libs)


def remove_excessive_flags(cmd):
    flags = []
    for flag in cmd:
        if not flag.endswith('.ios.interface') and not flag.endswith('.pkg.fake'):
            flags.append(flag)
    return flags


def fix_sanitize_flag(cmd, opts):
    """
    Remove -fsanitize=address flag if sanitazers are linked explicitly for linux target.
    """
    for flag in cmd:
        if flag.startswith('--target') and 'linux' not in flag.lower():
            # use toolchained sanitize libraries
            return cmd
    assert opts.clang_ver
    CLANG_RT = 'contrib/libs/clang' + opts.clang_ver + '-rt/lib/'
    sanitize_flags = {
        '-fsanitize=address': CLANG_RT + 'asan',
        '-fsanitize=memory': CLANG_RT + 'msan',
        '-fsanitize=leak': CLANG_RT + 'lsan',
        '-fsanitize=undefined': CLANG_RT + 'ubsan',
        '-fsanitize=thread': CLANG_RT + 'tsan',
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
    parser.add_option('--build-root')
    parser.add_option('--clang-ver')
    parser.add_option('--dynamic-cuda', action='store_true')
    parser.add_option('--cuda-architectures',
                      help='List of supported CUDA architectures, separated by ":" (e.g. "sm_52:compute_70:lto_90a"')
    parser.add_option('--nvprune-exe')
    parser.add_option('--objcopy-exe')
    parser.add_option('--arch')
    parser.add_option('--linker-output')
    parser.add_option('--whole-archive-peers', action='append')
    parser.add_option('--whole-archive-libs', action='append')
    thinlto_cache.add_options(parser)
    return parser.parse_args()


if __name__ == '__main__':
    opts, args = parse_args()
    args = pcf.skip_markers(args)

    cmd = fix_blas_resolving(args)
    cmd = fix_py2(cmd)
    cmd = remove_excessive_flags(cmd)
    if opts.musl:
        cmd = fix_cmd_for_musl(cmd)

    cmd = fix_sanitize_flag(cmd, opts)

    if opts.dynamic_cuda:
        cmd = fix_cmd_for_dynamic_cuda(cmd)
    else:
        cuda_manager = CUDAManager(opts.cuda_architectures, opts.nvprune_exe)
        cmd = process_cuda_libraries_by_nvprune(cmd, cuda_manager, opts.build_root)
        cmd = process_cuda_libraries_by_objcopy(cmd, opts.build_root, opts.objcopy_exe)
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

    thinlto_cache.preprocess(opts, cmd)
    rc = subprocess.call(cmd, shell=False, stderr=sys.stderr, stdout=stdout)
    thinlto_cache.postprocess(opts)

    sys.exit(rc)
