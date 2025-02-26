import os
import sys
import json
import textwrap
import itertools
import subprocess


CUDA_LIBRARIES = {
    '-lcublas_static': '-lcublas',
    '-lcublasLt_static': '-lcublasLt',
    '-lcudart_static': '-lcudart',
    '-lcudnn_static': '-lcudnn',
    '-lcudnn_adv_infer_static': '-lcudnn',
    '-lcudnn_adv_train_static': '-lcudnn',
    '-lcudnn_cnn_infer_static': '-lcudnn',
    '-lcudnn_cnn_train_static': '-lcudnn',
    '-lcudnn_ops_infer_static': '-lcudnn',
    '-lcudnn_ops_train_static': '-lcudnn',
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


def fix_cmd_for_dynamic_cuda(cmd):
    flags = []
    for flag in cmd:
        if flag in CUDA_LIBRARIES:
            flags.append(CUDA_LIBRARIES[flag])
        else:
            flags.append(flag)
    return flags


def get_flag(f):
    return sys.argv[sys.argv.index(f) + 1]


def parse_kv(args, prefix):
    rest = []
    kv = {}

    for a in args:
        if a.startswith(prefix) and '=' in a:
            k, v = a[len(prefix):].split('=')
            kv[k] = v
        else:
            rest.append(a)

    return rest, kv


if __name__ == '__main__':
    cmd, kv = parse_kv(sys.argv[1:], '-L/CUDA:')

    if '--dynamic-cuda' in cmd:
        cmd = fix_cmd_for_dynamic_cuda(cmd)
    else:
        ca = kv['ARCH']
        nv = kv['NVPRUNE']
        oc = kv['OBJCOPY']

        try:
            br = get_flag('--build-root')
        except ValueError:
            br = os.getcwd()

        cuda_manager = CUDAManager(ca, nv)

        cmd = process_cuda_libraries_by_nvprune(cmd, cuda_manager, br)
        cmd = process_cuda_libraries_by_objcopy(cmd, br, oc)

    sys.stdout.write(json.dumps(list(cmd)))
