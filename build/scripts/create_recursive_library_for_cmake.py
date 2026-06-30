# Custom script is necessary because CMake does not yet support creating static libraries combined with dependencies
# https://gitlab.kitware.com/cmake/cmake/-/issues/22975
#
# This script is intended to be used set as a CXX_LINKER_LAUNCHER property for recursive library targets.
# It parses the linking command and transforms it to archiving commands combining static libraries from dependencies.

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile


class Opts(object):
    def __init__(self, args):
        argparser = argparse.ArgumentParser(allow_abbrev=False)
        argparser.add_argument('--project-binary-dir', required=True)
        argparser.add_argument('--cmake-ar', required=True)
        argparser.add_argument('--cmake-ranlib', required=True)
        argparser.add_argument('--cmake-host-system-name', required=True)
        argparser.add_argument('--cmake-cxx-standard-libraries')
        argparser.add_argument('--global-part-suffix', required=True)
        self.parsed_args, other_args = argparser.parse_known_args(args=args)

        if len(other_args) < 2:
            # must contain at least '--linking-cmdline' and orginal linking tool name
            raise Exception('not enough arguments')
        if other_args[0] != '--linking-cmdline':
            raise Exception("expected '--linking-cmdline' arg, got {}".format(other_args[0]))

        self.is_msvc_compatible_linker = other_args[1].endswith('\\link.exe') or other_args[1].endswith('\\lld-link.exe')

        is_host_system_windows = self.parsed_args.cmake_host_system_name == 'Windows'
        std_libraries_to_exclude_from_input = (
            set(self.parsed_args.cmake_cxx_standard_libraries.split())
            if self.parsed_args.cmake_cxx_standard_libraries is not None
            else set()
        )
        msvc_preserved_option_prefixes = [
            'machine:',
            'nodefaultlib',
            'nologo',
        ]

        self.preserved_options = []

        # these variables can contain paths absolute or relative to project_binary_dir
        self.global_libs_and_objects_input = []
        self.non_global_libs_input = []
        self.output = None

        def is_external_library(path):
            """
            Check whether this library has been built in this CMake project or came from Conan-provided dependencies
            (these use absolute paths).
            If it is a library that is added from some other path (like CUDA) return True
            """
            return not (os.path.exists(path) or os.path.exists(os.path.join(self.parsed_args.project_binary_dir, path)))

        def process_input(args):
            i = 0
            is_in_whole_archive = False

            while i < len(args):
                arg = args[i]
                if is_host_system_windows and ((arg[0] == '/') or (arg[0] == '-')):
                    arg_wo_specifier_lower = arg[1:].lower()
                    if arg_wo_specifier_lower.startswith('out:'):
                        self.output = arg[len('/out:') :]
                    elif arg_wo_specifier_lower.startswith('wholearchive:'):
                        lib_path = arg[len('/wholearchive:') :]
                        if not is_external_library(lib_path):
                            self.global_libs_and_objects_input.append(lib_path)
                    else:
                        for preserved_option_prefix in msvc_preserved_option_prefixes:
                            if arg_wo_specifier_lower.startswith(preserved_option_prefix):
                                self.preserved_options.append(arg)
                                break
                    # other flags are non-linking related and just ignored
                elif arg[0] == '-':
                    if arg == '-o':
                        if (i + 1) >= len(args):
                            raise Exception('-o flag without an argument')
                        self.output = args[i + 1]
                        i += 1
                    elif arg == '-Wl,--whole-archive':
                        is_in_whole_archive = True
                    elif arg == '-Wl,--no-whole-archive':
                        is_in_whole_archive = False
                    elif arg.startswith('-Wl,-force_load,'):
                        lib_path = arg[len('-Wl,-force_load,') :]
                        if not is_external_library(lib_path):
                            self.global_libs_and_objects_input.append(lib_path)
                    elif arg == '-isysroot':
                        i += 1
                    # other flags are non-linking related and just ignored
                elif arg[0] == '@':
                    # response file with args
                    with open(arg[1:]) as response_file:
                        parsed_args = shlex.shlex(response_file, posix=False, punctuation_chars=False)
                        parsed_args.whitespace_split = True
                        args_in_response_file = list(arg.strip('"') for arg in parsed_args)
                        process_input(args_in_response_file)
                elif not is_external_library(arg):
                    if is_in_whole_archive or arg.endswith('.o') or arg.endswith('.obj'):
                        self.global_libs_and_objects_input.append(arg)
                    elif arg not in std_libraries_to_exclude_from_input:
                        self.non_global_libs_input.append(arg)
                i += 1

        process_input(other_args[2:])

        if self.output is None:
            raise Exception("No output specified")

        if (len(self.global_libs_and_objects_input) == 0) and (len(self.non_global_libs_input) == 0):
            raise Exception("List of input objects and libraries is empty")


class FilesCombiner(object):
    def __init__(self, opts):
        self.opts = opts

        archiver_tool_path = opts.parsed_args.cmake_ar
        if sys.platform.startswith('darwin'):
            # force LIBTOOL even if CMAKE_AR is defined because 'ar' under Darwin does not contain the necessary options
            arch_type = 'LIBTOOL'
            archiver_tool_path = 'libtool'
        elif opts.is_msvc_compatible_linker:
            arch_type = 'LIB'
        elif re.match(r'^(|.*/)llvm\-ar(\-[\d])?', opts.parsed_args.cmake_ar):
            arch_type = 'LLVM_AR'
        elif re.match(r'^(|.*/)(gcc\-)?ar(\-[\d])?', opts.parsed_args.cmake_ar):
            arch_type = 'GNU_AR'
        else:
            raise Exception('Unsupported arch type for CMAKE_AR={}'.format(opts.parsed_args.cmake_ar))

        self.archiving_cmd_prefix = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'link_lib.py'),
            archiver_tool_path,
            arch_type,
            'gnu',  # llvm_ar_format, used only if arch_type == 'LLVM_AR'
            opts.parsed_args.project_binary_dir,
            'None',  # plugin. Unused for now
            '--',  # for plugins. Unused for now
            '--',
        ]
        # the remaining archiving cmd args are [output, .. input .. ]

    def do(self, output, input_list):
        input_file_path = None
        try:
            if self.opts.is_msvc_compatible_linker:
                # use response file for input (because of Windows cmdline length limitations)

                # can't use NamedTemporaryFile because of permissions issues on Windows
                input_file_fd, input_file_path = tempfile.mkstemp()
                try:
                    input_file = os.fdopen(input_file_fd, 'w')
                    for input in input_list:
                        if ' ' in input:
                            input_file.write('"{}" '.format(input))
                        else:
                            input_file.write('{} '.format(input))
                    input_file.flush()
                finally:
                    os.close(input_file_fd)
                input_args = ['@' + input_file_path]
            else:
                input_args = input_list

            cmd = self.archiving_cmd_prefix + [output] + self.opts.preserved_options + input_args
            subprocess.check_call(cmd)
        finally:
            if input_file_path is not None:
                os.remove(input_file_path)

        if not self.opts.is_msvc_compatible_linker:
            subprocess.check_call([self.opts.parsed_args.cmake_ranlib, output])


if __name__ == "__main__":
    opts = Opts(sys.argv[1:])

    output_prefix, output_ext = os.path.splitext(opts.output)
    globals_output = output_prefix + opts.parsed_args.global_part_suffix + output_ext

    if os.path.exists(globals_output):
        os.remove(globals_output)
    if os.path.exists(opts.output):
        os.remove(opts.output)

    files_combiner = FilesCombiner(opts)

    if len(opts.global_libs_and_objects_input) > 0:
        files_combiner.do(globals_output, opts.global_libs_and_objects_input)

    if len(opts.non_global_libs_input) > 0:
        files_combiner.do(opts.output, opts.non_global_libs_input)
