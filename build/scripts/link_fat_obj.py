import argparse
import subprocess
import sys

from process_whole_archive_option import ProcessWholeArchiveOption

YA_ARG_PREFIX = '-Ya,'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj')
    parser.add_argument('--globals-lib')
    parser.add_argument('--lib', required=True)
    parser.add_argument('--arch', required=True)
    parser.add_argument('--build-root', default=None)
    parser.add_argument('--with-own-obj', action='store_true', default=False)
    parser.add_argument('--with-global-srcs', action='store_true', default=False)

    groups = {}
    args_list = groups.setdefault('default', [])
    for arg in sys.argv[1:]:
        if arg == '--with-own-obj':
            groups['default'].append(arg)
        elif arg == '--globals-lib':
            groups['default'].append(arg)
        elif arg == '--with-global-srcs':
            groups['default'].append(arg)
        elif arg.startswith(YA_ARG_PREFIX):
            group_name = arg[len(YA_ARG_PREFIX):]
            args_list = groups.setdefault(group_name, [])
        else:
            args_list.append(arg)

    return parser.parse_args(groups['default']), groups


def strip_suppression_files(srcs):
    return [s for s in srcs if not s.endswith('.supp')]


def strip_forceload_prefix(srcs):
    force_load_prefix = '-Wl,-force_load,'
    return list(map(lambda lib: lib[lib.startswith(force_load_prefix) and len(force_load_prefix):], srcs))


def main():
    args, groups = get_args()

    # Inputs
    auto_input = groups['input']

    # Outputs
    lib_output = args.lib
    obj_output = args.obj

    # Dependencies
    global_srcs = groups['global_srcs']
    global_srcs = strip_suppression_files(global_srcs)
    global_srcs = ProcessWholeArchiveOption(args.arch).construct_cmd(global_srcs)
    global_srcs = strip_forceload_prefix(global_srcs)
    peers = groups['peers']

    # Tools
    linker = groups['linker']
    archiver = groups['archiver']

    do_link = linker + ['-o', obj_output, '-Wl,-r', '-nodefaultlibs', '-nostartfiles'] + global_srcs + auto_input
    do_archive = archiver + [lib_output] + peers
    do_globals = None
    if args.globals_lib:
        do_globals = archiver + [args.globals_lib] + auto_input + global_srcs
    if args.with_own_obj:
        do_archive += auto_input
    if args.with_global_srcs:
        do_archive += global_srcs

    def call(c):
        proc = subprocess.Popen(c, shell=False, stderr=sys.stderr, stdout=sys.stdout, cwd=args.build_root)
        proc.communicate()
        return proc.returncode

    if obj_output:
        link_res = call(do_link)
        if link_res:
            sys.exit(link_res)

    if do_globals:
        glob_res = call(do_globals)
        if glob_res:
            sys.exit(glob_res)

    sys.exit(call(do_archive))


if __name__ == '__main__':
    main()
