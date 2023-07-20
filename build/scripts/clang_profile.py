import fnmatch
import shutil
import optparse
import os

import process_command_files as pcf

def copy_clang_rt_profile(cmd, build_root, arch):
    profile_path = None
    resource_dir_path = None

    for flag in cmd:
        if fnmatch.fnmatch(flag, 'contrib/libs/clang14-rt/lib/profile/libclang_rt.profile-*.a'):
            profile_path = flag
        if flag.startswith('-resource-dir='):
            resource_dir_path = flag[len('-resource-dir='):]

    lib_profile = os.path.join(build_root, profile_path)
    profile_name = os.path.basename(lib_profile)

    if arch == 'DARWIN':
        dst_dir = os.path.join(build_root, resource_dir_path, 'lib/darwin')

    if arch == 'LINUX':
        dst_dir = os.path.join(build_root, resource_dir_path, 'lib/linux')

    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(lib_profile, os.path.join(dst_dir, profile_name))


def parse_args():
    parser = optparse.OptionParser()
    parser.disable_interspersed_args()
    parser.add_option('--build-root')
    parser.add_option('--arch')
    parser.add_option('--need-profile-runtime')
    return parser.parse_args()


if __name__ == '__main__':
    opts, args = parse_args()
    args = pcf.skip_markers(args)

    if opts.need_profile_runtime != "no":
        copy_clang_rt_profile(args, opts.build_root, opts.arch)
