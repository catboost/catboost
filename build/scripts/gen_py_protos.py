import os
from os import path
import shutil
import subprocess
import sys
import tempfile
import argparse
import re


OUT_DIR_ARG = '--python_out='

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffixes", nargs="*", default=[])
    parser.add_argument("protoc_args", nargs=argparse.REMAINDER)
    script_args = parser.parse_args()

    args = script_args.protoc_args

    if args[0] == "--":
        args = args[1:]

    out_dir_orig = None
    out_dir_temp = None
    plugin_out_dirs_orig = {}
    for i in range(len(args)):
        if args[i].startswith(OUT_DIR_ARG):
            assert not out_dir_orig, 'Duplicate "{0}" param'.format(OUT_DIR_ARG)
            out_dir_orig = args[i][len(OUT_DIR_ARG):]
            out_dir_temp = tempfile.mkdtemp(dir=out_dir_orig)
            args[i] = OUT_DIR_ARG + out_dir_temp
            continue

        match = re.match(r"^(--(\w+)_out=).*", args[i])
        if match:
            plugin_out_dir_arg = match.group(1)
            plugin = match.group(2)
            assert plugin not in plugin_out_dirs_orig, 'Duplicate "{0}" param'.format(plugin_out_dir_arg)
            plugin_out_dirs_orig[plugin] = args[i][len(plugin_out_dir_arg):]
            assert plugin_out_dirs_orig[plugin] == out_dir_orig, 'Params "{0}" and "{1}" expected to have the same value'.format(OUT_DIR_ARG, plugin_out_dir_arg)
            args[i] = plugin_out_dir_arg + out_dir_temp

    assert out_dir_temp, 'Param "{0}" not found'.format(OUT_DIR_ARG)

    retcode = subprocess.call(args)
    assert not retcode, 'Protoc failed for command {}'.format(' '.join(args))

    for root_temp, dirs, files in os.walk(out_dir_temp):
        sub_dir = path.relpath(root_temp, out_dir_temp)
        root_orig = path.join(out_dir_orig, sub_dir)
        for d in dirs:
            d_orig = path.join(root_orig, d)
            if not path.exists(d_orig):
                os.mkdir(d_orig)
        for f in files:
            f_orig = f
            for suf in script_args.suffixes:
                if f.endswith(suf):
                    f_orig = f[:-len(suf)] + "__int__" + suf
                    break
            os.rename(path.join(root_temp, f), path.join(root_orig, f_orig))
    shutil.rmtree(out_dir_temp)


if __name__ == '__main__':
    main()
