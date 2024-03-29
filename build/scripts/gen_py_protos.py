import os
from os import path
import shutil
import subprocess
import tempfile
import argparse
import re


OUT_DIR_ARG = '--python_out='


def _noext(fname):
    return fname[: fname.rfind('.')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffixes", nargs="*", default=[])
    parser.add_argument("--input")
    parser.add_argument("--ns")
    parser.add_argument("--py_ver")
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
            out_dir_orig = args[i][len(OUT_DIR_ARG) :]
            out_dir_temp = tempfile.mkdtemp(dir=out_dir_orig)
            args[i] = OUT_DIR_ARG + out_dir_temp
            continue

        match = re.match(r"^(--(\w+)_out=).*", args[i])
        if match:
            plugin_out_dir_arg = match.group(1)
            plugin = match.group(2)
            assert plugin not in plugin_out_dirs_orig, 'Duplicate "{0}" param'.format(plugin_out_dir_arg)
            plugin_out_dirs_orig[plugin] = args[i][len(plugin_out_dir_arg) :]
            assert (
                plugin_out_dirs_orig[plugin] == out_dir_orig
            ), 'Params "{0}" and "{1}" expected to have the same value'.format(OUT_DIR_ARG, plugin_out_dir_arg)
            args[i] = plugin_out_dir_arg + out_dir_temp

    assert out_dir_temp, 'Param "{0}" not found'.format(OUT_DIR_ARG)

    retcode = subprocess.call(args)
    assert not retcode, 'Protoc failed for command {}'.format(' '.join(args))

    temp_name = out_dir_temp
    orig_name = out_dir_orig
    dir_name, file_name = path.split(script_args.input[len(script_args.ns) - 1 :])
    for part in dir_name.split('/'):
        temp_part = part.replace('-', '_')
        temp_name = path.join(temp_name, temp_part)
        assert path.exists(temp_name)

        orig_name = path.join(orig_name, part)
        if not path.exists(orig_name):
            os.mkdir(orig_name)

    orig_base_name = _noext(file_name)
    temp_base_name = orig_base_name.replace('-', '_')
    for suf in script_args.suffixes:
        temp_file_name = path.join(temp_name, temp_base_name + suf)
        assert path.exists(temp_file_name)

        orig_file_name = path.join(orig_name, orig_base_name + '__int{}__'.format(script_args.py_ver) + suf)
        os.rename(temp_file_name, orig_file_name)

    shutil.rmtree(out_dir_temp)


if __name__ == '__main__':
    main()
