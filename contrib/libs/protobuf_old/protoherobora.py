#!/usr/bin/env python3

import subprocess
import os
import sys
import json


# TODO: dedup
def is_cmdfile_arg(arg):
    # type: (str) -> bool
    return arg.startswith('@')


def cmdfile_path(arg):
    # type: (str) -> str
    return arg[1:]


def read_from_command_file(arg):
    # type: (str) -> list[str]
    with open(arg) as afile:
        return afile.read().splitlines()


def skip_markers(args):
    # type: (list[str]) -> list[str]
    res = []
    for arg in args:
        if arg == '--ya-start-command-file' or arg == '--ya-end-command-file':
            continue
        res.append(arg)
    return res


def iter_args(
        args,  # type: list[str]
        ):
    for arg in args:
        if not is_cmdfile_arg(arg):
            if arg == '--ya-start-command-file' or arg == '--ya-end-command-file':
                continue
            yield arg
        else:
            for cmdfile_arg in read_from_command_file(cmdfile_path(arg)):
                yield cmdfile_arg


def get_args(args):
    # type: (list[str]) -> list[str]
    return list(iter_args(args))
# end TODO


def run(*args):
    return subprocess.check_output(list(args), shell=False).strip()


def gen_renames_1(d):
    for l in d.split('\n'):
        l = l.strip()

        if ' ' in l:
            yield l.split(' ')[-1]


def gen_renames_2(p, d):
    for s in gen_renames_1(d):
        """
        Since clang-17, the option -fsanitize-address-globals-dead-stripping
        has been enabled by default. Due to this, we have broken optimization
        that merges calls to the `asan.module_ctor` function, as we are renaming
        symbols with a prefix of 'py2_'. When this flag is enabled, and
        the functions are not merged, false-positive ODR (One Definition Rule)
        violations occur on objects such as `typeinfo std::exception`, because
        the runtime is trying to handle global objects that have already been handled.
        """
        if 'asan_globals' in s:
            continue
        yield s + ' ' + p + s


def gen_renames(p, d):
    return '\n'.join(gen_renames_2(p, d)).strip() + '\n'


def rename_syms(where, ret, libs):
    p = 'py2_'

    # join libs
    run(where + 'llvm-ar', 'qcL', ret, *libs)

    # find symbols to rename
    syms = run(where + 'llvm-nm', '--extern-only', '--defined-only', '-A', ret)

    # prepare rename plan
    renames = gen_renames(p, syms)

    with open('syms', 'w') as f:
        f.write(renames)

    # rename symbols
    run(where + 'llvm-objcopy', '--redefine-syms=syms', ret)

    # back-rename some symbols
    args = [
        where + 'llvm-objcopy',
        '--redefine-sym',
        p + 'init_api_implementation=init6google8protobuf8internal19_api_implementation',
        '--redefine-sym',
        p + 'init_message=init6google8protobuf5pyext8_message',
        '--redefine-sym',
        p + 'init6google8protobuf8internal19_api_implementation=init6google8protobuf8internal19_api_implementation',
        '--redefine-sym',
        p + 'init6google8protobuf5pyext8_message=init6google8protobuf5pyext8_message',
        '--redefine-sym',
        p + '_init6google8protobuf8internal19_api_implementation=_init6google8protobuf8internal19_api_implementation',
        '--redefine-sym',
        p + '_init6google8protobuf5pyext8_message=_init6google8protobuf5pyext8_message',
        ret,
    ]

    run(*args)
    return ret


def find_lld(args):
    for x in args:
        if 'lld-link' in x:
            return x

    raise IndexError()


def find_clang(args):
    for x in args:
        if 'clang++' in x:
            return x

    raise IndexError()


def fix_py2(cmd, have_comand_files=False, prefix='lib', suffix='a'):
    args = cmd

    if have_comand_files:
        args = get_args(cmd)

    if 'protobuf_old' not in str(args):
        return cmd

    py2_libs = [prefix + 'contrib-libs-protobuf_old.' + suffix, prefix + 'pypython-protobuf-py2.' + suffix]

    def need_rename(x):
        for v in py2_libs:
            if v in x:
                return True

        return False

    old = []
    lib = []

    try:
        where = os.path.dirname(cmd[cmd.index('--objcopy-exe') + 1]) + '/'
    except ValueError:
        try:
            where = os.path.dirname(find_lld(cmd)) + '/'
        except IndexError:
            where = os.path.dirname(find_clang(cmd)) + '/'

    for x in args:
        if need_rename(x):
            lib.append(x)
        else:
            old.append(x)

    name = rename_syms(where, 'libprotoherobora.' + suffix, lib)

    if not have_comand_files:
        return old + [name]

    for file in cmd:
        if is_cmdfile_arg(file):
            cmd_file_path = cmdfile_path(file)
            args = read_from_command_file(cmd_file_path)
            if not 'protobuf_old' in str(args):
                continue
            with open(cmd_file_path, 'w') as afile:
                for arg in args:
                    if not need_rename(arg):
                        afile.write(arg + '\n')
                afile.write(name)

    return cmd


if __name__ == '__main__':
    args = sys.argv[1:]

    if 'lld-link' in str(args):
        cmd = fix_py2(args, have_comand_files=True, prefix='', suffix='lib')
    else:
        cmd = fix_py2(args)

    sys.stdout.write(json.dumps(cmd))
