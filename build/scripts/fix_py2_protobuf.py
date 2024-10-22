import subprocess
import os

import process_command_files as pcf


def run(*args):
    # print >>sys.stderr, args
    return subprocess.check_output(list(args), shell=False).strip()


def gen_renames_1(d):
    for l in d.split('\n'):
        l = l.strip()

        if ' ' in l:
            yield l.split(' ')[-1]


def gen_renames_2(p, d):
    for s in gen_renames_1(d):
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


def fix_py2(cmd, have_comand_files=False, prefix='lib', suffix='a'):
    args = cmd
    if have_comand_files:
        args = pcf.get_args(cmd)
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

    where = os.path.dirname(cmd[0]) + '/'

    for x in args:
        if need_rename(x):
            lib.append(x)
        else:
            old.append(x)

    name = rename_syms(where, 'libprotoherobora.' + suffix, lib)

    if not have_comand_files:
        return old + [name]

    for file in cmd:
        if pcf.is_cmdfile_arg(file):
            cmd_file_path = pcf.cmdfile_path(file)
            args = pcf.read_from_command_file(cmd_file_path)
            if not 'protobuf_old' in str(args):
                continue
            with open(cmd_file_path, 'w') as afile:
                for arg in args:
                    if not need_rename(arg):
                        afile.write(arg + '\n')
                afile.write(name)

    return cmd
