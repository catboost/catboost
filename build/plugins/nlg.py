import os

from _common import rootrel_arc_src

EXTENSION = '.nlg'
MACRO_NAME = 'COMPILE_NLG'


class ResolveImportError(Exception):
    pass


class CompileNlgError(Exception):
    pass


def output_paths(input_paths):
    result = ['register.cpp', 'register.h']
    for input_path in input_paths:
        result += [input_path + '.cpp', input_path + '.h']
    return result


def oncompile_nlg(unit, *input_paths):
    unit.onpeerdir(['alice/nlg/runtime'])

    # tool invocation
    rp_args = ['alice/nlg/bin', 'compile-cpp']
    rp_args += ['--import-dir', '${CURDIR}']
    rp_args += ['--out-dir', '${BINDIR}']
    rp_args += ['--include-prefix', rootrel_arc_src(unit.path(), unit)]

    rp_args += (
        os.path.join('${CURDIR}', input_path)
        for input_path in input_paths
    )

    # input nodes (.nlg files)
    rp_args += ['IN']
    rp_args += input_paths

    # output nodes (.h/.cpp files)
    rp_args += ['OUT']
    rp_args += output_paths(input_paths)

    unit.onrun_program(rp_args)
