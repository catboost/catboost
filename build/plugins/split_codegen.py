from _common import sort_by_keywords

# This hard-coded many times in CppParts in various codegens
_DEFAULT_CPP_PARTS = 20
# See TCodegenParams::MethodStream usage in factor codegen
_ADDITIONAL_STREAM_COUNT = 5


def onsplit_codegen(unit, *args):
    '''
        @usage: SPLIT_CODEGEN(tool prefix opts... [OUT_NUM num] [OUTPUT_INCLUDES output_includes...])

        Generator of a certain number of parts of the .cpp file + one header .h file from .in

        Supports keywords:
        1. OUT_NUM <the number of generated Prefix.N.cpp default 25 (N varies from 0 to 24)>
        2. OUTPUT_INCLUDES <path to files that will be included in generalnyj of macro files>
    '''
    keywords = {"OUT_NUM": 1}
    flat_args, spec_args = sort_by_keywords(keywords, args)

    num_outputs = _DEFAULT_CPP_PARTS + _ADDITIONAL_STREAM_COUNT
    if "OUT_NUM" in spec_args:
        num_outputs = int(spec_args["OUT_NUM"][0])

    tool = flat_args[0]
    prefix = flat_args[1]

    cmd = [tool, prefix, 'OUT']
    for num in range(num_outputs):
        cmd.append('{}.{}.cpp'.format(prefix, num))

    cpp_parts = int(num_outputs) - _ADDITIONAL_STREAM_COUNT
    cpp_parts_args = ['--cpp-parts', str(cpp_parts)]

    if len(flat_args) > 2:
        if flat_args[2] != 'OUTPUT_INCLUDES':
            cmd.append('OPTS')
        cmd += cpp_parts_args + flat_args[2:]
    else:
        cmd += ['OPTS'] + cpp_parts_args

    unit.on_split_codegen_base(cmd)
