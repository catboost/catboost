from _common import sort_by_keywords

ANTLR4_JAR_PATH = 'antlr-4.7-complete.jar'


def onrun_antlr4(unit, *args):
    if len(args) < 1:
        raise Exception("Not enough arguments for RUN_ANTLR4 macro")

    unit.onpeerdir(['build/external_resources/antlr4'])

    arg_list = ['-jar', '$ANTLR4_RESOURCE_GLOBAL/' + ANTLR4_JAR_PATH]
    arg_list += list(args)

    unit.on_run_java(arg_list)


def onrun_antlr4_cpp(unit, *args):
    if len(args) < 1:
        raise Exception("Not enough arguments for RUN_ANTLR4_CPP macro")

    grammar_file = args[0]
    arg_list = [ grammar_file, '-o', '${BINDIR}' ]
    arg_list += args[1:]

    grammar = '${noext:"' + grammar_file + '"}'
    arg_list += [ 'IN', grammar_file ]
    arg_list += [ 'OUT', grammar + 'Lexer.cpp', grammar + 'Parser.cpp' ]
    arg_list += [ 'OUT', grammar + 'Lexer.h', grammar + 'Parser.h' ]
    if '-no-listener' not in args:
        arg_list += [ 'OUT', grammar + 'Listener.h', grammar + 'BaseListener.h' ]
    if '-visitor' in args:
        arg_list += [ 'OUT', grammar + 'Visitor.h', grammar + 'BaseVisitor.h' ]
    arg_list += [ 'CWD', '${BINDIR}' ]
    arg_list += [ 'OUTPUT_INCLUDES', '${ARCADIA_ROOT}/contrib/libs/antlr4_cpp_runtime/src/antlr4-runtime.h' ]
    unit.onrun_antlr4(arg_list)
