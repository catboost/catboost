"""
    pygments.lexers.julia
    ~~~~~~~~~~~~~~~~~~~~~

    Lexers for the Julia language.

    :copyright: Copyright 2006-2021 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import re

from pygments.lexer import Lexer, RegexLexer, bygroups, do_insertions, \
    words, include
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
    Number, Punctuation, Generic
from pygments.util import shebang_matches

__all__ = ['JuliaLexer', 'JuliaConsoleLexer']

allowed_variable = \
    '(?:[a-zA-Z_\u00A1-\U0010ffff]|%s)(?:[a-zA-Z_0-9\u00A1-\U0010ffff])*!*'


class JuliaLexer(RegexLexer):
    """
    For `Julia <http://julialang.org/>`_ source code.

    .. versionadded:: 1.6
    """

    name = 'Julia'
    aliases = ['julia', 'jl']
    filenames = ['*.jl']
    mimetypes = ['text/x-julia', 'application/x-julia']

    flags = re.MULTILINE | re.UNICODE

    tokens = {
        'root': [
            (r'\n', Text),
            (r'[^\S\n]+', Text),
            (r'#=', Comment.Multiline, "blockcomment"),
            (r'#.*$', Comment),
            (r'[\[\]{}(),;]', Punctuation),

            # keywords
            (r'in\b', Keyword.Pseudo),
            (r'isa\b', Keyword.Pseudo),
            (r'(true|false)\b', Keyword.Constant),
            (r'(local|global|const)\b', Keyword.Declaration),
            (words([
                'function', 'type', 'typealias', 'abstract', 'immutable',
                'baremodule', 'begin', 'bitstype', 'break', 'catch', 'ccall',
                'continue', 'do', 'else', 'elseif', 'end', 'export', 'finally',
                'for', 'if', 'import', 'importall', 'let', 'macro', 'module',
                'mutable', 'primitive', 'quote', 'return', 'struct', 'try',
                'using', 'while'],
                suffix=r'\b'), Keyword),

            # NOTE
            # Patterns below work only for definition sites and thus hardly reliable.
            #
            # functions
            # (r'(function)(\s+)(' + allowed_variable + ')',
            #  bygroups(Keyword, Text, Name.Function)),
            #
            # types
            # (r'(type|typealias|abstract|immutable)(\s+)(' + allowed_variable + ')',
            #  bygroups(Keyword, Text, Name.Class)),

            # type names
            (words([
                'ANY', 'ASCIIString', 'AbstractArray', 'AbstractChannel',
                'AbstractFloat', 'AbstractMatrix', 'AbstractRNG',
                'AbstractSparseArray', 'AbstractSparseMatrix',
                'AbstractSparseVector', 'AbstractString', 'AbstractVecOrMat',
                'AbstractVector', 'Any', 'ArgumentError', 'Array',
                'AssertionError', 'Associative', 'Base64DecodePipe',
                'Base64EncodePipe', 'Bidiagonal', 'BigFloat', 'BigInt',
                'BitArray', 'BitMatrix', 'BitVector', 'Bool', 'BoundsError',
                'Box', 'BufferStream', 'CapturedException', 'CartesianIndex',
                'CartesianRange', 'Cchar', 'Cdouble', 'Cfloat', 'Channel',
                'Char', 'Cint', 'Cintmax_t', 'Clong', 'Clonglong',
                'ClusterManager', 'Cmd', 'Coff_t', 'Colon', 'Complex',
                'Complex128', 'Complex32', 'Complex64', 'CompositeException',
                'Condition', 'Cptrdiff_t', 'Cshort', 'Csize_t', 'Cssize_t',
                'Cstring', 'Cuchar', 'Cuint', 'Cuintmax_t', 'Culong',
                'Culonglong', 'Cushort', 'Cwchar_t', 'Cwstring', 'DataType',
                'Date', 'DateTime', 'DenseArray', 'DenseMatrix',
                'DenseVecOrMat', 'DenseVector', 'Diagonal', 'Dict',
                'DimensionMismatch', 'Dims', 'DirectIndexString', 'Display',
                'DivideError', 'DomainError', 'EOFError', 'EachLine', 'Enum',
                'Enumerate', 'ErrorException', 'Exception', 'Expr',
                'Factorization', 'FileMonitor', 'FileOffset', 'Filter',
                'Float16', 'Float32', 'Float64', 'FloatRange', 'Function',
                'GenSym', 'GlobalRef', 'GotoNode', 'HTML', 'Hermitian', 'IO',
                'IOBuffer', 'IOStream', 'IPv4', 'IPv6', 'InexactError',
                'InitError', 'Int', 'Int128', 'Int16', 'Int32', 'Int64', 'Int8',
                'IntSet', 'Integer', 'InterruptException', 'IntrinsicFunction',
                'InvalidStateException', 'Irrational', 'KeyError', 'LabelNode',
                'LambdaStaticData', 'LinSpace', 'LineNumberNode', 'LoadError',
                'LocalProcess', 'LowerTriangular', 'MIME', 'Matrix',
                'MersenneTwister', 'Method', 'MethodError', 'MethodTable',
                'Module', 'NTuple', 'NewvarNode', 'NullException', 'Nullable',
                'Number', 'ObjectIdDict', 'OrdinalRange', 'OutOfMemoryError',
                'OverflowError', 'Pair', 'ParseError', 'PartialQuickSort',
                'Pipe', 'PollingFileWatcher', 'ProcessExitedException',
                'ProcessGroup', 'Ptr', 'QuoteNode', 'RandomDevice', 'Range',
                'Rational', 'RawFD', 'ReadOnlyMemoryError', 'Real',
                'ReentrantLock', 'Ref', 'Regex', 'RegexMatch',
                'RemoteException', 'RemoteRef', 'RepString', 'RevString',
                'RopeString', 'RoundingMode', 'SegmentationFault',
                'SerializationState', 'Set', 'SharedArray', 'SharedMatrix',
                'SharedVector', 'Signed', 'SimpleVector', 'SparseMatrixCSC',
                'StackOverflowError', 'StatStruct', 'StepRange', 'StridedArray',
                'StridedMatrix', 'StridedVecOrMat', 'StridedVector', 'SubArray',
                'SubString', 'SymTridiagonal', 'Symbol', 'SymbolNode',
                'Symmetric', 'SystemError', 'TCPSocket', 'Task', 'Text',
                'TextDisplay', 'Timer', 'TopNode', 'Tridiagonal', 'Tuple',
                'Type', 'TypeConstructor', 'TypeError', 'TypeName', 'TypeVar',
                'UDPSocket', 'UInt', 'UInt128', 'UInt16', 'UInt32', 'UInt64',
                'UInt8', 'UTF16String', 'UTF32String', 'UTF8String',
                'UndefRefError', 'UndefVarError', 'UnicodeError', 'UniformScaling',
                'Union', 'UnitRange', 'Unsigned', 'UpperTriangular', 'Val',
                'Vararg', 'VecOrMat', 'Vector', 'VersionNumber', 'Void', 'WString',
                'WeakKeyDict', 'WeakRef', 'WorkerConfig', 'Zip'], suffix=r'\b'),
                Keyword.Type),

            # builtins
            (words([
                'ARGS', 'CPU_CORES', 'C_NULL', 'DevNull', 'ENDIAN_BOM',
                'ENV', 'I', 'Inf', 'Inf16', 'Inf32', 'Inf64',
                'InsertionSort', 'JULIA_HOME', 'LOAD_PATH', 'MergeSort',
                'NaN', 'NaN16', 'NaN32', 'NaN64', 'OS_NAME',
                'QuickSort', 'RoundDown', 'RoundFromZero', 'RoundNearest',
                'RoundNearestTiesAway', 'RoundNearestTiesUp',
                'RoundToZero', 'RoundUp', 'STDERR', 'STDIN', 'STDOUT',
                'VERSION', 'WORD_SIZE', 'catalan', 'e', 'eu',
                'eulergamma', 'golden', 'im', 'nothing', 'pi', 'γ', 'π', 'φ'],
                suffix=r'\b'), Name.Builtin),

            # operators
            # see: https://github.com/JuliaLang/julia/blob/master/src/julia-parser.scm
            (words((
                # prec-assignment
                '=', ':=', '+=', '-=', '*=', '/=', '//=', './/=', '.*=', './=',
                '\\=', '.\\=', '^=', '.^=', '÷=', '.÷=', '%=', '.%=', '|=', '&=',
                '$=', '=>', '<<=', '>>=', '>>>=', '~', '.+=', '.-=',
                # prec-conditional
                '?',
                # prec-arrow
                '--', '-->',
                # prec-lazy-or
                '||',
                # prec-lazy-and
                '&&',
                # prec-comparison
                '>', '<', '>=', '≥', '<=', '≤', '==', '===', '≡', '!=', '≠',
                '!==', '≢', '.>', '.<', '.>=', '.≥', '.<=', '.≤', '.==', '.!=',
                '.≠', '.=', '.!', '<:', '>:', '∈', '∉', '∋', '∌', '⊆',
                '⊈', '⊂',
                '⊄', '⊊',
                # prec-pipe
                '|>', '<|',
                # prec-colon
                ':',
                # prec-plus
                '.+', '.-', '|', '∪', '$',
                # prec-bitshift
                '<<', '>>', '>>>', '.<<', '.>>', '.>>>',
                # prec-times
                '*', '/', './', '÷', '.÷', '%', '⋅', '.%', '.*', '\\', '.\\', '&', '∩',
                # prec-rational
                '//', './/',
                # prec-power
                '^', '.^',
                # prec-decl
                '::',
                # prec-dot
                '.',
                # unary op
                '+', '-', '!', '√', '∛', '∜',
            )), Operator),

            # chars
            (r"'(\\.|\\[0-7]{1,3}|\\x[a-fA-F0-9]{1,3}|\\u[a-fA-F0-9]{1,4}|"
             r"\\U[a-fA-F0-9]{1,6}|[^\\\'\n])'", String.Char),

            # try to match trailing transpose
            (r'(?<=[.\w)\]])\'+', Operator),

            # strings
            (r'"""', String, 'tqstring'),
            (r'"', String, 'string'),

            # regular expressions
            (r'r"""', String.Regex, 'tqregex'),
            (r'r"', String.Regex, 'regex'),

            # backticks
            (r'`', String.Backtick, 'command'),

            # names
            (allowed_variable, Name),
            (r'@' + allowed_variable, Name.Decorator),

            # numbers
            (r'(\d+(_\d+)+\.\d*|\d*\.\d+(_\d+)+)([eEf][+-]?[0-9]+)?', Number.Float),
            (r'(\d+\.\d*|\d*\.\d+)([eEf][+-]?[0-9]+)?', Number.Float),
            (r'\d+(_\d+)+[eEf][+-]?[0-9]+', Number.Float),
            (r'\d+[eEf][+-]?[0-9]+', Number.Float),
            (r'0b[01]+(_[01]+)+', Number.Bin),
            (r'0b[01]+', Number.Bin),
            (r'0o[0-7]+(_[0-7]+)+', Number.Oct),
            (r'0o[0-7]+', Number.Oct),
            (r'0x[a-fA-F0-9]+(_[a-fA-F0-9]+)+', Number.Hex),
            (r'0x[a-fA-F0-9]+', Number.Hex),
            (r'\d+(_\d+)+', Number.Integer),
            (r'\d+', Number.Integer)
        ],

        "blockcomment": [
            (r'[^=#]', Comment.Multiline),
            (r'#=', Comment.Multiline, '#push'),
            (r'=#', Comment.Multiline, '#pop'),
            (r'[=#]', Comment.Multiline),
        ],

        'string': [
            (r'"', String, '#pop'),
            # FIXME: This escape pattern is not perfect.
            (r'\\([\\"\'$nrbtfav]|(x|u|U)[a-fA-F0-9]+|\d+)', String.Escape),
            # Interpolation is defined as "$" followed by the shortest full
            # expression, which is something we can't parse.
            # Include the most common cases here: $word, and $(paren'd expr).
            (r'\$' + allowed_variable, String.Interpol),
            # (r'\$[a-zA-Z_]+', String.Interpol),
            (r'(\$)(\()', bygroups(String.Interpol, Punctuation), 'in-intp'),
            # @printf and @sprintf formats
            (r'%[-#0 +]*([0-9]+|[*])?(\.([0-9]+|[*]))?[hlL]?[E-GXc-giorsux%]',
             String.Interpol),
            (r'.|\s', String),
        ],

        'tqstring': [
            (r'"""', String, '#pop'),
            (r'\\([\\"\'$nrbtfav]|(x|u|U)[a-fA-F0-9]+|\d+)', String.Escape),
            (r'\$' + allowed_variable, String.Interpol),
            (r'(\$)(\()', bygroups(String.Interpol, Punctuation), 'in-intp'),
            (r'.|\s', String),
        ],

        'regex': [
            (r'"', String.Regex, '#pop'),
            (r'\\"', String.Regex),
            (r'.|\s', String.Regex),
        ],

        'tqregex': [
            (r'"""', String.Regex, '#pop'),
            (r'.|\s', String.Regex),
        ],

        'command': [
            (r'`', String.Backtick, '#pop'),
            (r'\$' + allowed_variable, String.Interpol),
            (r'(\$)(\()', bygroups(String.Interpol, Punctuation), 'in-intp'),
            (r'.|\s', String.Backtick)
        ],

        'in-intp': [
            (r'\(', Punctuation, '#push'),
            (r'\)', Punctuation, '#pop'),
            include('root'),
        ]
    }

    def analyse_text(text):
        return shebang_matches(text, r'julia')


class JuliaConsoleLexer(Lexer):
    """
    For Julia console sessions. Modeled after MatlabSessionLexer.

    .. versionadded:: 1.6
    """
    name = 'Julia console'
    aliases = ['jlcon']

    def get_tokens_unprocessed(self, text):
        jllexer = JuliaLexer(**self.options)
        start = 0
        curcode = ''
        insertions = []
        output = False
        error = False

        for line in text.splitlines(True):
            if line.startswith('julia>'):
                insertions.append((len(curcode), [(0, Generic.Prompt, line[:6])]))
                curcode += line[6:]
                output = False
                error = False
            elif line.startswith('help?>') or line.startswith('shell>'):
                yield start, Generic.Prompt, line[:6]
                yield start + 6, Text, line[6:]
                output = False
                error = False
            elif line.startswith('      ') and not output:
                insertions.append((len(curcode), [(0, Text, line[:6])]))
                curcode += line[6:]
            else:
                if curcode:
                    yield from do_insertions(
                        insertions, jllexer.get_tokens_unprocessed(curcode))
                    curcode = ''
                    insertions = []
                if line.startswith('ERROR: ') or error:
                    yield start, Generic.Error, line
                    error = True
                else:
                    yield start, Generic.Output, line
                output = True
            start += len(line)

        if curcode:
            yield from do_insertions(
                insertions, jllexer.get_tokens_unprocessed(curcode))
