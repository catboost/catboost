# -*- coding: utf-8 -*-
"""
    pygments.lexers.asm
    ~~~~~~~~~~~~~~~~~~~

    Lexers for assembly languages.

    :copyright: Copyright 2006-2019 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import re

from pygments.lexer import RegexLexer, include, bygroups, using, words, \
    DelegatingLexer
from pygments.lexers.c_cpp import CppLexer, CLexer
from pygments.lexers.d import DLexer
from pygments.token import Text, Name, Number, String, Comment, Punctuation, \
    Other, Keyword, Operator

__all__ = ['GasLexer', 'ObjdumpLexer', 'DObjdumpLexer', 'CppObjdumpLexer',
           'CObjdumpLexer', 'HsailLexer', 'LlvmLexer', 'NasmLexer',
           'NasmObjdumpLexer', 'TasmLexer', 'Ca65Lexer', 'Dasm16Lexer']


class GasLexer(RegexLexer):
    """
    For Gas (AT&T) assembly code.
    """
    name = 'GAS'
    aliases = ['gas', 'asm']
    filenames = ['*.s', '*.S']
    mimetypes = ['text/x-gas']

    #: optional Comment or Whitespace
    string = r'"(\\"|[^"])*"'
    char = r'[\w$.@-]'
    identifier = r'(?:[a-zA-Z$_]' + char + r'*|\.' + char + '+)'
    number = r'(?:0[xX][a-zA-Z0-9]+|\d+)'
    register = '%' + identifier

    tokens = {
        'root': [
            include('whitespace'),
            (identifier + ':', Name.Label),
            (r'\.' + identifier, Name.Attribute, 'directive-args'),
            (r'lock|rep(n?z)?|data\d+', Name.Attribute),
            (identifier, Name.Function, 'instruction-args'),
            (r'[\r\n]+', Text)
        ],
        'directive-args': [
            (identifier, Name.Constant),
            (string, String),
            ('@' + identifier, Name.Attribute),
            (number, Number.Integer),
            (register, Name.Variable),
            (r'[\r\n]+', Text, '#pop'),
            (r'[;#].*?\n', Comment, '#pop'),

            include('punctuation'),
            include('whitespace')
        ],
        'instruction-args': [
            # For objdump-disassembled code, shouldn't occur in
            # actual assembler input
            ('([a-z0-9]+)( )(<)('+identifier+')(>)',
                bygroups(Number.Hex, Text, Punctuation, Name.Constant,
                         Punctuation)),
            ('([a-z0-9]+)( )(<)('+identifier+')([-+])('+number+')(>)',
                bygroups(Number.Hex, Text, Punctuation, Name.Constant,
                         Punctuation, Number.Integer, Punctuation)),

            # Address constants
            (identifier, Name.Constant),
            (number, Number.Integer),
            # Registers
            (register, Name.Variable),
            # Numeric constants
            ('$'+number, Number.Integer),
            (r"$'(.|\\')'", String.Char),
            (r'[\r\n]+', Text, '#pop'),
            (r'[;#].*?\n', Comment, '#pop'),

            include('punctuation'),
            include('whitespace')
        ],
        'whitespace': [
            (r'\n', Text),
            (r'\s+', Text),
            (r'[;#].*?\n', Comment)
        ],
        'punctuation': [
            (r'[-*,.()\[\]!:]+', Punctuation)
        ]
    }

    def analyse_text(text):
        if re.match(r'^\.(text|data|section)', text, re.M):
            return True
        elif re.match(r'^\.\w+', text, re.M):
            return 0.1


def _objdump_lexer_tokens(asm_lexer):
    """
    Common objdump lexer tokens to wrap an ASM lexer.
    """
    hex_re = r'[0-9A-Za-z]'
    return {
        'root': [
            # File name & format:
            ('(.*?)(:)( +file format )(.*?)$',
                bygroups(Name.Label, Punctuation, Text, String)),
            # Section header
            ('(Disassembly of section )(.*?)(:)$',
                bygroups(Text, Name.Label, Punctuation)),
            # Function labels
            # (With offset)
            ('('+hex_re+'+)( )(<)(.*?)([-+])(0[xX][A-Za-z0-9]+)(>:)$',
                bygroups(Number.Hex, Text, Punctuation, Name.Function,
                         Punctuation, Number.Hex, Punctuation)),
            # (Without offset)
            ('('+hex_re+'+)( )(<)(.*?)(>:)$',
                bygroups(Number.Hex, Text, Punctuation, Name.Function,
                         Punctuation)),
            # Code line with disassembled instructions
            ('( *)('+hex_re+r'+:)(\t)((?:'+hex_re+hex_re+' )+)( *\t)([a-zA-Z].*?)$',
                bygroups(Text, Name.Label, Text, Number.Hex, Text,
                         using(asm_lexer))),
            # Code line with ascii
            ('( *)('+hex_re+r'+:)(\t)((?:'+hex_re+hex_re+' )+)( *)(.*?)$',
                bygroups(Text, Name.Label, Text, Number.Hex, Text, String)),
            # Continued code line, only raw opcodes without disassembled
            # instruction
            ('( *)('+hex_re+r'+:)(\t)((?:'+hex_re+hex_re+' )+)$',
                bygroups(Text, Name.Label, Text, Number.Hex)),
            # Skipped a few bytes
            (r'\t\.\.\.$', Text),
            # Relocation line
            # (With offset)
            (r'(\t\t\t)('+hex_re+r'+:)( )([^\t]+)(\t)(.*?)([-+])(0x'+hex_re+'+)$',
                bygroups(Text, Name.Label, Text, Name.Property, Text,
                         Name.Constant, Punctuation, Number.Hex)),
            # (Without offset)
            (r'(\t\t\t)('+hex_re+r'+:)( )([^\t]+)(\t)(.*?)$',
                bygroups(Text, Name.Label, Text, Name.Property, Text,
                         Name.Constant)),
            (r'[^\n]+\n', Other)
        ]
    }


class ObjdumpLexer(RegexLexer):
    """
    For the output of 'objdump -dr'
    """
    name = 'objdump'
    aliases = ['objdump']
    filenames = ['*.objdump']
    mimetypes = ['text/x-objdump']

    tokens = _objdump_lexer_tokens(GasLexer)


class DObjdumpLexer(DelegatingLexer):
    """
    For the output of 'objdump -Sr on compiled D files'
    """
    name = 'd-objdump'
    aliases = ['d-objdump']
    filenames = ['*.d-objdump']
    mimetypes = ['text/x-d-objdump']

    def __init__(self, **options):
        super(DObjdumpLexer, self).__init__(DLexer, ObjdumpLexer, **options)


class CppObjdumpLexer(DelegatingLexer):
    """
    For the output of 'objdump -Sr on compiled C++ files'
    """
    name = 'cpp-objdump'
    aliases = ['cpp-objdump', 'c++-objdumb', 'cxx-objdump']
    filenames = ['*.cpp-objdump', '*.c++-objdump', '*.cxx-objdump']
    mimetypes = ['text/x-cpp-objdump']

    def __init__(self, **options):
        super(CppObjdumpLexer, self).__init__(CppLexer, ObjdumpLexer, **options)


class CObjdumpLexer(DelegatingLexer):
    """
    For the output of 'objdump -Sr on compiled C files'
    """
    name = 'c-objdump'
    aliases = ['c-objdump']
    filenames = ['*.c-objdump']
    mimetypes = ['text/x-c-objdump']

    def __init__(self, **options):
        super(CObjdumpLexer, self).__init__(CLexer, ObjdumpLexer, **options)


class HsailLexer(RegexLexer):
    """
    For HSAIL assembly code.

    .. versionadded:: 2.2
    """
    name = 'HSAIL'
    aliases = ['hsail', 'hsa']
    filenames = ['*.hsail']
    mimetypes = ['text/x-hsail']

    string = r'"[^"]*?"'
    identifier = r'[a-zA-Z_][\w.]*'
    # Registers
    register_number = r'[0-9]+'
    register = r'(\$(c|s|d|q)' + register_number + ')'
    # Qualifiers
    alignQual = r'(align\(\d+\))'
    widthQual = r'(width\((\d+|all)\))'
    allocQual = r'(alloc\(agent\))'
    # Instruction Modifiers
    roundingMod = (r'((_ftz)?(_up|_down|_zero|_near))')
    datatypeMod = (r'_('
                   # packedTypes
                   r'u8x4|s8x4|u16x2|s16x2|u8x8|s8x8|u16x4|s16x4|u32x2|s32x2|'
                   r'u8x16|s8x16|u16x8|s16x8|u32x4|s32x4|u64x2|s64x2|'
                   r'f16x2|f16x4|f16x8|f32x2|f32x4|f64x2|'
                   # baseTypes
                   r'u8|s8|u16|s16|u32|s32|u64|s64|'
                   r'b128|b8|b16|b32|b64|b1|'
                   r'f16|f32|f64|'
                   # opaqueType
                   r'roimg|woimg|rwimg|samp|sig32|sig64)')

    # Numeric Constant
    float = r'((\d+\.)|(\d*\.\d+))[eE][+-]?\d+'
    hexfloat = r'0[xX](([0-9a-fA-F]+\.[0-9a-fA-F]*)|([0-9a-fA-F]*\.[0-9a-fA-F]+))[pP][+-]?\d+'
    ieeefloat = r'0((h|H)[0-9a-fA-F]{4}|(f|F)[0-9a-fA-F]{8}|(d|D)[0-9a-fA-F]{16})'

    tokens = {
        'root': [
            include('whitespace'),
            include('comments'),

            (string, String),

            (r'@' + identifier + ':?', Name.Label),

            (register, Name.Variable.Anonymous),

            include('keyword'),

            (r'&' + identifier, Name.Variable.Global),
            (r'%' + identifier, Name.Variable),

            (hexfloat, Number.Hex),
            (r'0[xX][a-fA-F0-9]+', Number.Hex),
            (ieeefloat, Number.Float),
            (float, Number.Float),
            (r'\d+', Number.Integer),

            (r'[=<>{}\[\]()*.,:;!]|x\b', Punctuation)
        ],
        'whitespace': [
            (r'(\n|\s)+', Text),
        ],
        'comments': [
            (r'/\*.*?\*/', Comment.Multiline),
            (r'//.*?\n', Comment.Single),
        ],
        'keyword': [
            # Types
            (r'kernarg' + datatypeMod, Keyword.Type),

            # Regular keywords
            (r'\$(full|base|small|large|default|zero|near)', Keyword),
            (words((
                'module', 'extension', 'pragma', 'prog', 'indirect', 'signature',
                'decl', 'kernel', 'function', 'enablebreakexceptions',
                'enabledetectexceptions', 'maxdynamicgroupsize', 'maxflatgridsize',
                'maxflatworkgroupsize', 'requireddim', 'requiredgridsize',
                'requiredworkgroupsize', 'requirenopartialworkgroups'),
                suffix=r'\b'), Keyword),

            # instructions
            (roundingMod, Keyword),
            (datatypeMod, Keyword),
            (r'_(' + alignQual + '|' + widthQual + ')', Keyword),
            (r'_kernarg', Keyword),
            (r'(nop|imagefence)\b', Keyword),
            (words((
                'cleardetectexcept', 'clock', 'cuid', 'debugtrap', 'dim',
                'getdetectexcept', 'groupbaseptr', 'kernargbaseptr', 'laneid',
                'maxcuid', 'maxwaveid', 'packetid', 'setdetectexcept', 'waveid',
                'workitemflatabsid', 'workitemflatid', 'nullptr', 'abs', 'bitrev',
                'currentworkgroupsize', 'currentworkitemflatid', 'fract', 'ncos',
                'neg', 'nexp2', 'nlog2', 'nrcp', 'nrsqrt', 'nsin', 'nsqrt',
                'gridgroups', 'gridsize', 'not', 'sqrt', 'workgroupid',
                'workgroupsize', 'workitemabsid', 'workitemid', 'ceil', 'floor',
                'rint', 'trunc', 'add', 'bitmask', 'borrow', 'carry', 'copysign',
                'div', 'rem', 'sub', 'shl', 'shr', 'and', 'or', 'xor', 'unpackhi',
                'unpacklo', 'max', 'min', 'fma', 'mad', 'bitextract', 'bitselect',
                'shuffle', 'cmov', 'bitalign', 'bytealign', 'lerp', 'nfma', 'mul',
                'mulhi', 'mul24hi', 'mul24', 'mad24', 'mad24hi', 'bitinsert',
                'combine', 'expand', 'lda', 'mov', 'pack', 'unpack', 'packcvt',
                'unpackcvt', 'sad', 'sementp', 'ftos', 'stof', 'cmp', 'ld', 'st',
                '_eq', '_ne', '_lt', '_le', '_gt', '_ge', '_equ', '_neu', '_ltu',
                '_leu', '_gtu', '_geu', '_num', '_nan', '_seq', '_sne', '_slt',
                '_sle', '_sgt', '_sge', '_snum', '_snan', '_sequ', '_sneu', '_sltu',
                '_sleu', '_sgtu', '_sgeu', 'atomic', '_ld', '_st', '_cas', '_add',
                '_and', '_exch', '_max', '_min', '_or', '_sub', '_wrapdec',
                '_wrapinc', '_xor', 'ret', 'cvt', '_readonly', '_kernarg', '_global',
                'br', 'cbr', 'sbr', '_scacq', '_screl', '_scar', '_rlx', '_wave',
                '_wg', '_agent', '_system', 'ldimage', 'stimage', '_v2', '_v3', '_v4',
                '_1d', '_2d', '_3d', '_1da', '_2da', '_1db', '_2ddepth', '_2dadepth',
                '_width', '_height', '_depth', '_array', '_channelorder',
                '_channeltype', 'querysampler', '_coord', '_filter', '_addressing',
                'barrier', 'wavebarrier', 'initfbar', 'joinfbar', 'waitfbar',
                'arrivefbar', 'leavefbar', 'releasefbar', 'ldf', 'activelaneid',
                'activelanecount', 'activelanemask', 'activelanepermute', 'call',
                'scall', 'icall', 'alloca', 'packetcompletionsig',
                'addqueuewriteindex', 'casqueuewriteindex', 'ldqueuereadindex',
                'stqueuereadindex', 'readonly', 'global', 'private', 'group',
                'spill', 'arg', '_upi', '_downi', '_zeroi', '_neari', '_upi_sat',
                '_downi_sat', '_zeroi_sat', '_neari_sat', '_supi', '_sdowni',
                '_szeroi', '_sneari', '_supi_sat', '_sdowni_sat', '_szeroi_sat',
                '_sneari_sat', '_pp', '_ps', '_sp', '_ss', '_s', '_p', '_pp_sat',
                '_ps_sat', '_sp_sat', '_ss_sat', '_s_sat', '_p_sat')), Keyword),

            # Integer types
            (r'i[1-9]\d*', Keyword)
        ]
    }


class LlvmLexer(RegexLexer):
    """
    For LLVM assembly code.
    """
    name = 'LLVM'
    aliases = ['llvm']
    filenames = ['*.ll']
    mimetypes = ['text/x-llvm']

    #: optional Comment or Whitespace
    string = r'"[^"]*?"'
    identifier = r'([-a-zA-Z$._][\w\-$.]*|' + string + ')'

    tokens = {
        'root': [
            include('whitespace'),

            # Before keywords, because keywords are valid label names :(...
            (identifier + r'\s*:', Name.Label),

            include('keyword'),

            (r'%' + identifier, Name.Variable),
            (r'@' + identifier, Name.Variable.Global),
            (r'%\d+', Name.Variable.Anonymous),
            (r'@\d+', Name.Variable.Global),
            (r'#\d+', Name.Variable.Global),
            (r'!' + identifier, Name.Variable),
            (r'!\d+', Name.Variable.Anonymous),
            (r'c?' + string, String),

            (r'0[xX][a-fA-F0-9]+', Number),
            (r'-?\d+(?:[.]\d+)?(?:[eE][-+]?\d+(?:[.]\d+)?)?', Number),

            (r'[=<>{}\[\]()*.,!]|x\b', Punctuation)
        ],
        'whitespace': [
            (r'(\n|\s)+', Text),
            (r';.*?\n', Comment)
        ],
        'keyword': [
            # Regular keywords
            (words((
                'acq_rel', 'acquire', 'add', 'addrspace', 'addrspacecast', 'afn', 'alias',
                'aliasee', 'align', 'alignLog2', 'alignstack', 'alloca', 'allocsize', 'allOnes',
                'alwaysinline', 'amdgpu_cs', 'amdgpu_es', 'amdgpu_gs', 'amdgpu_hs',
                'amdgpu_kernel', 'amdgpu_ls', 'amdgpu_ps', 'amdgpu_vs', 'and', 'any',
                'anyregcc', 'appending', 'arcp', 'argmemonly', 'args', 'arm_aapcs_vfpcc',
                'arm_aapcscc', 'arm_apcscc', 'ashr', 'asm', 'atomic', 'atomicrmw', 'attributes',
                'available_externally', 'avr_intrcc', 'avr_signalcc', 'bit', 'bitcast',
                'bitMask', 'blockaddress', 'br', 'branchFunnel', 'builtin', 'byArg', 'byte',
                'byteArray', 'byval', 'c', 'call', 'callee', 'caller', 'calls', 'catch',
                'catchpad', 'catchret', 'catchswitch', 'cc', 'ccc', 'cleanup', 'cleanuppad',
                'cleanupret', 'cmpxchg', 'cold', 'coldcc', 'comdat', 'common', 'constant',
                'contract', 'convergent', 'critical', 'cxx_fast_tlscc', 'datalayout', 'declare',
                'default', 'define', 'deplibs', 'dereferenceable', 'dereferenceable_or_null',
                'distinct', 'dllexport', 'dllimport', 'double', 'dso_local', 'dso_preemptable',
                'dsoLocal', 'eq', 'exact', 'exactmatch', 'extern_weak', 'external',
                'externally_initialized', 'extractelement', 'extractvalue', 'fadd', 'false',
                'fast', 'fastcc', 'fcmp', 'fdiv', 'fence', 'filter', 'flags', 'float', 'fmul',
                'fp128', 'fpext', 'fptosi', 'fptoui', 'fptrunc', 'frem', 'from', 'fsub',
                'funcFlags', 'function', 'gc', 'getelementptr', 'ghccc', 'global', 'guid', 'gv',
                'half', 'hash', 'hhvm_ccc', 'hhvmcc', 'hidden', 'hot', 'hotness', 'icmp',
                'ifunc', 'inaccessiblemem_or_argmemonly', 'inaccessiblememonly', 'inalloca',
                'inbounds', 'indir', 'indirectbr', 'info', 'initialexec', 'inline',
                'inlineBits', 'inlinehint', 'inrange', 'inreg', 'insertelement', 'insertvalue',
                'insts', 'intel_ocl_bicc', 'inteldialect', 'internal', 'inttoptr', 'invoke',
                'jumptable', 'kind', 'label', 'landingpad', 'largest', 'linkage', 'linkonce',
                'linkonce_odr', 'live', 'load', 'local_unnamed_addr', 'localdynamic',
                'localexec', 'lshr', 'max', 'metadata', 'min', 'minsize', 'module', 'monotonic',
                'msp430_intrcc', 'mul', 'musttail', 'naked', 'name', 'nand', 'ne', 'nest',
                'ninf', 'nnan', 'noalias', 'nobuiltin', 'nocapture', 'nocf_check',
                'noduplicate', 'noduplicates', 'noimplicitfloat', 'noinline', 'none',
                'nonlazybind', 'nonnull', 'norecurse', 'noRecurse', 'noredzone', 'noreturn',
                'notail', 'notEligibleToImport', 'nounwind', 'nsw', 'nsz', 'null', 'nuw', 'oeq',
                'offset', 'oge', 'ogt', 'ole', 'olt', 'one', 'opaque', 'optforfuzzing',
                'optnone', 'optsize', 'or', 'ord', 'path', 'personality', 'phi', 'ppc_fp128',
                'prefix', 'preserve_allcc', 'preserve_mostcc', 'private', 'prologue',
                'protected', 'ptrtoint', 'ptx_device', 'ptx_kernel', 'readnone', 'readNone',
                'readonly', 'readOnly', 'reassoc', 'refs', 'relbf', 'release', 'resByArg',
                'resume', 'ret', 'returnDoesNotAlias', 'returned', 'returns_twice', 'safestack',
                'samesize', 'sanitize_address', 'sanitize_hwaddress', 'sanitize_memory',
                'sanitize_thread', 'sdiv', 'section', 'select', 'seq_cst', 'sext', 'sge', 'sgt',
                'shadowcallstack', 'shl', 'shufflevector', 'sideeffect', 'signext', 'single',
                'singleImpl', 'singleImplName', 'sitofp', 'sizeM1', 'sizeM1BitWidth', 'sle',
                'slt', 'source_filename', 'speculatable', 'spir_func', 'spir_kernel', 'srem',
                'sret', 'ssp', 'sspreq', 'sspstrong', 'store', 'strictfp', 'sub', 'summaries',
                'summary', 'swiftcc', 'swifterror', 'swiftself', 'switch', 'syncscope', 'tail',
                'target', 'thread_local', 'to', 'token', 'triple', 'true', 'trunc', 'type',
                'typeCheckedLoadConstVCalls', 'typeCheckedLoadVCalls', 'typeid', 'typeIdInfo',
                'typeTestAssumeConstVCalls', 'typeTestAssumeVCalls', 'typeTestRes', 'typeTests',
                'udiv', 'ueq', 'uge', 'ugt', 'uitofp', 'ule', 'ult', 'umax', 'umin', 'undef',
                'une', 'uniformRetVal', 'uniqueRetVal', 'unknown', 'unnamed_addr', 'uno',
                'unordered', 'unreachable', 'unsat', 'unwind', 'urem', 'uselistorder',
                'uselistorder_bb', 'uwtable', 'va_arg', 'variable', 'vFuncId',
                'virtualConstProp', 'void', 'volatile', 'weak', 'weak_odr', 'webkit_jscc',
                'win64cc', 'within', 'wpdRes', 'wpdResolutions', 'writeonly', 'x',
                'x86_64_sysvcc', 'x86_fastcallcc', 'x86_fp80', 'x86_intrcc', 'x86_mmx',
                'x86_regcallcc', 'x86_stdcallcc', 'x86_thiscallcc', 'x86_vectorcallcc', 'xchg',
                'xor', 'zeroext', 'zeroinitializer', 'zext'),
                suffix=r'\b'), Keyword),

            # Types
            (words(('void', 'half', 'float', 'double', 'x86_fp80', 'fp128',
                    'ppc_fp128', 'label', 'metadata', 'token')), Keyword.Type),

            # Integer types
            (r'i[1-9]\d*', Keyword)
        ]
    }


class NasmLexer(RegexLexer):
    """
    For Nasm (Intel) assembly code.
    """
    name = 'NASM'
    aliases = ['nasm']
    filenames = ['*.asm', '*.ASM']
    mimetypes = ['text/x-nasm']

    # Tasm uses the same file endings, but TASM is not as common as NASM, so
    # we prioritize NASM higher by default
    priority = 1.0

    identifier = r'[a-z$._?][\w$.?#@~]*'
    hexn = r'(?:0x[0-9a-f]+|$0[0-9a-f]*|[0-9]+[0-9a-f]*h)'
    octn = r'[0-7]+q'
    binn = r'[01]+b'
    decn = r'[0-9]+'
    floatn = decn + r'\.e?' + decn
    string = r'"(\\"|[^"\n])*"|' + r"'(\\'|[^'\n])*'|" + r"`(\\`|[^`\n])*`"
    declkw = r'(?:res|d)[bwdqt]|times'
    register = (r'r[0-9][0-5]?[bwd]|'
                r'[a-d][lh]|[er]?[a-d]x|[er]?[sb]p|[er]?[sd]i|[c-gs]s|st[0-7]|'
                r'mm[0-7]|cr[0-4]|dr[0-367]|tr[3-7]')
    wordop = r'seg|wrt|strict'
    type = r'byte|[dq]?word'
    # Directives must be followed by whitespace, otherwise CPU will match
    # cpuid for instance.
    directives = (r'(?:BITS|USE16|USE32|SECTION|SEGMENT|ABSOLUTE|EXTERN|GLOBAL|'
                  r'ORG|ALIGN|STRUC|ENDSTRUC|COMMON|CPU|GROUP|UPPERCASE|IMPORT|'
                  r'EXPORT|LIBRARY|MODULE)\s+')

    flags = re.IGNORECASE | re.MULTILINE
    tokens = {
        'root': [
            (r'^\s*%', Comment.Preproc, 'preproc'),
            include('whitespace'),
            (identifier + ':', Name.Label),
            (r'(%s)(\s+)(equ)' % identifier,
                bygroups(Name.Constant, Keyword.Declaration, Keyword.Declaration),
                'instruction-args'),
            (directives, Keyword, 'instruction-args'),
            (declkw, Keyword.Declaration, 'instruction-args'),
            (identifier, Name.Function, 'instruction-args'),
            (r'[\r\n]+', Text)
        ],
        'instruction-args': [
            (string, String),
            (hexn, Number.Hex),
            (octn, Number.Oct),
            (binn, Number.Bin),
            (floatn, Number.Float),
            (decn, Number.Integer),
            include('punctuation'),
            (register, Name.Builtin),
            (identifier, Name.Variable),
            (r'[\r\n]+', Text, '#pop'),
            include('whitespace')
        ],
        'preproc': [
            (r'[^;\n]+', Comment.Preproc),
            (r';.*?\n', Comment.Single, '#pop'),
            (r'\n', Comment.Preproc, '#pop'),
        ],
        'whitespace': [
            (r'\n', Text),
            (r'[ \t]+', Text),
            (r';.*', Comment.Single)
        ],
        'punctuation': [
            (r'[,():\[\]]+', Punctuation),
            (r'[&|^<>+*/%~-]+', Operator),
            (r'[$]+', Keyword.Constant),
            (wordop, Operator.Word),
            (type, Keyword.Type)
        ],
    }

    def analyse_text(text):
        # Probably TASM
        if re.match(r'PROC', text, re.IGNORECASE):
            return False


class NasmObjdumpLexer(ObjdumpLexer):
    """
    For the output of 'objdump -d -M intel'.

    .. versionadded:: 2.0
    """
    name = 'objdump-nasm'
    aliases = ['objdump-nasm']
    filenames = ['*.objdump-intel']
    mimetypes = ['text/x-nasm-objdump']

    tokens = _objdump_lexer_tokens(NasmLexer)


class TasmLexer(RegexLexer):
    """
    For Tasm (Turbo Assembler) assembly code.
    """
    name = 'TASM'
    aliases = ['tasm']
    filenames = ['*.asm', '*.ASM', '*.tasm']
    mimetypes = ['text/x-tasm']

    identifier = r'[@a-z$._?][\w$.?#@~]*'
    hexn = r'(?:0x[0-9a-f]+|$0[0-9a-f]*|[0-9]+[0-9a-f]*h)'
    octn = r'[0-7]+q'
    binn = r'[01]+b'
    decn = r'[0-9]+'
    floatn = decn + r'\.e?' + decn
    string = r'"(\\"|[^"\n])*"|' + r"'(\\'|[^'\n])*'|" + r"`(\\`|[^`\n])*`"
    declkw = r'(?:res|d)[bwdqt]|times'
    register = (r'r[0-9][0-5]?[bwd]|'
                r'[a-d][lh]|[er]?[a-d]x|[er]?[sb]p|[er]?[sd]i|[c-gs]s|st[0-7]|'
                r'mm[0-7]|cr[0-4]|dr[0-367]|tr[3-7]')
    wordop = r'seg|wrt|strict'
    type = r'byte|[dq]?word'
    directives = (r'BITS|USE16|USE32|SECTION|SEGMENT|ABSOLUTE|EXTERN|GLOBAL|'
                  r'ORG|ALIGN|STRUC|ENDSTRUC|ENDS|COMMON|CPU|GROUP|UPPERCASE|INCLUDE|'
                  r'EXPORT|LIBRARY|MODULE|PROC|ENDP|USES|ARG|DATASEG|UDATASEG|END|IDEAL|'
                  r'P386|MODEL|ASSUME|CODESEG|SIZE')
    # T[A-Z][a-z] is more of a convention. Lexer should filter out STRUC definitions
    # and then 'add' them to datatype somehow.
    datatype = (r'db|dd|dw|T[A-Z][a-z]+')

    flags = re.IGNORECASE | re.MULTILINE
    tokens = {
        'root': [
            (r'^\s*%', Comment.Preproc, 'preproc'),
            include('whitespace'),
            (identifier + ':', Name.Label),
            (directives, Keyword, 'instruction-args'),
            (r'(%s)(\s+)(%s)' % (identifier, datatype),
                bygroups(Name.Constant, Keyword.Declaration, Keyword.Declaration),
                'instruction-args'),
            (declkw, Keyword.Declaration, 'instruction-args'),
            (identifier, Name.Function, 'instruction-args'),
            (r'[\r\n]+', Text)
        ],
        'instruction-args': [
            (string, String),
            (hexn, Number.Hex),
            (octn, Number.Oct),
            (binn, Number.Bin),
            (floatn, Number.Float),
            (decn, Number.Integer),
            include('punctuation'),
            (register, Name.Builtin),
            (identifier, Name.Variable),
            # Do not match newline when it's preceeded by a backslash
            (r'(\\\s*)(;.*)([\r\n])', bygroups(Text, Comment.Single, Text)),
            (r'[\r\n]+', Text, '#pop'),
            include('whitespace')
        ],
        'preproc': [
            (r'[^;\n]+', Comment.Preproc),
            (r';.*?\n', Comment.Single, '#pop'),
            (r'\n', Comment.Preproc, '#pop'),
        ],
        'whitespace': [
            (r'[\n\r]', Text),
            (r'\\[\n\r]', Text),
            (r'[ \t]+', Text),
            (r';.*', Comment.Single)
        ],
        'punctuation': [
            (r'[,():\[\]]+', Punctuation),
            (r'[&|^<>+*=/%~-]+', Operator),
            (r'[$]+', Keyword.Constant),
            (wordop, Operator.Word),
            (type, Keyword.Type)
        ],
    }

    def analyse_text(text):
        # See above
        if re.match(r'PROC', text, re.I):
            return True


class Ca65Lexer(RegexLexer):
    """
    For ca65 assembler sources.

    .. versionadded:: 1.6
    """
    name = 'ca65 assembler'
    aliases = ['ca65']
    filenames = ['*.s']

    flags = re.IGNORECASE

    tokens = {
        'root': [
            (r';.*', Comment.Single),
            (r'\s+', Text),
            (r'[a-z_.@$][\w.@$]*:', Name.Label),
            (r'((ld|st)[axy]|(in|de)[cxy]|asl|lsr|ro[lr]|adc|sbc|cmp|cp[xy]'
             r'|cl[cvdi]|se[cdi]|jmp|jsr|bne|beq|bpl|bmi|bvc|bvs|bcc|bcs'
             r'|p[lh][ap]|rt[is]|brk|nop|ta[xy]|t[xy]a|txs|tsx|and|ora|eor'
             r'|bit)\b', Keyword),
            (r'\.\w+', Keyword.Pseudo),
            (r'[-+~*/^&|!<>=]', Operator),
            (r'"[^"\n]*.', String),
            (r"'[^'\n]*.", String.Char),
            (r'\$[0-9a-f]+|[0-9a-f]+h\b', Number.Hex),
            (r'\d+', Number.Integer),
            (r'%[01]+', Number.Bin),
            (r'[#,.:()=\[\]]', Punctuation),
            (r'[a-z_.@$][\w.@$]*', Name),
        ]
    }

    def analyse_text(self, text):
        # comments in GAS start with "#"
        if re.match(r'^\s*;', text, re.MULTILINE):
            return 0.9


class Dasm16Lexer(RegexLexer):
    """
    Simple lexer for DCPU-16 Assembly

    Check http://0x10c.com/doc/dcpu-16.txt

    .. versionadded:: 2.4
    """
    name = 'DASM16'
    aliases = ['dasm16']
    filenames = ['*.dasm16', '*.dasm']
    mimetypes = ['text/x-dasm16']

    INSTRUCTIONS = [
        'SET',
        'ADD', 'SUB',
        'MUL', 'MLI',
        'DIV', 'DVI',
        'MOD', 'MDI',
        'AND', 'BOR', 'XOR',
        'SHR', 'ASR', 'SHL',
        'IFB', 'IFC', 'IFE', 'IFN', 'IFG', 'IFA', 'IFL', 'IFU',
        'ADX', 'SBX',
        'STI', 'STD',
        'JSR',
        'INT', 'IAG', 'IAS', 'RFI', 'IAQ', 'HWN', 'HWQ', 'HWI',
    ]

    REGISTERS = [
        'A', 'B', 'C',
        'X', 'Y', 'Z',
        'I', 'J',
        'SP', 'PC', 'EX',
        'POP', 'PEEK', 'PUSH'
    ]

    # Regexes yo
    char = r'[a-zA-Z$._0-9@]'
    identifier = r'(?:[a-zA-Z$_]' + char + r'*|\.' + char + '+)'
    number = r'[+-]?(?:0[xX][a-zA-Z0-9]+|\d+)'
    binary_number = r'0b[01_]+'
    instruction = r'(?i)(' + '|'.join(INSTRUCTIONS) + ')'
    single_char = r"'\\?" + char + "'"
    string = r'"(\\"|[^"])*"'

    def guess_identifier(lexer, match):
        ident = match.group(0)
        klass = Name.Variable if ident.upper() in lexer.REGISTERS else Name.Label
        yield match.start(), klass, ident

    tokens = {
        'root': [
            include('whitespace'),
            (':' + identifier, Name.Label),
            (identifier + ':', Name.Label),
            (instruction, Name.Function, 'instruction-args'),
            (r'\.' + identifier, Name.Function, 'data-args'),
            (r'[\r\n]+', Text)
        ],

        'numeric' : [
            (binary_number, Number.Integer),
            (number, Number.Integer),
            (single_char, String),
        ],

        'arg' : [
            (identifier, guess_identifier),
            include('numeric')
        ],

        'deref' : [
            (r'\+', Punctuation),
            (r'\]', Punctuation, '#pop'),
            include('arg'),
            include('whitespace')
        ],

        'instruction-line' : [
            (r'[\r\n]+', Text, '#pop'),
            (r';.*?$', Comment, '#pop'),
            include('whitespace')
        ],

        'instruction-args': [
            (r',', Punctuation),
            (r'\[', Punctuation, 'deref'),
            include('arg'),
            include('instruction-line')
        ],

        'data-args' : [
            (r',', Punctuation),
            include('numeric'),
            (string, String),
            include('instruction-line')
        ],

        'whitespace': [
            (r'\n', Text),
            (r'\s+', Text),
            (r';.*?\n', Comment)
        ],
    }
