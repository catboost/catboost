"""
    pygments.lexers.chapel
    ~~~~~~~~~~~~~~~~~~~~~~

    Lexer for the Chapel language.

    :copyright: Copyright 2006-2021 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

from pygments.lexer import RegexLexer, bygroups, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
    Number, Punctuation

__all__ = ['ChapelLexer']


class ChapelLexer(RegexLexer):
    """
    For `Chapel <https://chapel-lang.org/>`_ source.

    .. versionadded:: 2.0
    """
    name = 'Chapel'
    filenames = ['*.chpl']
    aliases = ['chapel', 'chpl']
    # mimetypes = ['text/x-chapel']

    known_types = ('bool', 'bytes', 'complex', 'imag', 'int', 'locale',
                   'nothing', 'opaque', 'range', 'real', 'string', 'uint',
                   'void')

    type_modifiers_par = ('atomic', 'single', 'sync')
    type_modifiers_mem = ('borrowed', 'owned', 'shared', 'unmanaged')
    type_modifiers = (*type_modifiers_par, *type_modifiers_mem)

    declarations = ('config', 'const', 'in', 'inout', 'out', 'param', 'ref',
                    'type', 'var')

    constants = ('false', 'nil', 'none', 'true')

    other_keywords = ('align', 'as',
                      'begin', 'break', 'by',
                      'catch', 'cobegin', 'coforall', 'continue',
                      'defer', 'delete', 'dmapped', 'do', 'domain',
                      'else', 'enum', 'except', 'export', 'extern',
                      'for', 'forall', 'foreach', 'forwarding',
                      'if', 'implements', 'import', 'index', 'init', 'inline',
                      'label', 'lambda', 'let', 'lifetime', 'local',
                      'new', 'noinit',
                      'on', 'only', 'otherwise', 'override',
                      'pragma', 'primitive', 'private', 'prototype', 'public',
                      'reduce', 'require', 'return',
                      'scan', 'select', 'serial', 'sparse', 'subdomain',
                      'then', 'this', 'throw', 'throws', 'try',
                      'use',
                      'when', 'where', 'while', 'with',
                      'yield',
                      'zip')

    tokens = {
        'root': [
            (r'\n', Text),
            (r'\s+', Text),
            (r'\\\n', Text),

            (r'//(.*?)\n', Comment.Single),
            (r'/(\\\n)?[*](.|\n)*?[*](\\\n)?/', Comment.Multiline),

            (words(declarations, suffix=r'\b'), Keyword.Declaration),
            (words(constants, suffix=r'\b'), Keyword.Constant),
            (words(known_types, suffix=r'\b'), Keyword.Type),
            (words((*type_modifiers, *other_keywords), suffix=r'\b'), Keyword),

            (r'(iter)((?:\s)+)', bygroups(Keyword, Text), 'procname'),
            (r'(proc)((?:\s)+)', bygroups(Keyword, Text), 'procname'),
            (r'(operator)((?:\s)+)', bygroups(Keyword, Text), 'procname'),
            (r'(class|interface|module|record|union)(\s+)', bygroups(Keyword, Text),
             'classname'),

            # imaginary integers
            (r'\d+i', Number),
            (r'\d+\.\d*([Ee][-+]\d+)?i', Number),
            (r'\.\d+([Ee][-+]\d+)?i', Number),
            (r'\d+[Ee][-+]\d+i', Number),

            # reals cannot end with a period due to lexical ambiguity with
            # .. operator. See reference for rationale.
            (r'(\d*\.\d+)([eE][+-]?[0-9]+)?i?', Number.Float),
            (r'\d+[eE][+-]?[0-9]+i?', Number.Float),

            # integer literals
            # -- binary
            (r'0[bB][01]+', Number.Bin),
            # -- hex
            (r'0[xX][0-9a-fA-F]+', Number.Hex),
            # -- octal
            (r'0[oO][0-7]+', Number.Oct),
            # -- decimal
            (r'[0-9]+', Number.Integer),

            # strings
            (r'"(\\\\|\\"|[^"])*"', String),
            (r"'(\\\\|\\'|[^'])*'", String),

            # tokens
            (r'(=|\+=|-=|\*=|/=|\*\*=|%=|&=|\|=|\^=|&&=|\|\|=|<<=|>>=|'
             r'<=>|<~>|\.\.|by|#|\.\.\.|'
             r'&&|\|\||!|&|\||\^|~|<<|>>|'
             r'==|!=|<=|>=|<|>|'
             r'[+\-*/%]|\*\*)', Operator),
            (r'[:;,.?()\[\]{}]', Punctuation),

            # identifiers
            (r'[a-zA-Z_][\w$]*', Name.Other),
        ],
        'classname': [
            (r'[a-zA-Z_][\w$]*', Name.Class, '#pop'),
        ],
        'procname': [
            (r'([a-zA-Z_][.\w$]*|'  # regular function name, including secondary
             r'\~[a-zA-Z_][.\w$]*|'  # support for legacy destructors
             r'[+*/!~%<>=&^|\-:]{1,2})',  # operators
             Name.Function, '#pop'),

            # allow `proc (atomic T).foo`
            (r'\(', Punctuation, "receivertype"),
            (r'\)+\.', Punctuation),
        ],
        'receivertype': [
            (words(type_modifiers, suffix=r'\b'), Keyword),
            (words(known_types, suffix=r'\b'), Keyword.Type),
            (r'[^()]*', Name.Other, '#pop'),
        ],
    }
