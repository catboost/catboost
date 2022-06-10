"""
    pygments.lexers.spice
    ~~~~~~~~~~~~~~~~~~~~~

    Lexers for the Spice programming language.

    :copyright: Copyright 2006-2022 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import re

from pygments.lexer import RegexLexer, bygroups, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
    Number, Punctuation, Whitespace

__all__ = ['SpiceLexer']


class SpiceLexer(RegexLexer):
    """
    For Spice source.

    .. versionadded:: 2.11
    """
    name = 'Spice'
    url = 'https://www.spicelang.com'
    filenames = ['*.spice']
    aliases = ['spice', 'spicelang']
    mimetypes = ['text/x-spice']

    tokens = {
        'root': [
            (r'\n', Whitespace),
            (r'\s+', Whitespace),
            (r'\\\n', Text),  # line continuations
            (r'//(.*?)\n', Comment.Single),
            (r'/(\\\n)?[*](.|\n)*?[*](\\\n)?/', Comment.Multiline),
            (r'(import|as)\b', Keyword.Namespace),
            (r'(f|p|type|struct|const)\b', Keyword.Declaration),
            (words(('if', 'else', 'for', 'foreach', 'while', 'break', 'continue', 'return', 'ext', 'inline', 'public'), suffix=r'\b'), Keyword),
            (r'(true|false)\b', Keyword.Constant),
            (words(('printf', 'sizeof'), suffix=r'\b(\()'), bygroups(Name.Builtin, Punctuation)),
            (words(('double', 'int', 'short', 'long', 'byte', 'char', 'string', 'bool', 'dyn'), suffix=r'\b'), Keyword.Type),
            # double_lit
            (r'\d+(\.\d+[eE][+\-]?\d+|\.\d*|[eE][+\-]?\d+)', Number.Double),
            (r'\.\d+([eE][+\-]?\d+)?', Number.Double),
            # short_lit
            (r'(0|[1-9][0-9]*s)', Number.Integer),
            # long_lit
            (r'(0|[1-9][0-9]*l)', Number.Integer.Long),
            # int_lit
            (r'(0|[1-9][0-9]*)', Number.Integer),
            # string_lit
            (r'"(\\\\|\\[^\\]|[^"\\])*"', String),
            # char_lit
            (r'\'(\\\\|\\[^\\]|[^\'\\])\'', String.Char),
            # tokens
            (r'(<<=|>>=|<<|>>|<=|>=|\+=|-=|\*=|/=|&&|\|\||&|\||\+\+|--|\%|==|!=|[.]{3}|[+\-*/&])', Operator),
            (r'[|<>=!()\[\]{}.,;:\?]', Punctuation),
            # identifiers
            (r'[^\W\d]\w*', Name.Other),
        ]
    }
