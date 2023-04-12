"""
    pygments.lexers.cplint
    ~~~~~~~~~~~~~~~~~~~~~~

    Lexer for the cplint language
    
    :copyright: Copyright 2006-2022 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import re

from pygments.lexer import bygroups, inherit, words
from pygments.lexers import PrologLexer
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
    Number, Punctuation

__all__ = ['CplintLexer']


class CplintLexer(PrologLexer):
    """
    Lexer for cplint files, including CP-logic, Logic Programs with Annotated Disjunctions, 
    Distributional Clauses syntax, ProbLog, DTProbLog

    .. versionadded:: 2.12
    """
    name = 'cplint'
    url = 'https://cplint.eu'
    aliases = ['cplint']
    filenames = ['*.ecl', '*.prolog', '*.pro', '*.pl', '*.P', '*.lpad', '*.cpl']
    mimetypes = ['text/x-cplint']

    tokens = {
        'root': [
            (r'map_query',Keyword),
            (words(('gaussian','uniform_dens','dirichlet','gamma','beta','poisson','binomial','geometric',
              'exponential','pascal','multinomial','user','val',
              'uniform','discrete','finite')),Name.Builtin),
            (r'([a-z]+)(:)', bygroups(String.Atom, Punctuation)), # annotations of atoms
            (r':(-|=)|::?|~=?|=>', Operator),
            (r'\?', Name.Builtin),
            inherit,
        ],
    }


