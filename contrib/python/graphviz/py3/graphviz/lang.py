# lang.py - dot language creation helpers

"""Quote strings to be valid DOT identifiers, assemble attribute lists."""

import functools
import re
import typing

from . import tools

__all__ = ['quote', 'quote_edge',
           'a_list', 'attr_list',
           'escape', 'nohtml']

# https://www.graphviz.org/doc/info/lang.html
# https://www.graphviz.org/doc/info/attrs.html#k:escString

HTML_STRING = re.compile(r'<.*>$', re.DOTALL)

ID = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*|-?(\.[0-9]+|[0-9]+(\.[0-9]*)?))$')

KEYWORDS = {'node', 'edge', 'graph', 'digraph', 'subgraph', 'strict'}

COMPASS = {'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw', 'c', '_'}  # TODO

QUOTE_OPTIONAL_BACKSLASHES = re.compile(r'(?P<bs>(?:\\\\)*)'
                                        r'\\?(?P<quote>")')

ESCAPE_UNESCAPED_QUOTES = functools.partial(QUOTE_OPTIONAL_BACKSLASHES.sub,
                                            r'\g<bs>\\\g<quote>')


def quote(identifier: str,
          is_html_string=HTML_STRING.match,
          is_valid_id=ID.match,
          dot_keywords=KEYWORDS,
          escape_unescaped_quotes=ESCAPE_UNESCAPED_QUOTES) -> str:
    r"""Return DOT identifier from string, quote if needed.

    >>> quote('')
    '""'

    >>> quote('spam')
    'spam'

    >>> quote('spam spam')
    '"spam spam"'

    >>> quote('-4.2')
    '-4.2'

    >>> quote('.42')
    '.42'

    >>> quote('<<b>spam</b>>')
    '<<b>spam</b>>'

    >>> quote(nohtml('<>'))
    '"<>"'

    >>> print(quote('"'))
    "\""

    >>> print(quote('\\"'))
    "\""

    >>> print(quote('\\\\"'))
    "\\\""

    >>> print(quote('\\\\\\"'))
    "\\\""
    """
    if is_html_string(identifier) and not isinstance(identifier, NoHtml):
        pass
    elif not is_valid_id(identifier) or identifier.lower() in dot_keywords:
        return f'"{escape_unescaped_quotes(identifier)}"'
    return identifier


def quote_edge(identifier: str) -> str:
    """Return DOT edge statement node_id from string, quote if needed.

    >>> quote_edge('spam')
    'spam'

    >>> quote_edge('spam spam:eggs eggs')
    '"spam spam":"eggs eggs"'

    >>> quote_edge('spam:eggs:s')
    'spam:eggs:s'
    """
    node, _, rest = identifier.partition(':')
    parts = [quote(node)]
    if rest:
        port, _, compass = rest.partition(':')
        parts.append(quote(port))
        if compass:
            parts.append(compass)
    return ':'.join(parts)


def a_list(label: typing.Optional[str] = None,
           kwargs=None, attributes=None) -> str:
    """Return assembled DOT a_list string.

    >>> a_list('spam', {'spam': None, 'ham': 'ham ham', 'eggs': ''})
    'label=spam eggs="" ham="ham ham"'
    """
    result = [f'label={quote(label)}'] if label is not None else []
    if kwargs:
        items = [f'{quote(k)}={quote(v)}'
                 for k, v in tools.mapping_items(kwargs) if v is not None]
        result.extend(items)
    if attributes:
        if hasattr(attributes, 'items'):
            attributes = tools.mapping_items(attributes)
        items = [f'{quote(k)}={quote(v)}'
                 for k, v in attributes if v is not None]
        result.extend(items)
    return ' '.join(result)


def attr_list(label: typing.Optional[str] = None,
              kwargs=None, attributes=None) -> str:
    """Return assembled DOT attribute list string.

    Sorts ``kwargs`` and ``attributes`` if they are plain dicts (to avoid
    unpredictable order from hash randomization in Python 3 versions).

    >>> attr_list()
    ''

    >>> attr_list('spam spam', kwargs={'eggs': 'eggs', 'ham': 'ham ham'})
    ' [label="spam spam" eggs=eggs ham="ham ham"]'

    >>> attr_list(kwargs={'spam': None, 'eggs': ''})
    ' [eggs=""]'
    """
    content = a_list(label, kwargs, attributes)
    if not content:
        return ''
    return f' [{content}]'


def escape(s: str) -> 'NoHtml':
    r"""Return ``s`` as literal disabling special meaning of backslashes and ``'<...>'``.

    see also https://www.graphviz.org/doc/info/attrs.html#k:escString

    Args:
        s: String in which backslashes and ``'<...>'`` should be treated as literal.

    Returns:
        Escaped string subclass instance.

    Raises:
        TypeError: If ``s`` is not a ``str`` on Python 3, or a ``str``/``unicode`` on Python 2.

    Example:
        >>> import graphviz
        >>> print(graphviz.escape(r'\l'))
        \\l
    """
    return nohtml(s.replace('\\', '\\\\'))


class NoHtml(str):
    """String subclass that does not treat ``'<...>'`` as DOT HTML string."""

    __slots__ = ()


def nohtml(s: str) -> NoHtml:
    """Return copy of ``s`` that will not treat ``'<...>'`` as DOT HTML string in quoting.

    Args:
        s: String in which leading ``'<'`` and trailing ``'>'`` should be treated as literal.

    Returns:
        String subclass instance.

    Raises:
        TypeError: If ``s`` is not a ``str`` on Python 3, or a ``str``/``unicode`` on Python 2.

    Example:
        >>> import graphviz
        >>> g = graphviz.Graph()
        >>> g.node(graphviz.nohtml('<>-*-<>'))
        >>> print(g.source)  # doctest: +NORMALIZE_WHITESPACE
        graph {
            "<>-*-<>"
        }
    """
    return NoHtml(s)
