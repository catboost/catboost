from typing import Any, Optional, Union

from parso.grammar import Grammar as Grammar, load_grammar as load_grammar
from parso.parser import ParserSyntaxError as ParserSyntaxError
from parso.utils import python_bytes_to_unicode as python_bytes_to_unicode, split_lines as split_lines

__version__: str = ...

def parse(
    code: Optional[Union[str, bytes]],
    *,
    version: Optional[str] = None,
    error_recovery: bool = True,
    path: Optional[str] = None,
    start_symbol: Optional[str] = None,
    cache: bool = False,
    diff_cache: bool = False,
    cache_path: Optional[str] = None,
) -> Any: ...
