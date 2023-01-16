from typing import Any, Callable, Generic, Optional, Sequence, TypeVar, Union
from typing_extensions import Literal

from parso.utils import PythonVersionInfo

_Token = Any
_NodeT = TypeVar("_NodeT")

class Grammar(Generic[_NodeT]):
    _default_normalizer_config: Optional[Any] = ...
    _error_normalizer_config: Optional[Any] = None
    _start_nonterminal: str = ...
    _token_namespace: Optional[str] = None
    def __init__(
        self,
        text: str,
        tokenizer: Callable[[Sequence[str], int], Sequence[_Token]],
        parser: Any = ...,
        diff_parser: Any = ...,
    ) -> None: ...
    def parse(
        self,
        code: Union[str, bytes] = ...,
        error_recovery: bool = ...,
        path: Optional[str] = ...,
        start_symbol: Optional[str] = ...,
        cache: bool = ...,
        diff_cache: bool = ...,
        cache_path: Optional[str] = ...,
    ) -> _NodeT: ...

class PythonGrammar(Grammar):
    version_info: PythonVersionInfo
    def __init__(self, version_info: PythonVersionInfo, bnf_text: str) -> None: ...

def load_grammar(
    language: Literal["python"] = "python", version: Optional[str] = ..., path: str = ...
) -> Grammar: ...
