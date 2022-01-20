from typing import Generator, List, Optional, Tuple

from parso.python.token import TokenType

class GrammarParser:
    generator: Generator[TokenType, None, None]
    def __init__(self, bnf_grammar: str) -> None: ...
    def parse(self) -> Generator[Tuple[NFAState, NFAState], None, None]: ...

class NFAArc:
    next: NFAState
    nonterminal_or_string: Optional[str]
    def __init__(
        self, next_: NFAState, nonterminal_or_string: Optional[str]
    ) -> None: ...

class NFAState:
    from_rule: str
    arcs: List[NFAArc]
    def __init__(self, from_rule: str) -> None: ...
