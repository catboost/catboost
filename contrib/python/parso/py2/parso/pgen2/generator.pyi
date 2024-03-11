from typing import Any, Generic, Mapping, Sequence, Set, TypeVar, Union

from parso.pgen2.grammar_parser import NFAState

_TokenTypeT = TypeVar("_TokenTypeT")

class Grammar(Generic[_TokenTypeT]):
    nonterminal_to_dfas: Mapping[str, Sequence[DFAState[_TokenTypeT]]]
    reserved_syntax_strings: Mapping[str, ReservedString]
    start_nonterminal: str
    def __init__(
        self,
        start_nonterminal: str,
        rule_to_dfas: Mapping[str, Sequence[DFAState]],
        reserved_syntax_strings: Mapping[str, ReservedString],
    ) -> None: ...

class DFAPlan:
    next_dfa: DFAState
    dfa_pushes: Sequence[DFAState]

class DFAState(Generic[_TokenTypeT]):
    from_rule: str
    nfa_set: Set[NFAState]
    is_final: bool
    arcs: Mapping[str, DFAState]  # map from all terminals/nonterminals to DFAState
    nonterminal_arcs: Mapping[str, DFAState]
    transitions: Mapping[Union[_TokenTypeT, ReservedString], DFAPlan]
    def __init__(
        self, from_rule: str, nfa_set: Set[NFAState], final: NFAState
    ) -> None: ...

class ReservedString:
    value: str
    def __init__(self, value: str) -> None: ...
    def __repr__(self) -> str: ...

def generate_grammar(bnf_grammar: str, token_namespace: Any) -> Grammar[Any]: ...
