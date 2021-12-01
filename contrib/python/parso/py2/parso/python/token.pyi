from typing import Container, Iterable

class TokenType:
    name: str
    contains_syntax: bool
    def __init__(self, name: str, contains_syntax: bool) -> None: ...

class TokenTypes:
    def __init__(
        self, names: Iterable[str], contains_syntax: Container[str]
    ) -> None: ...

# not an actual class in the source code, but we need this class to type the fields of
# PythonTokenTypes
class _FakePythonTokenTypesClass(TokenTypes):
    STRING: TokenType
    NUMBER: TokenType
    NAME: TokenType
    ERRORTOKEN: TokenType
    NEWLINE: TokenType
    INDENT: TokenType
    DEDENT: TokenType
    ERROR_DEDENT: TokenType
    FSTRING_STRING: TokenType
    FSTRING_START: TokenType
    FSTRING_END: TokenType
    OP: TokenType
    ENDMARKER: TokenType

PythonTokenTypes: _FakePythonTokenTypesClass = ...
