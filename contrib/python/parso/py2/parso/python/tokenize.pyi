from typing import Generator, Iterable, NamedTuple, Tuple

from parso.python.token import TokenType
from parso.utils import PythonVersionInfo

class Token(NamedTuple):
    type: TokenType
    string: str
    start_pos: Tuple[int, int]
    prefix: str
    @property
    def end_pos(self) -> Tuple[int, int]: ...

class PythonToken(Token):
    def __repr__(self) -> str: ...

def tokenize(
    code: str, version_info: PythonVersionInfo, start_pos: Tuple[int, int] = (1, 0)
) -> Generator[PythonToken, None, None]: ...
def tokenize_lines(
    lines: Iterable[str],
    version_info: PythonVersionInfo,
    start_pos: Tuple[int, int] = (1, 0),
) -> Generator[PythonToken, None, None]: ...
