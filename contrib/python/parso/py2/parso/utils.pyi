from typing import NamedTuple, Optional, Sequence, Union

class Version(NamedTuple):
    major: int
    minor: int
    micro: int

def split_lines(string: str, keepends: bool = ...) -> Sequence[str]: ...
def python_bytes_to_unicode(
    source: Union[str, bytes], encoding: str = ..., errors: str = ...
) -> str: ...
def version_info() -> Version:
    """
    Returns a namedtuple of parso's version, similar to Python's
    ``sys.version_info``.
    """
    ...

class PythonVersionInfo(NamedTuple):
    major: int
    minor: int

def parse_version_string(version: Optional[str]) -> PythonVersionInfo:
    """
    Checks for a valid version number (e.g. `3.2` or `2.7.1` or `3`) and
    returns a corresponding version info that is always two characters long in
    decimal.
    """
    ...
