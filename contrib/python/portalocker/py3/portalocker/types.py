from __future__ import annotations

import pathlib
import typing
from typing import Union

# fmt: off
Mode = typing.Literal[
    # Text modes
    # Read text
    'r', 'rt', 'tr',
    # Write text
    'w', 'wt', 'tw',
    # Append text
    'a', 'at', 'ta',
    # Exclusive creation text
    'x', 'xt', 'tx',
    # Read and write text
    'r+', '+r', 'rt+', 'r+t', '+rt', 'tr+', 't+r', '+tr',
    # Write and read text
    'w+', '+w', 'wt+', 'w+t', '+wt', 'tw+', 't+w', '+tw',
    # Append and read text
    'a+', '+a', 'at+', 'a+t', '+at', 'ta+', 't+a', '+ta',
    # Exclusive creation and read text
    'x+', '+x', 'xt+', 'x+t', '+xt', 'tx+', 't+x', '+tx',
    # Universal newline support
    'U', 'rU', 'Ur', 'rtU', 'rUt', 'Urt', 'trU', 'tUr', 'Utr',

    # Binary modes
    # Read binary
    'rb', 'br',
    # Write binary
    'wb', 'bw',
    # Append binary
    'ab', 'ba',
    # Exclusive creation binary
    'xb', 'bx',
    # Read and write binary
    'rb+', 'r+b', '+rb', 'br+', 'b+r', '+br',
    # Write and read binary
    'wb+', 'w+b', '+wb', 'bw+', 'b+w', '+bw',
    # Append and read binary
    'ab+', 'a+b', '+ab', 'ba+', 'b+a', '+ba',
    # Exclusive creation and read binary
    'xb+', 'x+b', '+xb', 'bx+', 'b+x', '+bx',
    # Universal newline support in binary mode
    'rbU', 'rUb', 'Urb', 'brU', 'bUr', 'Ubr',
]
Filename = Union[str, pathlib.Path]
IO: typing.TypeAlias = Union[  # type: ignore[name-defined]
    typing.IO[str],
    typing.IO[bytes],
]


class FileOpenKwargs(typing.TypedDict):
    buffering: int | None
    encoding: str | None
    errors: str | None
    newline: str | None
    closefd: bool | None
    opener: typing.Callable[[str, int], int] | None
