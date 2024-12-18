## Typing stubs for protobuf

This is a [PEP 561](https://peps.python.org/pep-0561/)
type stub package for the [`protobuf`](https://github.com/protocolbuffers/protobuf) package.
It can be used by type-checking tools like
[mypy](https://github.com/python/mypy/),
[pyright](https://github.com/microsoft/pyright),
[pytype](https://github.com/google/pytype/),
[Pyre](https://pyre-check.org/),
PyCharm, etc. to check code that uses `protobuf`. This version of
`types-protobuf` aims to provide accurate annotations for
`protobuf~=5.28.3`.

Partially generated using [mypy-protobuf==3.6.0](https://github.com/nipunn1313/mypy-protobuf/tree/v3.6.0) and libprotoc 27.2 on [protobuf v28.3](https://github.com/protocolbuffers/protobuf/releases/tag/v28.3) (python `protobuf==5.28.3`).

This stub package is marked as [partial](https://peps.python.org/pep-0561/#partial-stub-packages).
If you find that annotations are missing, feel free to contribute and help complete them.


This package is part of the [typeshed project](https://github.com/python/typeshed).
All fixes for types and metadata should be contributed there.
See [the README](https://github.com/python/typeshed/blob/main/README.md)
for more details. The source for this package can be found in the
[`stubs/protobuf`](https://github.com/python/typeshed/tree/main/stubs/protobuf)
directory.

This package was tested with
mypy 1.13.0,
pyright 1.1.389,
and pytype 2024.10.11.
It was generated from typeshed commit
[`f7c6acde6e1718b5f8748815200e95ef05d96d32`](https://github.com/python/typeshed/commit/f7c6acde6e1718b5f8748815200e95ef05d96d32).