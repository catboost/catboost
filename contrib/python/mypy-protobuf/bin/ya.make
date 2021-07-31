PY3_PROGRAM(protoc-gen-mypy)



PEERDIR(
    contrib/python/six
    contrib/libs/protobuf/python/google_lib
)

SRCDIR(
    contrib/python/mypy-protobuf
)

PY_SRCS(
    __main__.py
)

VERSION(1.16)
LICENSE(Apache-2.0)

NO_LINT()

END()
