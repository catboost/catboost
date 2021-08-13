

PY3_PROGRAM(protoc-gen-mypy)

PEERDIR(
    contrib/python/mypy-protobuf
)

PY_MAIN(mypy_protobuf.main:main)

NO_LINT()

END()
