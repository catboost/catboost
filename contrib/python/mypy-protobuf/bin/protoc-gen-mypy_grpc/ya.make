

PY3_PROGRAM(protoc-gen-mypy_grpc)

PEERDIR(
    contrib/python/mypy-protobuf
)

PY_MAIN(mypy_protobuf.main:grpc)

NO_LINT()

END()
