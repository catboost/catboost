PY23_LIBRARY()

LICENSE(BSD-3-Clause)



# Use protoc to generate _pb2 files.
# As C++ implementation is already compiled into contrib/libs/protobuf,
# NO_OPTIMIZE_PY_PROTOS is legitimate

NO_OPTIMIZE_PY_PROTOS()

PEERDIR(contrib/libs/protobuf)

PY_NAMESPACE(.)
PROTO_NAMESPACE(contrib/libs/protoc/src)
SRCDIR(contrib/libs/protoc/src)
PY_SRCS(
    google/protobuf/compiler/plugin.proto
)

END()
