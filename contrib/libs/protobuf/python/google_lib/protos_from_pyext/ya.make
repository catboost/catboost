PY23_LIBRARY()

LICENSE(BSD-3-Clause)



PY_NAMESPACE(.)
PROTO_NAMESPACE(contrib/libs/protobuf/python)
SRCDIR(contrib/libs/protobuf/python)
PY_SRCS(
    google/protobuf/pyext/python.proto
)

END()
