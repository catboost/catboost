PACKAGE()

LICENSE(BSD-3-Clause)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)



GENERATE_PY_PROTOS(contrib/libs/protobuf/src/google/protobuf/descriptor.proto)

END()

RECURSE(
    google_lib
)
