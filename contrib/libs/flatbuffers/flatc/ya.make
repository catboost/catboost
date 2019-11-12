

LIBRARY()

LICENSE(
    APACHE2
)

NO_UTIL()

ADDINCL(
    contrib/libs/flatbuffers/include
)

PEERDIR(
    contrib/libs/flatbuffers
)

SRCDIR(
    contrib/libs/flatbuffers/src
)

SRCS(
    code_generators.cpp
    flatc.cpp
    idl_gen_cpp.cpp
    idl_gen_fbs.cpp
    idl_gen_general.cpp
    idl_gen_python.cpp
    idl_gen_go.cpp
    idl_gen_json_schema.cpp
    idl_gen_cpp_yandex_maps_iter.cpp
)

END()
