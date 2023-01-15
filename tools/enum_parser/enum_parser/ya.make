PROGRAM()



SRCS(
    main.cpp
)

INDUCED_DEPS(h+cpp
    ${ARCADIA_ROOT}/util/generic/typetraits.h
    ${ARCADIA_ROOT}/util/generic/singleton.h
    ${ARCADIA_ROOT}/util/generic/string.h
    ${ARCADIA_ROOT}/util/generic/vector.h
    ${ARCADIA_ROOT}/util/generic/map.h
    ${ARCADIA_ROOT}/util/string/cast.h
    ${ARCADIA_ROOT}/util/stream/output.h
    ${ARCADIA_ROOT}/tools/enum_parser/enum_serialization_runtime/enum_runtime.h
    ${ARCADIA_ROOT}/tools/enum_parser/enum_parser/stdlib_deps.h
)

INDUCED_DEPS(h
    ${ARCADIA_ROOT}/util/generic/serialized_enum.h
)

PEERDIR(
    library/cpp/getopt/small
    tools/enum_parser/parse_enum
)

END()
