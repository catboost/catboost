TOOL()



SRCS(
    main.cpp
)

INDUCED_DEPS(h+cpp
    util/generic/typetraits.h
    util/generic/singleton.h
    util/generic/string.h
    util/generic/vector.h
    util/generic/map.h
    util/string/cast.h
    util/stream/output.h
)

INDUCED_DEPS(h
    util/generic/serialized_enum.h
)

PEERDIR(
    library/getopt/small
    tools/enum_parser/parse_enum
)

END()
