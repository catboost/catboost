LIBRARY()



CREATE_BUILDINFO_FOR(buildinfo_data.h)

PEERDIR(
    library/string_utils/base64
)

SRCS(
    sandbox.cpp.in
    build_info.cpp.in
    build_info_static.cpp
)

END()
