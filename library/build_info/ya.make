LIBRARY()



DEFAULT(SANDBOX_TASK_ID 0)
DEFAULT(KOSHER_SVN_VERSION "")

CREATE_BUILDINFO_FOR(buildinfo_data.h)

PEERDIR(
    library/cpp/string_utils/base64
)

SRCS(
    sandbox.cpp.in
    build_info.cpp.in
    build_info_static.cpp
)

END()
