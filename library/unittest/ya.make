LIBRARY()



PEERDIR(
    library/colorizer
    library/dbg_output
    library/diff
    library/json/writer
)

SRCS(
    gtest.cpp
    checks.cpp
    plugin.cpp
    registar.cpp
    tests_data.cpp
    utmain.cpp
    env.cpp.in
)

END()
