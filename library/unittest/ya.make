LIBRARY()



PEERDIR(
    library/cpp/colorizer
    library/dbg_output
    library/cpp/diff
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
