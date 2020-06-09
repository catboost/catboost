LIBRARY()



PEERDIR(
    library/cpp/colorizer
    library/cpp/dbg_output
    library/cpp/diff
    library/cpp/json/writer
    library/cpp/testing/common
)

SRCS(
    gtest.cpp
    checks.cpp
    plugin.cpp
    registar.cpp
    tests_data.cpp
    utmain.cpp
)

END()
