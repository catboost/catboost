LIBRARY()



SRCS(
    env.cpp
    network.cpp
    probe.cpp
    scope.cpp
)

PEERDIR(
    library/cpp/json
)

END()

RECURSE_FOR_TESTS(ut)
