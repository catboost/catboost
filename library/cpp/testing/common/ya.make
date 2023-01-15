LIBRARY()



SRCS(
    env.cpp
    env_tmpl.cpp.in
    network.cpp
    probe.cpp
    scope.cpp
)

PEERDIR(
    library/cpp/json
)

END()

RECURSE_FOR_TESTS(ut)
