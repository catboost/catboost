LIBRARY()



SRCS(
    env.cpp
    env_tmpl.cpp.in
    network.cpp
    scope.cpp
)

END()

RECURSE_FOR_TESTS(ut)
