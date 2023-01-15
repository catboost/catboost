LIBRARY()



SRCS(
    env.cpp
    env_tmpl.cpp.in
    scope.cpp
)

END()

RECURSE_FOR_TESTS(ut)
