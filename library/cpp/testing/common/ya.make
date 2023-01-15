LIBRARY()



SRCS(
    env.cpp
    env_tmpl.cpp.in
)

END()

RECURSE_FOR_TESTS(ut)
