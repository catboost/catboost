LIBRARY()



SRCS(
    GLOBAL terminate_handler.cpp
    segv_handler.cpp
)

END()

RECURSE(
    sample
)
