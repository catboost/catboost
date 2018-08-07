LIBRARY()



SRCS(
    hash.cpp
)

PY_SRCS(
    TOP_LEVEL
    cityhash.pyx
)

PY_REGISTER(cityhash)

END()

NEED_CHECK()
