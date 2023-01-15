PY23_LIBRARY()



NO_COMPILER_WARNINGS()


ADDINCL(
    contrib/python/scipy/scipy/special
)

PEERDIR(
    contrib/python/numpy
)

SRCS(
    besselpoly.c
    double2.h
    fsolve.c
    gammaincinv.c
    gammasgn.c
    misc.h
    poch.c
    struve.c
)

END()
