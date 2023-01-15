LIBRARY()

LICENSE(
    BSD3
)



NO_COMPILER_WARNINGS()

NO_UTIL()

ADDINCL(
    GLOBAL contrib/libs/double-conversion/include
)

SRCS(
    cached-powers.cc
    bignum-dtoa.cc
    double-conversion.cc
    diy-fp.cc
    fixed-dtoa.cc
    strtod.cc
    bignum.cc
    fast-dtoa.cc
)

END()
