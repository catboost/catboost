LIBRARY()

VERSION(3.1.0)

LICENSE(BSD-3-Clause)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)



OPENSOURCE_EXPORT_REPLACEMENT(
    CMAKE double-conversion
    CMAKE_TARGET double-conversion::double-conversion
    CONAN double-conversion/3.1.4
)

NO_COMPILER_WARNINGS()

NO_UTIL()

ADDINCL(GLOBAL contrib/libs/double-conversion/include)

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
