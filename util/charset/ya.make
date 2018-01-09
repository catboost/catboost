LIBRARY()



NO_UTIL()

JOIN_SRCS(
    all_charset.cpp
    generated/unidata.cpp
    recode_result.cpp
    unicode_table.cpp
    unidata.cpp
    utf8.cpp
    wide.cpp
)

IF (ARCH_X86_64)
    PEERDIR(
        util/charset/sse41
    )
ELSE()
    PEERDIR(
        util/charset/scalar
    )
ENDIF()

END()
