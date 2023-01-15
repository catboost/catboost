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

IF (ARCH_X86_64 AND NOT DISABLE_INSTRUCTION_SETS)
    SRC_CPP_SSE41(wide_sse41.cpp)
ELSE()
    SRC(
        wide_sse41.cpp
        -DSSE41_STUB
    )
ENDIF()

END()
