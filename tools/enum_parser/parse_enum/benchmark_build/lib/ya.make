

LIBRARY()

SRCS()

RUN_PYTHON3(
    ${ARCADIA_ROOT}/tools/enum_parser/parse_enum/benchmark_build/lib/gen.py
    --range 1000
    --namespace NHuge
    --enum EHuge 1000
    STDOUT enum_huge.h
)

RUN_PYTHON3(
    ${ARCADIA_ROOT}/tools/enum_parser/parse_enum/benchmark_build/lib/gen.py
    --enum EEnormous 9000
    STDOUT enum_enormous.h
)

GENERATE_ENUM_SERIALIZATION_WITH_HEADER(enum_huge.h)
GENERATE_ENUM_SERIALIZATION_WITH_HEADER(enum_enormous.h)

END()
