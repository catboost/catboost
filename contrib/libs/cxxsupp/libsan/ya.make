LIBRARY()

WITHOUT_LICENSE_TEXTS()

LICENSE(YandexOpen)

NO_PLATFORM()

NO_SANITIZE()

NO_SANITIZE_COVERAGE()

OWNER(somov)

PYTHON(
    generate_symbolizer.py ${CXX_COMPILER}
    STDOUT symbolizer.c
)

CFLAGS(-fPIC)

SRCS(
    GLOBAL inject.c
)

END()
