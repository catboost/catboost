LIBRARY()

LICENSE(Apache-2.0 MIT)

NO_PLATFORM()
NO_SANITIZE()
NO_SANITIZE_COVERAGE()



PYTHON(generate_symbolizer.py ${CXX_COMPILER} STDOUT symbolizer.c)

CFLAGS(-fPIC)

SRCS(
    GLOBAL inject.c
)

END()
