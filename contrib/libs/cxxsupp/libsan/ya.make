LIBRARY()

NO_PLATFORM()
NO_SANITIZE()
NO_SANITIZE_COVERAGE()



BUILTIN_PYTHON(generate_symbolizer.py ${CXX_COMPILER} STDOUT symbolizer.c)

CFLAGS(-fPIC)

SRCS(
    GLOBAL inject.c
    symbolizer.c
)

END()
