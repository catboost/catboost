LIBRARY()



PEERDIR(
    contrib/tools/python3/src
    library/cpp/resource
)

ADDINCL(
    contrib/tools/python3/src/Include
)

CFLAGS(
    -DPy_BUILD_CORE
)

SRCS(
    main.c
    get_py_main.cpp
)

END()
