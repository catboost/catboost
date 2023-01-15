LIBRARY()



PEERDIR(
    contrib/tools/python3/src
    library/cpp/resource
)

ADDINCL(
    contrib/tools/python3/src/Include
)

SRCS(
    main.c
    get_py_main.cpp
)

END()
