LIBRARY()



PEERDIR(
    library/pybind
    library/yson/node
)

PYTHON_ADDINCL()

SRCS(
    node.cpp
)

END()
