LIBRARY()



PEERDIR(
    library/cpp/pybind
    library/yson/node
)

PYTHON_ADDINCL()

SRCS(
    node.cpp
)

END()
