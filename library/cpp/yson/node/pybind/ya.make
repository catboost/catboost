LIBRARY()



PEERDIR(
    library/cpp/pybind
    library/cpp/yson/node
)

PYTHON_ADDINCL()

SRCS(
    node.cpp
)

END()
