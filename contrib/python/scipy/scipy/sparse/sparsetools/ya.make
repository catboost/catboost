

PY23_LIBRARY()

LICENSE(BSD-3-Clause)

PEERDIR(
    contrib/python/numpy
)

NO_COMPILER_WARNINGS()

ADDINCLSELF()

SRCS(
    bsr.cxx
    csc.cxx
    csr.cxx
    other.cxx
    sparsetools.cxx
)

PY_REGISTER(scipy.sparse._sparsetools)

END()
