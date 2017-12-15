LIBRARY()



NO_WSHADOW()

INCLUDE(${ARCADIA_ROOT}/contrib/tools/python/pyconfig.inc)

ADDINCL(GLOBAL ${PYTHON_SRC_DIR}/Include)

SRCDIR(${PYTHON_SRC_DIR}/Include)

END()
