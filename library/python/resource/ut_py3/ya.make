

# This is temporary solution to test Python3 compatibility
# Eventually we'll support PY23TEST as module for ../ut that will build and run in both Python2 and Python3 modes
# But for thsi we should enable PYTEST macro which (among others) brings dependency on library/python/resource

EXECTEST()

RUN(check)

DEPENDS(library/python/resource/ut_py3/check)

TEST_CWD(library/python/resource/ut_py3/check)

END()

NEED_CHECK()
