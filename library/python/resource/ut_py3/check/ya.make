

PY3_PROGRAM()

PY3_SRCS(
    main.py
    test_simple.py
)

PEERDIR(
    library/python/resource
)

RESOURCE(
    qw.txt /qw.txt
    qw.txt /prefix/1.txt
    qw.txt /prefix/2.txt
)

PY3_MAIN(library.python.resource.ut_py3.check.main)

END()

NEED_CHECK()
