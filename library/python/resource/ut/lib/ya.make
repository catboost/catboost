PY23_LIBRARY()



TEST_SRCS(test_simple.py)

PEERDIR(
    library/python/resource
)

RESOURCE(
    qw.txt /qw.txt
    qw.txt /prefix/1.txt
    qw.txt /prefix/2.txt
)

END()
