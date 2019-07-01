PYTEST()



TEST_SRCS(test_fetch.py)

TAG(ya:external)

REQUIREMENTS(network:full)

PEERDIR(
    library/python/resource
    certs
)

END()
