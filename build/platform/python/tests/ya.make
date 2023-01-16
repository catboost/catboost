PY3TEST()



IF (OS_DARWIN)
    SIZE(LARGE)

    TAG(
        ya:fat
        ya:force_sandbox ya:exotic_platform
    )
ENDIF()

PY_SRCS(
    testlib.py
)

TEST_SRCS(
    test_common.py
)

PEERDIR(
    build/platform/python/python27
    build/platform/python/python34
    build/platform/python/python35
    build/platform/python/python36
    build/platform/python/python37
    build/platform/python/python38
    build/platform/python/python39
    build/platform/python/python310
)

END()
