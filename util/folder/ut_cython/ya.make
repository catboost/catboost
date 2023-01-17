PY23_TEST()


SUBSCRIBER(g:util-subscribers)

SRCDIR(util/folder)

PY_SRCS(
    NAMESPACE util.folder
    path_ut.pyx
)

TEST_SRCS(
    test_folder.py
)

END()
