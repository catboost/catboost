PY3TEST()



PEERDIR(
    build/plugins/lib/nots/semver
)

TEST_SRCS(
    test_version_range.py
    test_version.py
)

END()
