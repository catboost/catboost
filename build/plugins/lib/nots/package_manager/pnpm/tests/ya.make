PY23_TEST()



TEST_SRCS(
    lockfile.py
    workspace.py
)

PEERDIR(
    build/plugins/lib/nots/package_manager/base
    build/plugins/lib/nots/package_manager/pnpm
)

END()
