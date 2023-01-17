PY23_LIBRARY()



PY_SRCS(
    __init__.py
    constants.py
    lockfile.py
    node_modules_bundler.py
    package_json.py
    package_manager.py
    utils.py
)

PEERDIR(
    contrib/python/six
)

END()

RECURSE_FOR_TESTS(
    tests
)
