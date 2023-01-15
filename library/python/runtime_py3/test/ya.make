PY3TEST()



DEPENDS(library/python/runtime_py3/test/traceback)

PY_SRCS(
    TOP_LEVEL
    resources/__init__.py
    resources/submodule/__init__.py
)

TEST_SRCS(
    test_metadata.py
    test_resources.py
    test_traceback.py
)

RESOURCE_FILES(
    PREFIX library/python/runtime_py3/test/
    .dist-info/METADATA
    .dist-info/RECORD
    .dist-info/entry_points.txt
    .dist-info/top_level.txt
    resources/foo.txt
    resources/submodule/bar.txt
)

END()

RECURSE_FOR_TESTS(traceback)
