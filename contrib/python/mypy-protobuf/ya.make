PY3_LIBRARY()



VERSION(2.9)

LICENSE(Apache-2.0)

PEERDIR(
    contrib/libs/protobuf/python/google_lib
)

PY_SRCS(
    TOP_LEVEL
    mypy_protobuf/__init__.py
    mypy_protobuf/extensions_pb2.py
    mypy_protobuf/main.py
)

NO_LINT()

RESOURCE_FILES(
    PREFIX contrib/python/mypy-protobuf/
    .dist-info/METADATA
    .dist-info/entry_points.txt
    .dist-info/top_level.txt
)

END()

RECURSE(
    bin
)
