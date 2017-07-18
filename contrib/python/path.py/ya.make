PY_LIBRARY()



# Version: 10.3.1

RESOURCE(
    .dist-info/METADATA      /fs/contrib/python/path.py/.dist-info/METADATA
    .dist-info/top_level.txt /fs/contrib/python/path.py/.dist-info/top_level.txt
)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    path.py
)

END()
