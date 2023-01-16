PY23_LIBRARY()



RESOURCE_FILES(
    PREFIX library/python/certifi/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

PY_SRCS(
    TOP_LEVEL
    certifi/__init__.py
    certifi/binary.py
    certifi/source.py
)

END()
