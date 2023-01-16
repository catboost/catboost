

PY23_LIBRARY()

LICENSE(MIT)

VERSION(0.10.2)

PY_SRCS(
    TOP_LEVEL
    toml/__init__.py
    toml/decoder.py
    toml/encoder.py
    toml/ordered.py
    toml/tz.py
)

RESOURCE_FILES(
    PREFIX contrib/python/toml/
    .dist-info/METADATA
    .dist-info/top_level.txt
    toml/__init__.pyi
    toml/decoder.pyi
    toml/encoder.pyi
    toml/ordered.pyi
    toml/tz.pyi
)

NO_LINT()

END()
