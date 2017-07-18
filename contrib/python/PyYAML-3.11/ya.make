LIBRARY(yaml)



IF (NOT MSVC)
    PEERDIR(contrib/python/PyYAML-3.11/ext)
ENDIF()

PY_SRCS(
    TOP_LEVEL
    yaml/__init__.py
    yaml/composer.py
    yaml/constructor.py
    yaml/cyaml.py
    yaml/dumper.py
    yaml/emitter.py
    yaml/error.py
    yaml/events.py
    yaml/loader.py
    yaml/nodes.py
    yaml/parser.py
    yaml/reader.py
    yaml/representer.py
    yaml/resolver.py
    yaml/scanner.py
    yaml/serializer.py
    yaml/tokens.py
)

SRCDIR(
    contrib/python/PyYAML-3.11/lib
)

NO_LINT()

END()
