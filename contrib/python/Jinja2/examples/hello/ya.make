PY_PROGRAM(jinja2-hello)



PEERDIR(
    contrib/python/Jinja2
)

PY_SRCS(
    NAMESPACE hello
    __init__.py
    __main__.py
)

RESOURCE_FILES(
    PREFIX contrib/python/Jinja2/examples/hello/
    templates/hello
)

END()
