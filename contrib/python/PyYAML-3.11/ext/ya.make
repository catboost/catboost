LIBRARY()



PEERDIR(contrib/libs/yaml)

NO_WSHADOW()

PY_SRCS(
    TOP_LEVEL
    _yaml.pyx
)

END()
