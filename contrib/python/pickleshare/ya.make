LIBRARY(pickleshare)

VERSION(0.6)

LICENSE(
    MIT
)



PY_SRCS(
    TOP_LEVEL
    pickleshare.py
)

PEERDIR(
    contrib/python/path.py
)

NO_LINT()

END()
