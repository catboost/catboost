PY23_LIBRARY(pickleshare)

VERSION(0.7.5)

LICENSE(MIT)



PY_SRCS(
    TOP_LEVEL
    pickleshare.py
)

PEERDIR(
    contrib/python/path.py
)

NO_LINT()

END()
