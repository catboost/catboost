PY23_LIBRARY(pickleshare)

LICENSE(MIT)

VERSION(0.7.5)



PY_SRCS(
    TOP_LEVEL
    pickleshare.py
)

PEERDIR(
    contrib/python/path.py
)

NO_LINT()

END()
