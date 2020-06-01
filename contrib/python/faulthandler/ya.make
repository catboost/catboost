PY_LIBRARY()



VERSION(3.2)

LICENSE(BSD)

NO_LINT()

SRCS(
    faulthandler.c
    traceback.c
)

NO_COMPILER_WARNINGS()

PY_REGISTER(faulthandler)

RESOURCE_FILES(
    PREFIX contrib/python/faulthandler/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

CFLAGS(
    -DUSE_SIGINFO
)

END()
