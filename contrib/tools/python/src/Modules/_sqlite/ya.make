LIBRARY()



LICENSE(Python-2.0)

PEERDIR(
    contrib/libs/sqlite3
)

ADDINCL(
    contrib/libs/sqlite3
)

PYTHON_ADDINCL()

NO_COMPILER_WARNINGS()

NO_RUNTIME()

SRCS(
    cache.c
    connection.c
    cursor.c
    microprotocols.c
    module.c
    prepare_protocol.c
    row.c
    statement.c
    util.c
)

PY_REGISTER(_sqlite3)

END()
