LIBRARY()



ADDINCL(
    GLOBAL contrib/libs/yaml/include
)

NO_COMPILER_WARNINGS()

SRCS(
    src/api.c
    src/dumper.c
    src/emitter.c
    src/loader.c
    src/parser.c
    src/reader.c
    src/scanner.c
    src/writer.c
)

END()
