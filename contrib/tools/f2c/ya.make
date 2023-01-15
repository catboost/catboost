PROGRAM()



NO_UTIL()
NO_RUNTIME()
NO_COMPILER_WARNINGS()

IF (OS_WINDOWS)
    CFLAGS(
          -D_WIN32
          -D_CONSOLE
          -DWIN32_LEAN_AND_MEAN
          -DNOMINMAX
    )

    ALLOCATOR(LF)  # by some reason f2c crashes on Windows with default allocator
ENDIF()

SRCDIR(
    contrib/tools/f2c/src
)

SRCS(
    main.c
    init.c
    gram.c
    lex.c
    proc.c
    equiv.c
    data.c
    format.c
    expr.c
    exec.c
    intr.c
    io.c
    misc.c
    error.c
    mem.c
    names.c
    output.c
    p1output.c
    pread.c
    put.c
    putpcc.c
    vax.c
    formatdata.c
    parse_args.c
    niceprintf.c
    cds.c
    sysdep.c
    version.c
)

END()
