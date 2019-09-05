PROGRAM()



NO_UTIL()
NO_RUNTIME()
NO_COMPILER_WARNINGS()

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
