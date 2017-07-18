TOOL(ragel6)



IF (OS_CYGWIN)
    #use default
ELSE ()
    ALLOCATOR(J)
ENDIF ()

NO_UTIL()
NO_COMPILER_WARNINGS()

#TODO - force OPTIMIZE()
IF (MSVC)
    #TODO
ELSE ()
    CFLAGS(-O2)
ENDIF()

PEERDIR(
    contrib/tools/ragel5/aapl
)

JOIN_SRCS(
    all_src1.cpp
    rubycodegen.cpp
    rubytable.cpp
    cdfflat.cpp
    cdgoto.cpp
    common.cpp
    csflat.cpp
    cssplit.cpp
    parsedata.cpp
)

JOIN_SRCS(
    all_src2.cpp
    rlparse.cpp
    cdflat.cpp
    cdsplit.cpp
    csgoto.cpp
    fsmstate.cpp
    main.cpp
    xmlcodegen.cpp
)

JOIN_SRCS(
    all_src3.cpp
    cdcodegen.cpp
    cdftable.cpp
    cdtable.cpp
    csfgoto.cpp
    csipgoto.cpp
    fsmap.cpp
    fsmmin.cpp
    fsmattach.cpp
    csfflat.cpp
    rubyflat.cpp
)

JOIN_SRCS(
    all_src4.cpp
    rubyfflat.cpp
    cdfgoto.cpp
    cdipgoto.cpp
    cscodegen.cpp
    csftable.cpp
    cstable.cpp
    fsmbase.cpp
    gendata.cpp
    rubyftable.cpp
)

JOIN_SRCS(
    all_src5.cpp
    dotcodegen.cpp
    fsmgraph.cpp
    inputdata.cpp
    parsetree.cpp
    redfsm.cpp
    javacodegen.cpp
    rbxgoto.cpp
)

SRCS(
    rlscan.cpp
)

END()
