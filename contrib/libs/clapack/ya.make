LIBRARY()



NO_JOIN_SRC()
NO_COMPILER_WARNINGS()

IF (HAVE_MKL)
    PEERDIR(contrib/libs/intel/mkl)
ELSE ()
    PEERDIR(
        contrib/libs/clapack/part1
        contrib/libs/clapack/part2
    )
ENDIF ()

END()
