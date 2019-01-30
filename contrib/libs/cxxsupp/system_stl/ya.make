LIBRARY()



NO_PLATFORM()

ADDINCL(GLOBAL contrib/libs/cxxsupp/system_stl/include)

IF (NOT OS_IOS AND NOT OS_DARWIN)
    LDFLAGS(
        -lgcc_s
        -lstdc++
    )
ELSE()
    LDFLAGS(
        -lc++
    )
ENDIF()

END()
