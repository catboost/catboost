LIBRARY()



NO_PLATFORM()

ADDINCL(GLOBAL contrib/libs/cxxsupp/system_stl/include)

IF (NOT OS_IOS AND NOT OS_DARWIN)
    IF (NOT OS_ANDROID)
        LDFLAGS(
            -lgcc_s
        )
    ENDIF()
    LDFLAGS(
        -lstdc++
    )
ELSE()
    LDFLAGS(
        -lc++
    )
ENDIF()

END()
