LIBRARY()



NO_PLATFORM()

ADDINCL(GLOBAL contrib/libs/cxxsupp/system_stl/include)

IF (OS_IOS OR OS_DARWIN)
    LDFLAGS(
        -lc++
    )
ELSEIF (OS_ANDROID)
    IF (STATIC_STL)
        LDFLAGS(
            -l:libc++.a
        )
    ELSE()
        LDFLAGS(
            -lc++
        )
    ENDIF()
ELSE()
    LDFLAGS(
        -lgcc_s
    )
    IF (STATIC_STL)
        LDFLAGS(
            -l:libstdc++.a
        )
    ELSE()
        LDFLAGS(
            -lstdc++
        )
    ENDIF()
ENDIF()

END()
