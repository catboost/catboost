LIBRARY()

WITHOUT_LICENSE_TEXTS()

LICENSE(YandexOpen)

OWNER(
    g:contrib
    g:cpp-contrib
    somov
)

NO_PLATFORM()

ADDINCL(GLOBAL contrib/libs/cxxsupp/system_stl/include)

IF (OS_IOS OR OS_DARWIN)
    LDFLAGS(-lc++)
ELSEIF (OS_ANDROID)
    IF (STATIC_STL)
        LDFLAGS(-l:libc++.a)
    ELSE()
        LDFLAGS(-lc++)
    ENDIF()
ELSE()
    CFLAGS(GLOBAL -DLIBCXX_BUILDING_LIBGCC)
    LDFLAGS(-lgcc_s)

    # libatomic.a is needed in order to make atomic operations work
    LDFLAGS(-l:libatomic.a)

    IF (STATIC_STL)
        LDFLAGS(-l:libstdc++.a)
    ELSE()
        LDFLAGS(-lstdc++)
    ENDIF()
ENDIF()

END()
