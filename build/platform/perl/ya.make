RESOURCES_LIBRARY()



IF (USE_PERL_SYSTEM)
    IF (OS_SDK == "ubuntu-12")
        DECLARE_EXTERNAL_RESOURCE(SYSTEM_PERL sbr:337748278)
    ELSEIF (OS_SDK == "ubuntu-14")
        DECLARE_EXTERNAL_RESOURCE(SYSTEM_PERL sbr:1655582861)
    ELSEIF (OS_SDK == "ubuntu-16")
        DECLARE_EXTERNAL_RESOURCE(SYSTEM_PERL sbr:323251590)
    ELSEIF (OS_SDK == "ubuntu-18")
        DECLARE_EXTERNAL_RESOURCE(SYSTEM_PERL sbr:616700320)
    ELSEIF (OS_SDK == "ubuntu-20")
        DECLARE_EXTERNAL_RESOURCE(SYSTEM_PERL sbr:2001114055)
    ELSE()
        MESSAGE(FATAL_ERROR "Building against system perl is not supported on ${OS_SDK}")
    ENDIF()

    IF (PERL_INCLUDE)
        CFLAGS(GLOBAL $PERL_INCLUDE)
    ENDIF()

    CFLAGS(GLOBAL -I$PERL_ARCHLIB/CORE)

    IF (PERL_LIBS)
        LDFLAGS(-L${PERL_LIBS})
    ENDIF()

    IF (NOT OS_WINDOWS)
        LDFLAGS(-lperl)
    ELSE()
        LDFLAGS(perl.lib)
    ENDIF()

ELSE()

    MESSAGE(FATAL_ERROR "There is no perl ready for static linkage. Try using the system one.")

ENDIF()

CFLAGS(GLOBAL -DUSE_PERL)

END()
