LIBRARY()



NO_PLATFORM()

IF (USE_PERL_SYSTEM)
    IF (PERL_SDK == "ubuntu-12")
        PEERDIR(build/platform/perl/5.14)
    ELSEIF (PERL_SDK == "ubuntu-14")
        PEERDIR(build/platform/perl/5.18)
    ELSEIF (PERL_SDK == "ubuntu-16")
        PEERDIR(build/platform/perl/5.22)
    ELSEIF (PERL_SDK == "ubuntu-18")
        PEERDIR(build/platform/perl/5.26)
    ELSEIF (PERL_SDK == "ubuntu-20")
        PEERDIR(build/platform/perl/5.30)
    ELSE()
        MESSAGE(FATAL_ERROR "Building against system perl is not supported on ${PERL_SDK}")
    ENDIF()

ELSE()

    MESSAGE(FATAL_ERROR "There is no perl ready for static linkage. Try using the system one.")

ENDIF()

END()
