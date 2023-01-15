LIBRARY()



# http://www.cpan.org/src/5.0/perl-5.14.4.tar.gz

# ./Configure -Dccflags="-O2 -g [-fPIC]" -Dprefix=$HOME/perl[_pic] -Dusethreads -Duselargefiles -Uuseshrplib -d -e

IF (OS_LINUX)
    IF (NOT PIC)
        FROM_SANDBOX(81213253 OUT libperl.a)
    ELSE()
        FROM_SANDBOX(81212225 OUT libperl.a)
    ENDIF()
ELSEIF (OS_FREEBSD)
    IF (NOT PIC)
        FROM_SANDBOX(81393842 OUT libperl.a)
    ELSE()
        FROM_SANDBOX(81393930 OUT libperl.a)
    ENDIF()
ELSEIF (OS_DARWIN)
    IF (NOT PIC)
        FROM_SANDBOX(81402066 OUT libperl.a)
    ELSE()
        FROM_SANDBOX(81402131 OUT libperl.a)
    ENDIF()
ENDIF()

ADDINCL(GLOBAL build/platform/perl/5.14.4)

END()
