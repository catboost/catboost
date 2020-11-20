# Generated automatically by ../modules.py

ADDINCL(contrib/libs/expat)
ADDINCL(contrib/libs/libbz2)
ADDINCL(contrib/libs/zlib)
PEERDIR(contrib/libs/expat)
PEERDIR(contrib/libs/libbz2)
PEERDIR(contrib/libs/openssl)
PEERDIR(contrib/libs/zlib)
SRCS(Modules/_bisectmodule.c
     Modules/_collectionsmodule.c
     Modules/_csv.c
     Modules/_elementtree.c
     Modules/_functoolsmodule.c
     Modules/_hashopenssl.c
     Modules/_heapqmodule.c
     Modules/_hotshot.c
     Modules/_io/_iomodule.c
     Modules/_io/bufferedio.c
     Modules/_io/bytesio.c
     Modules/_io/fileio.c
     Modules/_io/iobase.c
     Modules/_io/stringio.c
     Modules/_io/textio.c
     Modules/_json.c
     Modules/_localemodule.c
     Modules/_lsprof.c
     Modules/_math.c
     Modules/_randommodule.c
     Modules/_struct.c
     Modules/arraymodule.c
     Modules/audioop.c
     Modules/binascii.c
     Modules/bz2module.c
     Modules/cPickle.c
     Modules/cStringIO.c
     Modules/cjkcodecs/multibytecodec.c
     Modules/cmathmodule.c
     Modules/datetimemodule.c
     Modules/future_builtins.c
     Modules/itertoolsmodule.c
     Modules/mathmodule.c
     Modules/md5.c
     Modules/md5module.c
     Modules/mmapmodule.c
     Modules/operator.c
     Modules/parsermodule.c
     Modules/rotatingtree.c
     Modules/sha256module.c
     Modules/sha512module.c
     Modules/shamodule.c
     Modules/stropmodule.c
     Modules/timemodule.c
     Modules/unicodedata.c
     Modules/zlibmodule.c)

CFLAGS(-DBUILD_ZLIB)

IF (ARCH_X86_64 OR ARCH_I386 OR ARCH_PPC64LE OR ARCH_AARCH64)
    IF (OS_CYGWIN)
        #TODO
    ELSE()
        PEERDIR(contrib/restricted/libffi)

        ADDINCL(contrib/restricted/libffi/include)

        SRCS(
            Modules/_ctypes/_ctypes.c
            Modules/_ctypes/callbacks.c
            Modules/_ctypes/callproc.c
            Modules/_ctypes/cfield.c
            Modules/_ctypes/stgdict.c
        )
    ENDIF()
ENDIF()

IF(FREEBSD)
    IF(FREEBSD_VER GREATER 7)
        SRCS(Modules/_multiprocessing/multiprocessing.c
             Modules/_multiprocessing/semaphore.c
             Modules/_multiprocessing/socket_connection.c)
    ENDIF(FREEBSD_VER GREATER 7)
    IF(FREEBSD_VER LESS 8)
        SRCS(Modules/_multiprocessing/multiprocessing.c
             Modules/_multiprocessing/socket_connection.c)
    ENDIF(FREEBSD_VER LESS 8)
ENDIF(FREEBSD)

IF(LINUX)
    SRCS(Modules/_multiprocessing/multiprocessing.c
         Modules/_multiprocessing/semaphore.c
         Modules/_multiprocessing/socket_connection.c
         Modules/spwdmodule.c
    )
ENDIF(LINUX)

IF(DARWIN OR OS_CYGWIN OR OS_IOS)
    SRCS(Modules/_multiprocessing/multiprocessing.c
         Modules/_multiprocessing/semaphore.c
         Modules/_multiprocessing/socket_connection.c
    )
ENDIF(DARWIN OR OS_CYGWIN OR OS_IOS)

IF (OS_DARWIN)
    SRCS(
        Mac/Modules/_scproxy.c
    )
ENDIF()

IF (UNIX)
    SRCS(Modules/_ssl.c
         Modules/cryptmodule.c
         Modules/fcntlmodule.c
         Modules/grpmodule.c
         Modules/posixmodule.c
         Modules/pwdmodule.c
         Modules/pyexpat.c
         Modules/resource.c
         Modules/selectmodule.c
         Modules/socketmodule.c
         Modules/syslogmodule.c
         Modules/termios.c
    )

    IF (OS_CYGWIN)
        #PASS
    ELSEIF (NOT DARWIN AND NOT OS_IOS)
        EXTRALIBS(crypt)
    ENDIF()

    # bpo-30947: Python uses best available entropy sources to
    # call XML_SetHashSalt(), expat entropy sources are not needed
    CFLAGS(-DHAVE_EXPAT_CONFIG_H -DXML_POOR_ENTROPY -DUSE_PYEXPAT_CAPI)
ENDIF(UNIX)

IF (OS_WINDOWS)
    SRCS(Modules/_multiprocessing/multiprocessing.c
         Modules/_multiprocessing/pipe_connection.c
         Modules/_multiprocessing/semaphore.c
         Modules/_multiprocessing/socket_connection.c
         Modules/_multiprocessing/win32_functions.c
         Modules/_ssl.c
         Modules/posixmodule.c
         Modules/pyexpat.c
         Modules/selectmodule.c
         Modules/socketmodule.c
    )
    LDFLAGS(crypt32.lib ws2_32.lib)

    CFLAGS(-DHAVE_EXPAT_CONFIG_H -DXML_POOR_ENTROPY -DHAVE_MEMMOVE -DUSE_PYEXPAT_CAPI)
    CFLAGS(-DEAI_ADDRFAMILY=1 -DHAVE_SOCKADDR_STORAGE=1 -DHAVE_ADDRINFO=1)
    IF (ARCH_X86_64)
        CFLAGS(/D_WIN64 /D_M_X64)
    ENDIF()
ENDIF(WIN32)
