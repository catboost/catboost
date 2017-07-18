# Generated automatically by ../modules.py

ADDINCL(contrib/libs/libbz2)
ADDINCL(contrib/libs/sqlite3)
ADDINCL(contrib/libs/zlib)
PEERDIR(contrib/libs/libbz2)
PEERDIR(contrib/libs/openssl)
PEERDIR(contrib/libs/sqlite3)
PEERDIR(contrib/libs/zlib)
SRCS(Modules/_bisectmodule.c
     Modules/_collectionsmodule.c
     Modules/_csv.c
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
     Modules/_sqlite/cache.c
     Modules/_sqlite/connection.c
     Modules/_sqlite/cursor.c
     Modules/_sqlite/microprotocols.c
     Modules/_sqlite/module.c
     Modules/_sqlite/prepare_protocol.c
     Modules/_sqlite/row.c
     Modules/_sqlite/statement.c
     Modules/_sqlite/util.c
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

IF (ARCH_X86_64 OR ARCH_I386)
    IF (OS_CYGWIN)
        #TODO
    ELSE()
        PEERDIR(contrib/tools/python/src/Modules/_ctypes/libffi)

        IF (UNIX)
            ADDINCL(${PYTHON_SRC_DIR}/Modules/_ctypes/libffi/include)
        ELSEIF (WIN32)
            ADDINCL(${PYTHON_SRC_DIR}/Modules/_ctypes/libffi_msvc)
        ENDIF()

        SRCS(
            Modules/_ctypes/_ctypes.c
            Modules/_ctypes/callbacks.c
            Modules/_ctypes/callproc.c
            Modules/_ctypes/cfield.c
            Modules/_ctypes/malloc_closure.c
            Modules/_ctypes/stgdict.c
        )
    ENDIF()
ENDIF()

IF (FREEBSD OR LINUX)
    IF (NOT MUSL)
        SRCS(Modules/nismodule.c)

        IF (LINUX)
            EXTRALIBS(nsl)
        ENDIF ()

        CFLAGS(-D_HAVE_NIS=1)
    ENDIF ()
ENDIF ()

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

IF(DARWIN OR OS_CYGWIN)
    SRCS(Modules/_multiprocessing/multiprocessing.c
         Modules/_multiprocessing/semaphore.c
         Modules/_multiprocessing/socket_connection.c
    )
ENDIF(DARWIN OR OS_CYGWIN)

IF (UNIX)
    ADDINCL(${PYTHON_SRC_DIR}/Modules/expat)
    SRCS(Modules/_elementtree.c
         Modules/_ssl.c
         Modules/cryptmodule.c
         Modules/expat/xmlparse.c
         Modules/expat/xmlrole.c
         Modules/expat/xmltok.c
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
    ELSEIF (NOT DARWIN)
        EXTRALIBS(crypt)
    ENDIF()

    CFLAGS(-DHAVE_EXPAT_CONFIG_H=1)
    CFLAGS(-DHAVE_EXPAT_CONFIG_H=1 -DUSE_PYEXPAT_CAPI)
ENDIF(UNIX)

IF (WIN32)
    ADDINCL(${PYTHON_SRC_DIR}/Modules/expat)

    SRCS(Modules/_elementtree.c
         Modules/_multiprocessing/multiprocessing.c
         Modules/_multiprocessing/pipe_connection.c
         Modules/_multiprocessing/semaphore.c
         Modules/_multiprocessing/socket_connection.c
         Modules/_multiprocessing/win32_functions.c
         Modules/_ssl.c
         Modules/expat/xmlparse.c
         Modules/expat/xmlrole.c
         Modules/expat/xmltok.c
         Modules/posixmodule.c
         Modules/pyexpat.c
         Modules/selectmodule.c
         Modules/socketmodule.c
    )
    EXTRALIBS(crypt32.lib ws2_32.lib)

    CFLAGS(-DHAVE_EXPAT_CONFIG_H=1 -DHAVE_MEMMOVE)
    CFLAGS(-DHAVE_EXPAT_CONFIG_H=1 -DHAVE_MEMMOVE -DUSE_PYEXPAT_CAPI)
    CFLAGS(-DEAI_ADDRFAMILY=1 -DHAVE_SOCKADDR_STORAGE=1 -DHAVE_ADDRINFO=1)
    CFLAGS(/D_WIN64 /D_M_X64)
ENDIF(WIN32)
