LIBRARY()



LICENSE(Python-2.0)

PEERDIR(
    ADDINCL contrib/libs/expat
    ADDINCL contrib/libs/libbz2
    ADDINCL contrib/libs/openssl
    ADDINCL contrib/libs/zlib
    contrib/restricted/libffi
)

ADDINCL(
    contrib/restricted/libffi/include
    contrib/tools/python3/src/Include
    contrib/tools/python3/src/Modules
    contrib/tools/python3/src/Modules/_decimal/libmpdec
    contrib/tools/python3/src/PC
)

NO_COMPILER_WARNINGS()

NO_RUNTIME()

SRCS(
    _abc.c
    _asynciomodule.c
    _bisectmodule.c
    _blake2/blake2b_impl.c
    _blake2/blake2module.c
    _blake2/blake2s_impl.c
    _bz2module.c
    _codecsmodule.c
    _collectionsmodule.c
    _contextvarsmodule.c
    _csv.c
    _ctypes/_ctypes.c
    _ctypes/callbacks.c
    _ctypes/callproc.c
    _ctypes/cfield.c
    _ctypes/stgdict.c
    _datetimemodule.c
    _decimal/_decimal.c
    _decimal/libmpdec/basearith.c
    _decimal/libmpdec/constants.c
    _decimal/libmpdec/context.c
    _decimal/libmpdec/convolute.c
    _decimal/libmpdec/crt.c
    _decimal/libmpdec/difradix2.c
    _decimal/libmpdec/fnt.c
    _decimal/libmpdec/fourstep.c
    _decimal/libmpdec/io.c
    _decimal/libmpdec/memory.c
    _decimal/libmpdec/mpdecimal.c
    _decimal/libmpdec/numbertheory.c
    _decimal/libmpdec/sixstep.c
    _decimal/libmpdec/transpose.c
    _elementtree.c
    _hashopenssl.c
    _heapqmodule.c
    _json.c
    _localemodule.c
    _lsprof.c
    _math.c
    _multiprocessing/multiprocessing.c
    _multiprocessing/semaphore.c
    _opcode.c
    _operator.c
    _pickle.c
    _queuemodule.c
    _randommodule.c
    _sha3/sha3module.c
    _sre.c
    _ssl.c
    _stat.c
    _struct.c
    _tracemalloc.c
    _weakref.c
    _xxtestfuzz/_xxtestfuzz.c
    _xxtestfuzz/fuzzer.c
    arraymodule.c
    atexitmodule.c
    audioop.c
    binascii.c
    cjkcodecs/_codecs_cn.c
    cjkcodecs/_codecs_hk.c
    cjkcodecs/_codecs_iso2022.c
    cjkcodecs/_codecs_jp.c
    cjkcodecs/_codecs_kr.c
    cjkcodecs/_codecs_tw.c
    cjkcodecs/multibytecodec.c
    cmathmodule.c
    errnomodule.c
    faulthandler.c
    getbuildinfo.c
    hashtable.c
    itertoolsmodule.c
    mathmodule.c
    md5module.c
    parsermodule.c
    pyexpat.c
    rotatingtree.c
    selectmodule.c
    sha1module.c
    sha256module.c
    sha512module.c
    socketmodule.c
    symtablemodule.c
    unicodedata.c
    zlibmodule.c
)

IF (OS_WINDOWS)
    SRCS(
        _winapi.c
        overlapped.c
    )
ELSE()
    SRCS(
        _cryptmodule.c
        _posixsubprocess.c
        fcntlmodule.c
        grpmodule.c
        pwdmodule.c
        resource.c
        syslogmodule.c
        termios.c
    )
    IF (OS_LINUX)
        IF (NOT MUSL)
            EXTRALIBS(crypt)
        ENDIF()
        SRCS(
            spwdmodule.c
        )
    ELSEIF (OS_DARWIN)
        SRCS(
            _ctypes/darwin/dlfcn_simple.c
            _scproxy.c
        )
    ENDIF()
ENDIF()

END()
