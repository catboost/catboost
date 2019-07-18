LIBRARY()



LICENSE(Python-2.0)

PEERDIR(
    contrib/tools/python3/src/Modules
)

ADDINCL(
    contrib/tools/python3/src/Include
    contrib/tools/python3/src/Modules
    contrib/tools/python3/src/Modules/_decimal/libmpdec
    contrib/tools/python3/src/PC
)

CFLAGS(
    -DPy_BUILD_CORE
)

IF (OS_DARWIN)
    LDFLAGS(
        -framework CoreFoundation
        -framework SystemConfiguration
    )
ELSEIF (OS_WINDOWS)
    LDFLAGS(
        Mincore.lib
        Shlwapi.lib
        Winmm.lib
    )

    DISABLE(MSVC_INLINE_OPTIMIZED)
ENDIF()

NO_COMPILER_WARNINGS()

NO_UTIL()

SRCS(
    Modules/_functoolsmodule.c
    Modules/_io/_iomodule.c
    Modules/_io/bufferedio.c
    Modules/_io/bytesio.c
    Modules/_io/fileio.c
    Modules/_io/iobase.c
    Modules/_io/stringio.c
    Modules/_io/textio.c
    Modules/_io/winconsoleio.c
    Modules/_threadmodule.c
    Modules/config.c
    Modules/gcmodule.c
    Modules/main.c
    Modules/mmapmodule.c
    Modules/posixmodule.c
    Modules/signalmodule.c
    Modules/timemodule.c
    Modules/zipimport.c
    Objects/abstract.c
    Objects/accu.c
    Objects/boolobject.c
    Objects/bytearrayobject.c
    Objects/bytes_methods.c
    Objects/bytesobject.c
    Objects/call.c
    Objects/capsule.c
    Objects/cellobject.c
    Objects/classobject.c
    Objects/codeobject.c
    Objects/complexobject.c
    Objects/descrobject.c
    Objects/dictobject.c
    Objects/enumobject.c
    Objects/exceptions.c
    Objects/fileobject.c
    Objects/floatobject.c
    Objects/frameobject.c
    Objects/funcobject.c
    Objects/genobject.c
    Objects/iterobject.c
    Objects/listobject.c
    Objects/longobject.c
    Objects/memoryobject.c
    Objects/methodobject.c
    Objects/moduleobject.c
    Objects/namespaceobject.c
    Objects/object.c
    Objects/obmalloc.c
    Objects/odictobject.c
    Objects/rangeobject.c
    Objects/setobject.c
    Objects/sliceobject.c
    Objects/structseq.c
    Objects/tupleobject.c
    Objects/typeobject.c
    Objects/unicodectype.c
    Objects/unicodeobject.c
    Objects/weakrefobject.c
    Parser/acceler.c
    Parser/bitset.c
    Parser/firstsets.c
    Parser/grammar.c
    Parser/grammar1.c
    Parser/listnode.c
    Parser/metagrammar.c
    Parser/myreadline.c
    Parser/node.c
    Parser/parser.c
    Parser/parsetok.c
    Parser/pgen.c
    Parser/printgrammar.c
    Parser/tokenizer.c
    Python/Python-ast.c
    Python/_warnings.c
    Python/asdl.c
    Python/ast.c
    Python/ast_opt.c
    Python/ast_unparse.c
    Python/bltinmodule.c
    Python/bootstrap_hash.c
    Python/ceval.c
    Python/codecs.c
    Python/compile.c
    Python/context.c
    Python/dtoa.c
    Python/dynamic_annotations.c
    Python/errors.c
    Python/fileutils.c
    Python/formatter_unicode.c
    Python/frozen.c
    Python/frozenmain.c
    Python/future.c
    Python/getargs.c
    Python/getcompiler.c
    Python/getcopyright.c
    Python/getopt.c
    Python/getplatform.c
    Python/getversion.c
    Python/graminit.c
    Python/hamt.c
    Python/import.c
    Python/importdl.c
    Python/marshal.c
    Python/modsupport.c
    Python/mysnprintf.c
    Python/mystrtoul.c
    Python/pathconfig.c
    Python/peephole.c
    Python/pyarena.c
    Python/pyctype.c
    Python/pyfpe.c
    Python/pyhash.c
    Python/pylifecycle.c
    Python/pymath.c
    Python/pystate.c
    Python/pystrcmp.c
    Python/pystrhex.c
    Python/pystrtod.c
    Python/pythonrun.c
    Python/pytime.c
    Python/structmember.c
    Python/symtable.c
    Python/sysmodule.c
    Python/thread.c
    Python/traceback.c
)

IF (OS_WINDOWS) SRCS(
    PC/WinMain.c
    PC/dl_nt.c
    PC/getpathp.c
    PC/invalid_parameter_handler.c
    PC/msvcrtmodule.c
    PC/winreg.c
    PC/winsound.c
)
ENDIF()

IF (OS_WINDOWS)
    SRCS(
        Python/dynload_win.c
    )
ELSE()
    SRCS(
        Modules/getpath.c
        Python/dynload_shlib.c
    )
ENDIF()

END()
