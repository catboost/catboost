ENABLE(PYBUILD_NO_PYC)

PY3_LIBRARY()



LICENSE(Python-2.0)

NO_PYTHON_INCLUDES()

SRCDIR(contrib/tools/python3/src/Lib)

INCLUDE(../srcs.cmake)

NO_CHECK_IMPORTS(
    antigravity
    asyncio.windows_events
    asyncio.windows_utils
    crypt
    ctypes.wintypes
    curses.*
    dbm.gnu
    dbm.ndbm
    distutils._msvccompiler
    distutils.command.bdist_msi
    distutils.msvc9compiler
    encodings.cp65001
    encodings.mbcs
    encodings.oem
    lzma
    msilib.*
    multiprocessing.popen_spawn_win32
    sqlite3.*
    turtle
)

PY_SRCS(
    TOP_LEVEL
    ${PYTHON3_LIB_SRCS}
)

NO_LINT()

END()
