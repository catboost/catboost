/* -*- C -*- ***********************************************
Copyright (c) 2000, BeOpen.com.
Copyright (c) 1995-2000, Corporation for National Research Initiatives.
Copyright (c) 1990-1995, Stichting Mathematisch Centrum.
All rights reserved.

See the file "Misc/COPYRIGHT" for information on usage and
redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
******************************************************************/

/* Module configuration */

/* !!! !!! !!! This file is edited by the makesetup script !!! !!! !!! */

/* This file contains the table of built-in modules.
   See create_builtin() in import.c. */

#include "Python.h"

#ifdef __cplusplus
extern "C" {
#endif


extern PyObject* PyInit__abc(void); // _abc
extern PyObject* PyInit__asyncio(void); // _asyncio
extern PyObject* PyInit__bisect(void); // _bisect
extern PyObject* PyInit__blake2(void); // _blake2
extern PyObject* PyInit__bz2(void); // _bz2
extern PyObject* PyInit__codecs(void); // _codecs
extern PyObject* PyInit__codecs_cn(void); // _codecs_cn
extern PyObject* PyInit__codecs_hk(void); // _codecs_hk
extern PyObject* PyInit__codecs_iso2022(void); // _codecs_iso2022
extern PyObject* PyInit__codecs_jp(void); // _codecs_jp
extern PyObject* PyInit__codecs_kr(void); // _codecs_kr
extern PyObject* PyInit__codecs_tw(void); // _codecs_tw
extern PyObject* PyInit__collections(void); // _collections
extern PyObject* PyInit__contextvars(void); // _contextvars
#if !defined(_MSC_VER)
extern PyObject* PyInit__crypt(void); // _crypt
#endif
extern PyObject* PyInit__csv(void); // _csv
extern PyObject* PyInit__ctypes(void); // _ctypes
extern PyObject* PyInit__datetime(void); // _datetime
extern PyObject* PyInit__decimal(void); // _decimal
extern PyObject* PyInit__elementtree(void); // _elementtree
extern PyObject* PyInit__functools(void); // _functools
extern PyObject* PyInit__hashlib(void); // _hashlib
extern PyObject* PyInit__heapq(void); // _heapq
extern PyObject* PyInit__imp(void); // _imp
extern PyObject* PyInit__io(void); // _io
extern PyObject* PyInit__json(void); // _json
extern PyObject* PyInit__locale(void); // _locale
extern PyObject* PyInit__lsprof(void); // _lsprof
extern PyObject* PyInit__lzma(void); // _lzma
extern PyObject* PyInit__md5(void); // _md5
extern PyObject* PyInit__multibytecodec(void); // _multibytecodec
extern PyObject* PyInit__multiprocessing(void); // _multiprocessing
extern PyObject* PyInit__opcode(void); // _opcode
extern PyObject* PyInit__operator(void); // _operator
extern PyObject* PyInit__pickle(void); // _pickle
extern PyObject* PyInit__posixshmem(void); // _posixshmem
#if !defined(_MSC_VER)
extern PyObject* PyInit__posixsubprocess(void); // _posixsubprocess
#endif
extern PyObject* PyInit__queue(void); // _queue
extern PyObject* PyInit__random(void); // _random
extern PyObject* PyInit__sha1(void); // _sha1
extern PyObject* PyInit__sha2(void); // _sha2
extern PyObject* PyInit__sha3(void); // _sha3
extern PyObject* PyInit__signal(void); // _signal
extern PyObject* PyInit__socket(void); // _socket
extern PyObject* PyInit__sre(void); // _sre
extern PyObject* PyInit__ssl(void); // _ssl
extern PyObject* PyInit__stat(void); // _stat
extern PyObject* PyInit__statistics(void); // _statistics
extern PyObject* PyInit__string(void); // _string
extern PyObject* PyInit__struct(void); // _struct
extern PyObject* PyInit__symtable(void); // _symtable
extern PyObject* PyInit__thread(void); // _thread
extern PyObject* PyInit__tracemalloc(void); // _tracemalloc
extern PyObject* PyInit__typing(void); // _typing
extern PyObject* PyInit__weakref(void); // _weakref
extern PyObject* PyInit__xxinterpchannels(void); // _xxinterpchannels
extern PyObject* PyInit__xxsubinterpreters(void); // _xxsubinterpreters
extern PyObject* PyInit__xxtestfuzz(void); // _xxtestfuzz
extern PyObject* PyInit__zoneinfo(void); // _zoneinfo
extern PyObject* PyInit_array(void); // array
extern PyObject* PyInit_atexit(void); // atexit
extern PyObject* PyInit_audioop(void); // audioop
extern PyObject* PyInit_binascii(void); // binascii
extern PyObject* PyInit_cmath(void); // cmath
extern PyObject* PyInit_errno(void); // errno
extern PyObject* PyInit_faulthandler(void); // faulthandler
#if !defined(_MSC_VER)
extern PyObject* PyInit_fcntl(void); // fcntl
#endif
#if !defined(_MSC_VER)
extern PyObject* PyInit_grp(void); // grp
#endif
extern PyObject* PyInit_itertools(void); // itertools
extern PyObject* PyInit_math(void); // math
extern PyObject* PyInit_mmap(void); // mmap
#if defined(_MSC_VER)
extern PyObject* PyInit_nt(void); // nt
#endif
#if !defined(_MSC_VER)
extern PyObject* PyInit_posix(void); // posix
#endif
#if !defined(_MSC_VER)
extern PyObject* PyInit_pwd(void); // pwd
#endif
extern PyObject* PyInit_pyexpat(void); // pyexpat
#if !defined(_MSC_VER)
extern PyObject* PyInit_resource(void); // resource
#endif
extern PyObject* PyInit_select(void); // select
#if defined(__linux__)
extern PyObject* PyInit_spwd(void); // spwd
#endif
#if !defined(_MSC_VER)
extern PyObject* PyInit_syslog(void); // syslog
#endif
#if !defined(_MSC_VER)
extern PyObject* PyInit_termios(void); // termios
#endif
extern PyObject* PyInit_time(void); // time
extern PyObject* PyInit_unicodedata(void); // unicodedata
extern PyObject* PyInit_zlib(void); // zlib
#if defined(__APPLE__)
extern PyObject* PyInit__scproxy(void); // _scproxy
#endif
#if defined(_MSC_VER)
extern PyObject* PyInit__overlapped(void); // _overlapped
#endif
#if defined(_MSC_VER)
extern PyObject* PyInit__winapi(void); // _winapi
#endif
#if defined(_MSC_VER)
extern PyObject* PyInit_msvcrt(void); // msvcrt
#endif
#if defined(_MSC_VER)
extern PyObject* PyInit_winreg(void); // winreg
#endif
#if defined(_MSC_VER)
extern PyObject* PyInit_winsound(void); // winsound
#endif

extern PyObject* PyMarshal_Init(void);
extern PyObject* PyInit__imp(void);
extern PyObject* PyInit_gc(void);
extern PyObject* PyInit__ast(void);
extern PyObject* PyInit__tokenize(void);
extern PyObject* _PyWarnings_Init(void);
extern PyObject* PyInit__string(void);

struct _inittab _PyImport_Inittab[] = {

    {"_abc", PyInit__abc},
    {"_asyncio", PyInit__asyncio},
    {"_bisect", PyInit__bisect},
    {"_blake2", PyInit__blake2},
    {"_bz2", PyInit__bz2},
    {"_codecs", PyInit__codecs},
    {"_codecs_cn", PyInit__codecs_cn},
    {"_codecs_hk", PyInit__codecs_hk},
    {"_codecs_iso2022", PyInit__codecs_iso2022},
    {"_codecs_jp", PyInit__codecs_jp},
    {"_codecs_kr", PyInit__codecs_kr},
    {"_codecs_tw", PyInit__codecs_tw},
    {"_collections", PyInit__collections},
    {"_contextvars", PyInit__contextvars},
#if !defined(_MSC_VER)
    {"_crypt", PyInit__crypt},
#endif
    {"_csv", PyInit__csv},
    {"_ctypes", PyInit__ctypes},
    {"_datetime", PyInit__datetime},
    {"_decimal", PyInit__decimal},
    {"_elementtree", PyInit__elementtree},
    {"_functools", PyInit__functools},
    {"_hashlib", PyInit__hashlib},
    {"_heapq", PyInit__heapq},
    {"_imp", PyInit__imp},
    {"_io", PyInit__io},
    {"_json", PyInit__json},
    {"_locale", PyInit__locale},
    {"_lsprof", PyInit__lsprof},
    {"_lzma", PyInit__lzma},
    {"_md5", PyInit__md5},
    {"_multibytecodec", PyInit__multibytecodec},
    {"_multiprocessing", PyInit__multiprocessing},
    {"_opcode", PyInit__opcode},
    {"_operator", PyInit__operator},
    {"_pickle", PyInit__pickle},
    {"_posixshmem", PyInit__posixshmem},
#if !defined(_MSC_VER)
    {"_posixsubprocess", PyInit__posixsubprocess},
#endif
    {"_queue", PyInit__queue},
    {"_random", PyInit__random},
    {"_sha1", PyInit__sha1},
    {"_sha2", PyInit__sha2},
    {"_sha3", PyInit__sha3},
    {"_signal", PyInit__signal},
    {"_socket", PyInit__socket},
    {"_sre", PyInit__sre},
    {"_ssl", PyInit__ssl},
    {"_stat", PyInit__stat},
    {"_statistics", PyInit__statistics},
    {"_string", PyInit__string},
    {"_struct", PyInit__struct},
    {"_symtable", PyInit__symtable},
    {"_thread", PyInit__thread},
    {"_tracemalloc", PyInit__tracemalloc},
    {"_typing", PyInit__typing},
    {"_weakref", PyInit__weakref},
    {"_xxinterpchannels", PyInit__xxinterpchannels},
    {"_xxsubinterpreters", PyInit__xxsubinterpreters},
    {"_xxtestfuzz", PyInit__xxtestfuzz},
    {"_zoneinfo", PyInit__zoneinfo},
    {"array", PyInit_array},
    {"atexit", PyInit_atexit},
    {"audioop", PyInit_audioop},
    {"binascii", PyInit_binascii},
    {"cmath", PyInit_cmath},
    {"errno", PyInit_errno},
    {"faulthandler", PyInit_faulthandler},
#if !defined(_MSC_VER)
    {"fcntl", PyInit_fcntl},
#endif
#if !defined(_MSC_VER)
    {"grp", PyInit_grp},
#endif
    {"itertools", PyInit_itertools},
    {"math", PyInit_math},
    {"mmap", PyInit_mmap},
#if defined(_MSC_VER)
    {"nt", PyInit_nt},
#endif
#if !defined(_MSC_VER)
    {"posix", PyInit_posix},
#endif
#if !defined(_MSC_VER)
    {"pwd", PyInit_pwd},
#endif
    {"pyexpat", PyInit_pyexpat},
#if !defined(_MSC_VER)
    {"resource", PyInit_resource},
#endif
    {"select", PyInit_select},
#if defined(__linux__)
    {"spwd", PyInit_spwd},
#endif
#if !defined(_MSC_VER)
    {"syslog", PyInit_syslog},
#endif
#if !defined(_MSC_VER)
    {"termios", PyInit_termios},
#endif
    {"time", PyInit_time},
    {"unicodedata", PyInit_unicodedata},
    {"zlib", PyInit_zlib},
#if defined(__APPLE__)
    {"_scproxy", PyInit__scproxy},
#endif
#if defined(_MSC_VER)
    {"_overlapped", PyInit__overlapped},
#endif
#if defined(_MSC_VER)
    {"_winapi", PyInit__winapi},
#endif
#if defined(_MSC_VER)
    {"msvcrt", PyInit_msvcrt},
#endif
#if defined(_MSC_VER)
    {"winreg", PyInit_winreg},
#endif
#if defined(_MSC_VER)
    {"winsound", PyInit_winsound},
#endif

    /* This module lives in marshal.c */
    {"marshal", PyMarshal_Init},

    /* This lives in import.c */
    {"_imp", PyInit__imp},

    /* This lives in Python/Python-ast.c */
    {"_ast", PyInit__ast},

    /* This lives in Python/Python-tokenizer.c */
    {"_tokenize", PyInit__tokenize},

    /* These entries are here for sys.builtin_module_names */
    {"builtins", NULL},
    {"sys", NULL},

    /* This lives in gcmodule.c */
    {"gc", PyInit_gc},

    /* This lives in _warnings.c */
    {"_warnings", _PyWarnings_Init},

    /* This lives in Objects/unicodeobject.c */
    {"_string", PyInit__string},

    /* Sentinel */
    {0, 0}
};


#ifdef __cplusplus
}
#endif
