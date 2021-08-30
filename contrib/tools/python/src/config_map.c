/* Generated automatically by ../modules.py */

#include "config_platform.h"

{"_bisect", init_bisect},
{"_collections", init_collections},
{"_csv", init_csv},
{"_elementtree", init_elementtree},
{"_functools", init_functools},
{"_hashlib", init_hashlib},
{"_heapq", init_heapq},
{"_hotshot", init_hotshot},
{"_io", init_io},
{"_json", init_json},
{"_locale", init_locale},
{"_lsprof", init_lsprof},
{"_md5", init_md5},
{"_multibytecodec", init_multibytecodec},
{"_random", init_random},
{"_sha", init_sha},
{"_sha256", init_sha256},
{"_sha512", init_sha512},
{"_ssl", init_ssl},
{"_struct", init_struct},
{"array", initarray},
{"audioop", initaudioop},
{"binascii", initbinascii},
{"bz2", initbz2},
{"cPickle", initcPickle},
{"cStringIO", initcStringIO},
{"cmath", initcmath},
{"datetime", initdatetime},
{"future_builtins", initfuture_builtins},
{"itertools", inititertools},
{"math", initmath},
{"mmap", initmmap},
{"operator", initoperator},
{"parser", initparser},
{"strop", initstrop},
{"time", inittime},
{"unicodedata", initunicodedata},
{"zlib", initzlib},

#ifdef _FREEBSD_
{"_multiprocessing", init_multiprocessing},
{"_multiprocessing", init_multiprocessing},
#endif

#ifdef _LINUX_
{"_multiprocessing", init_multiprocessing},
{"spwd", initspwd},
#endif

#ifdef _DARWIN_
#ifndef __IOS__
{"_multiprocessing", init_multiprocessing},
{"_scproxy", init_scproxy},
#endif
#endif

#ifdef _CYGWIN_
{"_multiprocessing", init_multiprocessing},
#endif

#ifdef _UNIX_
{"_socket", init_socket},
{"crypt", initcrypt},
{"fcntl", initfcntl},
{"grp", initgrp},
{"posix", initposix},
{"pwd", initpwd},
{"pyexpat", initpyexpat},
{"resource", initresource},
{"select", initselect},
{"syslog", initsyslog},
{"termios", inittermios},
#endif

#ifdef _WIN32_
{"_multiprocessing", init_multiprocessing},
{"_socket", init_socket},
{"nt", initnt},
{"pyexpat", initpyexpat},
{"select", initselect},
{"msvcrt", initmsvcrt},
{"_subprocess", init_subprocess},
{"_winreg", init_winreg},
#endif

#if defined(_x86_) && !defined(_CYGWIN_) || defined(__powerpc__) || defined(__aarch64__)
{"_ctypes", init_ctypes},
#endif
