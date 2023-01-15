/* Generated automatically by ../modules.py */

#include "config_platform.h"

extern void init_bisect(void);
extern void init_collections(void);
extern void init_csv(void);
extern void init_elementtree(void);
extern void init_functools(void);
extern void init_hashlib(void);
extern void init_heapq(void);
extern void init_hotshot(void);
extern void init_io(void);
extern void init_json(void);
extern void init_locale(void);
extern void init_lsprof(void);
extern void init_md5(void);
extern void init_multibytecodec(void);
extern void init_random(void);
extern void init_sha(void);
extern void init_sha256(void);
extern void init_sha512(void);
extern void init_ssl(void);
extern void init_struct(void);
extern void initarray(void);
extern void initaudioop(void);
extern void initbinascii(void);
extern void initbz2(void);
extern void initcPickle(void);
extern void initcStringIO(void);
extern void initcmath(void);
extern void initdatetime(void);
extern void initfuture_builtins(void);
extern void inititertools(void);
extern void initmath(void);
extern void initmmap(void);
extern void initoperator(void);
extern void initparser(void);
extern void initstrop(void);
extern void inittime(void);
extern void initunicodedata(void);
extern void initzlib(void);

#ifdef _FREEBSD_
extern void init_multiprocessing(void);
extern void init_multiprocessing(void);
#endif

#ifdef _LINUX_
extern void init_multiprocessing(void);
extern void initspwd(void);
#endif

#ifdef _DARWIN_
#ifndef __IOS__
extern void init_multiprocessing(void);
extern void init_scproxy(void);
#endif
#endif

#ifdef _CYGWIN_
extern void init_multiprocessing(void);
#endif

#ifdef _UNIX_
extern void init_socket(void);
extern void initcrypt(void);
extern void initfcntl(void);
extern void initgrp(void);
extern void initposix(void);
extern void initpwd(void);
extern void initpyexpat(void);
extern void initresource(void);
extern void initselect(void);
extern void initsyslog(void);
extern void inittermios(void);
#endif

#ifdef _WIN32_
extern void init_multiprocessing(void);
extern void init_socket(void);
extern void initnt(void);
extern void initpyexpat(void);
extern void initselect(void);
extern void initmsvcrt(void);
extern void init_subprocess(void);
extern void init_winreg(void);
#endif

#if !defined(_CYGWIN_)
extern void init_ctypes(void);
#endif
