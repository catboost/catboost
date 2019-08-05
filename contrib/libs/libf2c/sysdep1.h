#ifndef SYSDEP_H_INCLUDED
#define SYSDEP_H_INCLUDED
#undef USE_LARGEFILE
#ifndef NO_LONG_LONG

#ifdef __sun__
#define USE_LARGEFILE
#define OFF_T off64_t
#endif

#ifdef __linux__
#define USE_LARGEFILE
#define OFF_T off64_t
#endif

#ifdef _AIX43
#define _LARGE_FILES
#define _LARGE_FILE_API
#define USE_LARGEFILE
#endif /*_AIX43*/

#ifdef __hpux
#define _FILE64
#define _LARGEFILE64_SOURCE
#define USE_LARGEFILE
#endif /*__hpux*/

#ifdef __sgi
#define USE_LARGEFILE
#endif /*__sgi*/

#ifdef __FreeBSD__
#define OFF_T off_t
#define FSEEK fseeko
#define FTELL ftello
#endif

#ifdef __ANDROID__
#undef USE_LARGEFILE
#define OFF_T off64_t
#endif

#ifdef USE_LARGEFILE
#ifndef OFF_T
#define OFF_T off64_t
#endif
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE
#include <sys/types.h>
#include <sys/stat.h>
#define FOPEN fopen64
#define FREOPEN freopen64
#define FSEEK fseeko64
#define FSTAT fstat64
#define FTELL ftello64
#define FTRUNCATE ftruncate64
#define STAT stat64
#define STAT_ST stat64
#endif /*USE_LARGEFILE*/
#endif /*NO_LONG_LONG*/

#ifndef NON_UNIX_STDIO
#ifndef USE_LARGEFILE
#define _INCLUDE_POSIX_SOURCE	/* for HP-UX */
#define _INCLUDE_XOPEN_SOURCE	/* for HP-UX */
#include "sys/types.h"
#include "sys/stat.h"
#endif
#endif

#endif /*SYSDEP_H_INCLUDED*/
