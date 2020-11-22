#include <util/system/platform.h>

#ifndef Py_PYCONFIG_H
#define Py_PYCONFIG_H

#define DOUBLE_IS_LITTLE_ENDIAN_IEEE754 1
#define ENABLE_IPV6 1
#define HAVE_ACOSH 1
#define HAVE_ADDRINFO 1
#define HAVE_ALARM 1
#define HAVE_ALLOCA_H 1
#define HAVE_ASINH 1
#define HAVE_ASM_TYPES_H 1
#define HAVE_ATANH 1
#define HAVE_BIND_TEXTDOMAIN_CODESET 1
#define HAVE_C99_BOOL 1
#define HAVE_CHOWN 1
#define HAVE_CHROOT 1
#define HAVE_CLOCK 1
#define HAVE_COMPUTED_GOTOS 1
#define HAVE_CONFSTR 1
#define HAVE_COPYSIGN 1
#define HAVE_CTERMID 1
#define HAVE_DECL_ISFINITE 1
#define HAVE_DECL_ISINF 1
#define HAVE_DECL_ISNAN 1
#define HAVE_DEVICE_MACROS 1
#define HAVE_DEV_PTMX 1
#define HAVE_DIRENT_H 1
#define HAVE_DLFCN_H 1
#define HAVE_DLOPEN 1
#define HAVE_DUP2 1
#define HAVE_DYNAMIC_LOADING 1
#define HAVE_EPOLL 1
#define HAVE_ERF 1
#define HAVE_ERFC 1
#define HAVE_ERRNO_H 1
#define HAVE_EXECV 1
#define HAVE_EXPM1 1
#define HAVE_FCHDIR 1
#define HAVE_FCHMOD 1
#define HAVE_FCHOWN 1
#define HAVE_FCNTL_H 1
#define HAVE_FDATASYNC 1
#define HAVE_FINITE 1
#define HAVE_FLOCK 1
#define HAVE_FORK 1
#define HAVE_FORKPTY 1
#define HAVE_FPATHCONF 1
#define HAVE_FSEEKO 1
#define HAVE_FSTATVFS 1
#define HAVE_FSYNC 1
#define HAVE_FTELLO 1
#define HAVE_FTIME 1
#define HAVE_FTRUNCATE 1
#define HAVE_GAI_STRERROR 1
#define HAVE_GAMMA 1
#if defined(__x86_64__) || defined(__i386__)
#  define HAVE_GCC_ASM_FOR_X87 1
#endif
#define HAVE_GETADDRINFO 1
#define HAVE_GETCWD 1
#define HAVE_GETC_UNLOCKED 1
#define HAVE_GETGROUPS 1
#define HAVE_GETHOSTBYNAME_R 1
#define HAVE_GETHOSTBYNAME_R_6_ARG 1
#define HAVE_GETITIMER 1
#define HAVE_GETLOADAVG 1
#define HAVE_GETLOGIN 1
#define HAVE_GETNAMEINFO 1
#define HAVE_GETPAGESIZE 1
#define HAVE_GETPEERNAME 1
#define HAVE_GETPGID 1
#define HAVE_GETPGRP 1
#define HAVE_GETPID 1
#define HAVE_GETPRIORITY 1
#define HAVE_GETPWENT 1
#define HAVE_GETRESGID 1
#define HAVE_GETRESUID 1
#define HAVE_GETSID 1
#define HAVE_GETSPENT 1
#define HAVE_GETSPNAM 1
#define HAVE_GETTIMEOFDAY 1
#define HAVE_GETWD 1
#define HAVE_GRP_H 1
#define HAVE_HSTRERROR 1
#define HAVE_HYPOT 1
#define HAVE_INET_ATON 1
#define HAVE_INET_PTON 1
#define HAVE_INITGROUPS 1
#define HAVE_INT32_T 1
#define HAVE_INT64_T 1
#define HAVE_INTTYPES_H 1
#define HAVE_KILL 1
#define HAVE_KILLPG 1
#define HAVE_LANGINFO_H 1
#define HAVE_LCHOWN 1
#define HAVE_LGAMMA 1
#define HAVE_LIBDL 1
#define HAVE_LIBINTL_H 1
#define HAVE_LINK 1
#define HAVE_LINUX_NETLINK_H 1
#define HAVE_LINUX_TIPC_H 1
#define HAVE_LOG1P 1
#define HAVE_LONG_DOUBLE 1
#define HAVE_LONG_LONG 1
#define HAVE_LSTAT 1
#define HAVE_MAKEDEV 1
#define HAVE_MEMMOVE 1
#define HAVE_MEMORY_H 1
#define HAVE_MKFIFO 1
#define HAVE_MKNOD 1
#define HAVE_MKTIME 1
#define HAVE_MMAP 1
#define HAVE_MREMAP 1
#define HAVE_NETPACKET_PACKET_H 1
#define HAVE_NICE 1
#define HAVE_OPENPTY 1
#define HAVE_PATHCONF 1
#define HAVE_PAUSE 1
#define HAVE_POLL 1
#define HAVE_POLL_H 1
#define HAVE_PROTOTYPES 1
#define HAVE_PTHREAD_ATFORK 1
#define HAVE_PTHREAD_H 1
#define HAVE_PTHREAD_SIGMASK 1
#define HAVE_PTY_H 1
#define HAVE_PUTENV 1
#define HAVE_RAND_EGD 1
#define HAVE_READLINK 1
#define HAVE_REALPATH 1
#define HAVE_ROUND 1
#define HAVE_SELECT 1
#define HAVE_SEM_GETVALUE 1
#define HAVE_SEM_OPEN 1
#define HAVE_SEM_TIMEDWAIT 1
#define HAVE_SEM_UNLINK 1
#define HAVE_SETEGID 1
#define HAVE_SETEUID 1
#define HAVE_SETGID 1
#define HAVE_SETGROUPS 1
#define HAVE_SETITIMER 1
#define HAVE_SETLOCALE 1
#define HAVE_SETPGID 1
#define HAVE_SETPGRP 1
#define HAVE_SETREGID 1
#define HAVE_SETRESGID 1
#define HAVE_SETRESUID 1
#define HAVE_SETREUID 1
#define HAVE_SETSID 1
#define HAVE_SETUID 1
#define HAVE_SETVBUF 1
#define HAVE_SHADOW_H 1
#define HAVE_SIGACTION 1
#define HAVE_SIGINTERRUPT 1
#define HAVE_SIGNAL_H 1
#define HAVE_SIGRELSE 1
#define HAVE_SNPRINTF 1
#define HAVE_SOCKADDR_STORAGE 1
#define HAVE_SOCKETPAIR 1
#define HAVE_SPAWN_H 1
#define HAVE_SSIZE_T 1
#define HAVE_STATVFS 1
#define HAVE_STAT_TV_NSEC 1
#define HAVE_STDARG_PROTOTYPES 1
#define HAVE_STDINT_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRDUP 1
#define HAVE_STRFTIME 1
#define HAVE_STRINGS_H 1
#define HAVE_STRING_H 1
//#define HAVE_STROPTS_H 1
#define HAVE_STRUCT_STAT_ST_BLKSIZE 1
#define HAVE_STRUCT_STAT_ST_BLOCKS 1
#define HAVE_STRUCT_STAT_ST_RDEV 1
#define HAVE_STRUCT_TM_TM_ZONE 1
#define HAVE_ST_BLOCKS 1
#define HAVE_SYMLINK 1
#define HAVE_SYSCONF 1
#define HAVE_SYSEXITS_H 1
#define HAVE_SYS_EPOLL_H 1
#define HAVE_SYS_FILE_H 1
#define HAVE_SYS_PARAM_H 1
#define HAVE_SYS_POLL_H 1
#define HAVE_SYS_RESOURCE_H 1
#define HAVE_SYS_SELECT_H 1
#define HAVE_SYS_SOCKET_H 1
#define HAVE_SYS_STATVFS_H 1
#define HAVE_SYS_STAT_H 1
#define HAVE_SYS_TIMES_H 1
#define HAVE_SYS_TIME_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_SYS_UN_H 1
#define HAVE_SYS_UTSNAME_H 1
#define HAVE_SYS_WAIT_H 1
#define HAVE_TCGETPGRP 1
#define HAVE_TCSETPGRP 1
#undef HAVE_TEMPNAM
#define HAVE_TERMIOS_H 1
#define HAVE_TGAMMA 1
#define HAVE_TIMEGM 1
#define HAVE_TIMES 1
#define HAVE_TMPFILE 1
#undef HAVE_TMPNAM
#undef HAVE_TMPNAM_R
#define HAVE_TM_ZONE 1
#define HAVE_TRUNCATE 1
#define HAVE_UINT32_T 1
#define HAVE_UINT64_T 1
#define HAVE_UINTPTR_T 1
#define HAVE_UNAME 1
#define HAVE_UNISTD_H 1
#define HAVE_UNSETENV 1
#define HAVE_UTIMES 1
#define HAVE_UTIME_H 1
#define HAVE_WAIT3 1
#define HAVE_WAIT4 1
#define HAVE_WAITPID 1
#define HAVE_WCHAR_H 1
#define HAVE_WCSCOLL 1
#define HAVE_WORKING_TZSET 1
#define HAVE_ZLIB_COPY 1
#define MAJOR_IN_SYSMACROS 1
#define PTHREAD_SYSTEM_SCHED_SUPPORTED 1
#define PY_FORMAT_LONG_LONG "ll"
#define PY_FORMAT_SIZE_T "z"
#define PY_UNICODE_TYPE unsigned short
#define Py_UNICODE_SIZE 2
#define Py_USING_UNICODE 1
#define RETSIGTYPE void
#define SHLIB_EXT ".so"

#if defined(_64_)
#  define SIZEOF_FPOS_T 16
#  define SIZEOF_LONG 8
#  define SIZEOF_PTHREAD_T 8
#  if !defined(SIZEOF_SIZE_T)
#    define SIZEOF_SIZE_T 8
#  endif
#  define SIZEOF_TIME_T 8
#  define SIZEOF_UINTPTR_T 8
#  define SIZEOF_VOID_P 8
#else
#  if defined(_arm32_)
#    define SIZEOF_FPOS_T 16
#  else
#    define SIZEOF_FPOS_T 8
#  endif
#  define SIZEOF_LONG 4
#  define SIZEOF_PTHREAD_T 4
#  if !defined(SIZEOF_SIZE_T)
#    define SIZEOF_SIZE_T 4
#  endif
#  define SIZEOF_TIME_T 4
#  define SIZEOF_UINTPTR_T 4
#  define SIZEOF_VOID_P 4
#endif

#define SIZEOF_DOUBLE 8
#if defined(_arm32_)
#  define SIZEOF_LONG_DOUBLE 8
#else
#  define SIZEOF_LONG_DOUBLE 16
#endif
#define SIZEOF_FLOAT 4
#define SIZEOF_INT 4
#define SIZEOF_LONG_LONG 8
#define SIZEOF_OFF_T 8
#define SIZEOF_PID_T 4
#define SIZEOF_SHORT 2
#define SIZEOF_WCHAR_T 4
#define SIZEOF__BOOL 1

#define STDC_HEADERS 1
#define SYS_SELECT_WITH_SYS_TIME 1
#define TANH_PRESERVES_ZERO_SIGN 1
#define TIME_WITH_SYS_TIME 1

#ifndef _ALL_SOURCE
#  define _ALL_SOURCE 1
#endif

#if defined(_64_)
#  if defined(__aarch64__) || defined(__powerpc__)
#  else
#    define VA_LIST_IS_ARRAY 1
#  endif
#endif
#define WITH_DOC_STRINGS 1
#define WITH_PYMALLOC 1
#define WITH_THREAD 1
#define _DARWIN_C_SOURCE 1
#define _FILE_OFFSET_BITS 64
#define _GNU_SOURCE 1
#define _LARGEFILE_SOURCE 1
#if !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200112L
#endif
#define __BSD_VISIBLE 1

#ifndef HAVE_VALGRIND
#  undef WITH_VALGRIND
#endif

#endif /*Py_PYCONFIG_H*/
