/* DO NOT EDIT! GENERATED AUTOMATICALLY! */
/* Definitions for POSIX spawn interface.
   Copyright (C) 2000, 2003-2004, 2008-2013 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#ifndef _GL_M4_SPAWN_H

#if __GNUC__ >= 3

#endif


/* The include_next requires a split double-inclusion guard.  */
#if 0
# include <spawn.h>
#endif

#ifndef _GL_M4_SPAWN_H
#define _GL_M4_SPAWN_H

/* Get definitions of 'struct sched_param' and 'sigset_t'.
   But avoid namespace pollution on glibc systems.  */
#if !(defined __GLIBC__ && !defined __UCLIBC__)
# include <sched.h>
# include <signal.h>
#endif

#include <sys/types.h>

#ifndef __THROW
# define __THROW
#endif

/* GCC 2.95 and later have "__restrict"; C99 compilers have
   "restrict", and "configure" may have defined "restrict".
   Other compilers use __restrict, __restrict__, and _Restrict, and
   'configure' might #define 'restrict' to those words, so pick a
   different name.  */
#ifndef _Restrict_
# if 199901L <= __STDC_VERSION__
#  define _Restrict_ restrict
# elif 2 < __GNUC__ || (2 == __GNUC__ && 95 <= __GNUC_MINOR__)
#  define _Restrict_ __restrict
# else
#  define _Restrict_
# endif
#endif
/* gcc 3.1 and up support the [restrict] syntax.  Don't trust
   sys/cdefs.h's definition of __restrict_arr, though, as it
   mishandles gcc -ansi -pedantic.  */
#ifndef _Restrict_arr_
# if ((199901L <= __STDC_VERSION__                                      \
       || ((3 < __GNUC__ || (3 == __GNUC__ && 1 <= __GNUC_MINOR__))     \
           && !defined __STRICT_ANSI__))                                        \
      && !defined __GNUG__)
#  define _Restrict_arr_ _Restrict_
# else
#  define _Restrict_arr_
# endif
#endif

/* The definitions of _GL_FUNCDECL_RPL etc. are copied here.  */
#ifndef _GL_CXXDEFS_H
#define _GL_CXXDEFS_H

/* The three most frequent use cases of these macros are:

   * For providing a substitute for a function that is missing on some
     platforms, but is declared and works fine on the platforms on which
     it exists:

       #if @GNULIB_FOO@
       # if !@HAVE_FOO@
       _GL_FUNCDECL_SYS (foo, ...);
       # endif
       _GL_CXXALIAS_SYS (foo, ...);
       _GL_CXXALIASWARN (foo);
       #elif defined GNULIB_POSIXCHECK
       ...
       #endif

   * For providing a replacement for a function that exists on all platforms,
     but is broken/insufficient and needs to be replaced on some platforms:

       #if @GNULIB_FOO@
       # if @REPLACE_FOO@
       #  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
       #   undef foo
       #   define foo rpl_foo
       #  endif
       _GL_FUNCDECL_RPL (foo, ...);
       _GL_CXXALIAS_RPL (foo, ...);
       # else
       _GL_CXXALIAS_SYS (foo, ...);
       # endif
       _GL_CXXALIASWARN (foo);
       #elif defined GNULIB_POSIXCHECK
       ...
       #endif

   * For providing a replacement for a function that exists on some platforms
     but is broken/insufficient and needs to be replaced on some of them and
     is additionally either missing or undeclared on some other platforms:

       #if @GNULIB_FOO@
       # if @REPLACE_FOO@
       #  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
       #   undef foo
       #   define foo rpl_foo
       #  endif
       _GL_FUNCDECL_RPL (foo, ...);
       _GL_CXXALIAS_RPL (foo, ...);
       # else
       #  if !@HAVE_FOO@   or   if !@HAVE_DECL_FOO@
       _GL_FUNCDECL_SYS (foo, ...);
       #  endif
       _GL_CXXALIAS_SYS (foo, ...);
       # endif
       _GL_CXXALIASWARN (foo);
       #elif defined GNULIB_POSIXCHECK
       ...
       #endif
*/

/* _GL_EXTERN_C declaration;
   performs the declaration with C linkage.  */
#if defined __cplusplus
# define _GL_EXTERN_C extern "C"
#else
# define _GL_EXTERN_C extern
#endif

/* _GL_FUNCDECL_RPL (func, rettype, parameters_and_attributes);
   declares a replacement function, named rpl_func, with the given prototype,
   consisting of return type, parameters, and attributes.
   Example:
     _GL_FUNCDECL_RPL (open, int, (const char *filename, int flags, ...)
                                  _GL_ARG_NONNULL ((1)));
 */
#define _GL_FUNCDECL_RPL(func,rettype,parameters_and_attributes) \
  _GL_FUNCDECL_RPL_1 (rpl_##func, rettype, parameters_and_attributes)
#define _GL_FUNCDECL_RPL_1(rpl_func,rettype,parameters_and_attributes) \
  _GL_EXTERN_C rettype rpl_func parameters_and_attributes

/* _GL_FUNCDECL_SYS (func, rettype, parameters_and_attributes);
   declares the system function, named func, with the given prototype,
   consisting of return type, parameters, and attributes.
   Example:
     _GL_FUNCDECL_SYS (open, int, (const char *filename, int flags, ...)
                                  _GL_ARG_NONNULL ((1)));
 */
#define _GL_FUNCDECL_SYS(func,rettype,parameters_and_attributes) \
  _GL_EXTERN_C rettype func parameters_and_attributes

/* _GL_CXXALIAS_RPL (func, rettype, parameters);
   declares a C++ alias called GNULIB_NAMESPACE::func
   that redirects to rpl_func, if GNULIB_NAMESPACE is defined.
   Example:
     _GL_CXXALIAS_RPL (open, int, (const char *filename, int flags, ...));
 */
#define _GL_CXXALIAS_RPL(func,rettype,parameters) \
  _GL_CXXALIAS_RPL_1 (func, rpl_##func, rettype, parameters)
#if defined __cplusplus && defined GNULIB_NAMESPACE
# define _GL_CXXALIAS_RPL_1(func,rpl_func,rettype,parameters) \
    namespace GNULIB_NAMESPACE                                \
    {                                                         \
      rettype (*const func) parameters = ::rpl_func;          \
    }                                                         \
    _GL_EXTERN_C int _gl_cxxalias_dummy
#else
# define _GL_CXXALIAS_RPL_1(func,rpl_func,rettype,parameters) \
    _GL_EXTERN_C int _gl_cxxalias_dummy
#endif

/* _GL_CXXALIAS_RPL_CAST_1 (func, rpl_func, rettype, parameters);
   is like  _GL_CXXALIAS_RPL_1 (func, rpl_func, rettype, parameters);
   except that the C function rpl_func may have a slightly different
   declaration.  A cast is used to silence the "invalid conversion" error
   that would otherwise occur.  */
#if defined __cplusplus && defined GNULIB_NAMESPACE
# define _GL_CXXALIAS_RPL_CAST_1(func,rpl_func,rettype,parameters) \
    namespace GNULIB_NAMESPACE                                     \
    {                                                              \
      rettype (*const func) parameters =                           \
        reinterpret_cast<rettype(*)parameters>(::rpl_func);        \
    }                                                              \
    _GL_EXTERN_C int _gl_cxxalias_dummy
#else
# define _GL_CXXALIAS_RPL_CAST_1(func,rpl_func,rettype,parameters) \
    _GL_EXTERN_C int _gl_cxxalias_dummy
#endif

/* _GL_CXXALIAS_SYS (func, rettype, parameters);
   declares a C++ alias called GNULIB_NAMESPACE::func
   that redirects to the system provided function func, if GNULIB_NAMESPACE
   is defined.
   Example:
     _GL_CXXALIAS_SYS (open, int, (const char *filename, int flags, ...));
 */
#if defined __cplusplus && defined GNULIB_NAMESPACE
  /* If we were to write
       rettype (*const func) parameters = ::func;
     like above in _GL_CXXALIAS_RPL_1, the compiler could optimize calls
     better (remove an indirection through a 'static' pointer variable),
     but then the _GL_CXXALIASWARN macro below would cause a warning not only
     for uses of ::func but also for uses of GNULIB_NAMESPACE::func.  */
# define _GL_CXXALIAS_SYS(func,rettype,parameters) \
    namespace GNULIB_NAMESPACE                     \
    {                                              \
      static rettype (*func) parameters = ::func;  \
    }                                              \
    _GL_EXTERN_C int _gl_cxxalias_dummy
#else
# define _GL_CXXALIAS_SYS(func,rettype,parameters) \
    _GL_EXTERN_C int _gl_cxxalias_dummy
#endif

/* _GL_CXXALIAS_SYS_CAST (func, rettype, parameters);
   is like  _GL_CXXALIAS_SYS (func, rettype, parameters);
   except that the C function func may have a slightly different declaration.
   A cast is used to silence the "invalid conversion" error that would
   otherwise occur.  */
#if defined __cplusplus && defined GNULIB_NAMESPACE
# define _GL_CXXALIAS_SYS_CAST(func,rettype,parameters) \
    namespace GNULIB_NAMESPACE                          \
    {                                                   \
      static rettype (*func) parameters =               \
        reinterpret_cast<rettype(*)parameters>(::func); \
    }                                                   \
    _GL_EXTERN_C int _gl_cxxalias_dummy
#else
# define _GL_CXXALIAS_SYS_CAST(func,rettype,parameters) \
    _GL_EXTERN_C int _gl_cxxalias_dummy
#endif

/* _GL_CXXALIAS_SYS_CAST2 (func, rettype, parameters, rettype2, parameters2);
   is like  _GL_CXXALIAS_SYS (func, rettype, parameters);
   except that the C function is picked among a set of overloaded functions,
   namely the one with rettype2 and parameters2.  Two consecutive casts
   are used to silence the "cannot find a match" and "invalid conversion"
   errors that would otherwise occur.  */
#if defined __cplusplus && defined GNULIB_NAMESPACE
  /* The outer cast must be a reinterpret_cast.
     The inner cast: When the function is defined as a set of overloaded
     functions, it works as a static_cast<>, choosing the designated variant.
     When the function is defined as a single variant, it works as a
     reinterpret_cast<>. The parenthesized cast syntax works both ways.  */
# define _GL_CXXALIAS_SYS_CAST2(func,rettype,parameters,rettype2,parameters2) \
    namespace GNULIB_NAMESPACE                                                \
    {                                                                         \
      static rettype (*func) parameters =                                     \
        reinterpret_cast<rettype(*)parameters>(                               \
          (rettype2(*)parameters2)(::func));                                  \
    }                                                                         \
    _GL_EXTERN_C int _gl_cxxalias_dummy
#else
# define _GL_CXXALIAS_SYS_CAST2(func,rettype,parameters,rettype2,parameters2) \
    _GL_EXTERN_C int _gl_cxxalias_dummy
#endif

/* _GL_CXXALIASWARN (func);
   causes a warning to be emitted when ::func is used but not when
   GNULIB_NAMESPACE::func is used.  func must be defined without overloaded
   variants.  */
#if defined __cplusplus && defined GNULIB_NAMESPACE
# define _GL_CXXALIASWARN(func) \
   _GL_CXXALIASWARN_1 (func, GNULIB_NAMESPACE)
# define _GL_CXXALIASWARN_1(func,namespace) \
   _GL_CXXALIASWARN_2 (func, namespace)
/* To work around GCC bug <http://gcc.gnu.org/bugzilla/show_bug.cgi?id=43881>,
   we enable the warning only when not optimizing.  */
# if !__OPTIMIZE__
#  define _GL_CXXALIASWARN_2(func,namespace) \
    _GL_WARN_ON_USE (func, \
                     "The symbol ::" #func " refers to the system function. " \
                     "Use " #namespace "::" #func " instead.")
# elif __GNUC__ >= 3 && GNULIB_STRICT_CHECKING
#  define _GL_CXXALIASWARN_2(func,namespace) \
     extern __typeof__ (func) func
# else
#  define _GL_CXXALIASWARN_2(func,namespace) \
     _GL_EXTERN_C int _gl_cxxalias_dummy
# endif
#else
# define _GL_CXXALIASWARN(func) \
    _GL_EXTERN_C int _gl_cxxalias_dummy
#endif

/* _GL_CXXALIASWARN1 (func, rettype, parameters_and_attributes);
   causes a warning to be emitted when the given overloaded variant of ::func
   is used but not when GNULIB_NAMESPACE::func is used.  */
#if defined __cplusplus && defined GNULIB_NAMESPACE
# define _GL_CXXALIASWARN1(func,rettype,parameters_and_attributes) \
   _GL_CXXALIASWARN1_1 (func, rettype, parameters_and_attributes, \
                        GNULIB_NAMESPACE)
# define _GL_CXXALIASWARN1_1(func,rettype,parameters_and_attributes,namespace) \
   _GL_CXXALIASWARN1_2 (func, rettype, parameters_and_attributes, namespace)
/* To work around GCC bug <http://gcc.gnu.org/bugzilla/show_bug.cgi?id=43881>,
   we enable the warning only when not optimizing.  */
# if !__OPTIMIZE__
#  define _GL_CXXALIASWARN1_2(func,rettype,parameters_and_attributes,namespace) \
    _GL_WARN_ON_USE_CXX (func, rettype, parameters_and_attributes, \
                         "The symbol ::" #func " refers to the system function. " \
                         "Use " #namespace "::" #func " instead.")
# elif __GNUC__ >= 3 && GNULIB_STRICT_CHECKING
#  define _GL_CXXALIASWARN1_2(func,rettype,parameters_and_attributes,namespace) \
     extern __typeof__ (func) func
# else
#  define _GL_CXXALIASWARN1_2(func,rettype,parameters_and_attributes,namespace) \
     _GL_EXTERN_C int _gl_cxxalias_dummy
# endif
#else
# define _GL_CXXALIASWARN1(func,rettype,parameters_and_attributes) \
    _GL_EXTERN_C int _gl_cxxalias_dummy
#endif

#endif /* _GL_CXXDEFS_H */

/* The definition of _GL_ARG_NONNULL is copied here.  */
/* _GL_ARG_NONNULL((n,...,m)) tells the compiler and static analyzer tools
   that the values passed as arguments n, ..., m must be non-NULL pointers.
   n = 1 stands for the first argument, n = 2 for the second argument etc.  */
#ifndef _GL_ARG_NONNULL
# if (__GNUC__ == 3 && __GNUC_MINOR__ >= 3) || __GNUC__ > 3
#  define _GL_ARG_NONNULL(params) __attribute__ ((__nonnull__ params))
# else
#  define _GL_ARG_NONNULL(params)
# endif
#endif

/* The definition of _GL_WARN_ON_USE is copied here.  */
#ifndef _GL_WARN_ON_USE

# if 4 < __GNUC__ || (__GNUC__ == 4 && 3 <= __GNUC_MINOR__)
/* A compiler attribute is available in gcc versions 4.3.0 and later.  */
#  define _GL_WARN_ON_USE(function, message) \
extern __typeof__ (function) function __attribute__ ((__warning__ (message)))
# elif __GNUC__ >= 3 && GNULIB_STRICT_CHECKING
/* Verify the existence of the function.  */
#  define _GL_WARN_ON_USE(function, message) \
extern __typeof__ (function) function
# else /* Unsupported.  */
#  define _GL_WARN_ON_USE(function, message) \
_GL_WARN_EXTERN_C int _gl_warn_on_use
# endif
#endif

/* _GL_WARN_ON_USE_CXX (function, rettype, parameters_and_attributes, "string")
   is like _GL_WARN_ON_USE (function, "string"), except that the function is
   declared with the given prototype, consisting of return type, parameters,
   and attributes.
   This variant is useful for overloaded functions in C++. _GL_WARN_ON_USE does
   not work in this case.  */
#ifndef _GL_WARN_ON_USE_CXX
# if 4 < __GNUC__ || (__GNUC__ == 4 && 3 <= __GNUC_MINOR__)
#  define _GL_WARN_ON_USE_CXX(function,rettype,parameters_and_attributes,msg) \
extern rettype function parameters_and_attributes \
     __attribute__ ((__warning__ (msg)))
# elif __GNUC__ >= 3 && GNULIB_STRICT_CHECKING
/* Verify the existence of the function.  */
#  define _GL_WARN_ON_USE_CXX(function,rettype,parameters_and_attributes,msg) \
extern rettype function parameters_and_attributes
# else /* Unsupported.  */
#  define _GL_WARN_ON_USE_CXX(function,rettype,parameters_and_attributes,msg) \
_GL_WARN_EXTERN_C int _gl_warn_on_use
# endif
#endif

/* _GL_WARN_EXTERN_C declaration;
   performs the declaration with C linkage.  */
#ifndef _GL_WARN_EXTERN_C
# if defined __cplusplus
#  define _GL_WARN_EXTERN_C extern "C"
# else
#  define _GL_WARN_EXTERN_C extern
# endif
#endif


/* Data structure to contain attributes for thread creation.  */
#if 0
# define posix_spawnattr_t rpl_posix_spawnattr_t
#endif
#if 0 || !0
# if !GNULIB_defined_posix_spawnattr_t
typedef struct
{
  short int _flags;
  pid_t _pgrp;
  sigset_t _sd;
  sigset_t _ss;
  struct sched_param _sp;
  int _policy;
  int __pad[16];
} posix_spawnattr_t;
#  define GNULIB_defined_posix_spawnattr_t 1
# endif
#endif


/* Data structure to contain information about the actions to be
   performed in the new process with respect to file descriptors.  */
#if 0
# define posix_spawn_file_actions_t rpl_posix_spawn_file_actions_t
#endif
#if 0 || !0
# if !GNULIB_defined_posix_spawn_file_actions_t
typedef struct
{
  int _allocated;
  int _used;
  struct __spawn_action *_actions;
  int __pad[16];
} posix_spawn_file_actions_t;
#  define GNULIB_defined_posix_spawn_file_actions_t 1
# endif
#endif


/* Flags to be set in the 'posix_spawnattr_t'.  */
#if 0
/* Use the values from the system, but provide the missing ones.  */
# ifndef POSIX_SPAWN_SETSCHEDPARAM
#  define POSIX_SPAWN_SETSCHEDPARAM 0
# endif
# ifndef POSIX_SPAWN_SETSCHEDULER
#  define POSIX_SPAWN_SETSCHEDULER 0
# endif
#else
# if 0
/* Use the values from the system, for better compatibility.  */
/* But this implementation does not support AIX extensions.  */
#  undef POSIX_SPAWN_FORK_HANDLERS
# else
#  define POSIX_SPAWN_RESETIDS           0x01
#  define POSIX_SPAWN_SETPGROUP          0x02
#  define POSIX_SPAWN_SETSIGDEF          0x04
#  define POSIX_SPAWN_SETSIGMASK         0x08
#  define POSIX_SPAWN_SETSCHEDPARAM      0x10
#  define POSIX_SPAWN_SETSCHEDULER       0x20
# endif
#endif
/* A GNU extension.  Use the next free bit position.  */
#define POSIX_SPAWN_USEVFORK \
  ((POSIX_SPAWN_RESETIDS | (POSIX_SPAWN_RESETIDS - 1)                     \
    | POSIX_SPAWN_SETPGROUP | (POSIX_SPAWN_SETPGROUP - 1)                 \
    | POSIX_SPAWN_SETSIGDEF | (POSIX_SPAWN_SETSIGDEF - 1)                 \
    | POSIX_SPAWN_SETSIGMASK | (POSIX_SPAWN_SETSIGMASK - 1)               \
    | POSIX_SPAWN_SETSCHEDPARAM                                           \
    | (POSIX_SPAWN_SETSCHEDPARAM > 0 ? POSIX_SPAWN_SETSCHEDPARAM - 1 : 0) \
    | POSIX_SPAWN_SETSCHEDULER                                            \
    | (POSIX_SPAWN_SETSCHEDULER > 0 ? POSIX_SPAWN_SETSCHEDULER - 1 : 0))  \
   + 1)
#if !GNULIB_defined_verify_POSIX_SPAWN_USEVFORK_no_overlap
typedef int verify_POSIX_SPAWN_USEVFORK_no_overlap
            [(((POSIX_SPAWN_RESETIDS | POSIX_SPAWN_SETPGROUP
                | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK
                | POSIX_SPAWN_SETSCHEDPARAM | POSIX_SPAWN_SETSCHEDULER)
               & POSIX_SPAWN_USEVFORK)
              == 0)
             ? 1 : -1];
# define GNULIB_defined_verify_POSIX_SPAWN_USEVFORK_no_overlap 1
#endif


#if 0
/* Spawn a new process executing PATH with the attributes describes in *ATTRP.
   Before running the process perform the actions described in FILE-ACTIONS.

   This function is a possible cancellation points and therefore not
   marked with __THROW. */
# if 0
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawn rpl_posix_spawn
#  endif
_GL_FUNCDECL_RPL (posix_spawn, int,
                  (pid_t *_Restrict_ __pid,
                   const char *_Restrict_ __path,
                   const posix_spawn_file_actions_t *_Restrict_ __file_actions,
                   const posix_spawnattr_t *_Restrict_ __attrp,
                   char *const argv[_Restrict_arr_],
                   char *const envp[_Restrict_arr_])
                  _GL_ARG_NONNULL ((2, 5, 6)));
_GL_CXXALIAS_RPL (posix_spawn, int,
                  (pid_t *_Restrict_ __pid,
                   const char *_Restrict_ __path,
                   const posix_spawn_file_actions_t *_Restrict_ __file_actions,
                   const posix_spawnattr_t *_Restrict_ __attrp,
                   char *const argv[_Restrict_arr_],
                   char *const envp[_Restrict_arr_]));
# else
#  if !0
_GL_FUNCDECL_SYS (posix_spawn, int,
                  (pid_t *_Restrict_ __pid,
                   const char *_Restrict_ __path,
                   const posix_spawn_file_actions_t *_Restrict_ __file_actions,
                   const posix_spawnattr_t *_Restrict_ __attrp,
                   char *const argv[_Restrict_arr_],
                   char *const envp[_Restrict_arr_])
                  _GL_ARG_NONNULL ((2, 5, 6)));
#  endif
_GL_CXXALIAS_SYS (posix_spawn, int,
                  (pid_t *_Restrict_ __pid,
                   const char *_Restrict_ __path,
                   const posix_spawn_file_actions_t *_Restrict_ __file_actions,
                   const posix_spawnattr_t *_Restrict_ __attrp,
                   char *const argv[_Restrict_arr_],
                   char *const envp[_Restrict_arr_]));
# endif
_GL_CXXALIASWARN (posix_spawn);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawn
# if HAVE_RAW_DECL_POSIX_SPAWN
_GL_WARN_ON_USE (posix_spawn, "posix_spawn is unportable - "
                 "use gnulib module posix_spawn for portability");
# endif
#endif

#if 1
/* Similar to 'posix_spawn' but search for FILE in the PATH.

   This function is a possible cancellation points and therefore not
   marked with __THROW.  */
# if 0
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawnp rpl_posix_spawnp
#  endif
_GL_FUNCDECL_RPL (posix_spawnp, int,
                  (pid_t *__pid, const char *__file,
                   const posix_spawn_file_actions_t *__file_actions,
                   const posix_spawnattr_t *__attrp,
                   char *const argv[], char *const envp[])
                  _GL_ARG_NONNULL ((2, 5, 6)));
_GL_CXXALIAS_RPL (posix_spawnp, int,
                  (pid_t *__pid, const char *__file,
                   const posix_spawn_file_actions_t *__file_actions,
                   const posix_spawnattr_t *__attrp,
                   char *const argv[], char *const envp[]));
# else
#  if !0
_GL_FUNCDECL_SYS (posix_spawnp, int,
                  (pid_t *__pid, const char *__file,
                   const posix_spawn_file_actions_t *__file_actions,
                   const posix_spawnattr_t *__attrp,
                   char *const argv[], char *const envp[])
                  _GL_ARG_NONNULL ((2, 5, 6)));
#  endif
_GL_CXXALIAS_SYS (posix_spawnp, int,
                  (pid_t *__pid, const char *__file,
                   const posix_spawn_file_actions_t *__file_actions,
                   const posix_spawnattr_t *__attrp,
                   char *const argv[], char *const envp[]));
# endif
_GL_CXXALIASWARN (posix_spawnp);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawnp
# if HAVE_RAW_DECL_POSIX_SPAWNP
_GL_WARN_ON_USE (posix_spawnp, "posix_spawnp is unportable - "
                 "use gnulib module posix_spawnp for portability");
# endif
#endif


#if 1
/* Initialize data structure with attributes for 'spawn' to default values.  */
# if 0
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawnattr_init rpl_posix_spawnattr_init
#  endif
_GL_FUNCDECL_RPL (posix_spawnattr_init, int, (posix_spawnattr_t *__attr)
                                             __THROW _GL_ARG_NONNULL ((1)));
_GL_CXXALIAS_RPL (posix_spawnattr_init, int, (posix_spawnattr_t *__attr));
# else
#  if !0
_GL_FUNCDECL_SYS (posix_spawnattr_init, int, (posix_spawnattr_t *__attr)
                                             __THROW _GL_ARG_NONNULL ((1)));
#  endif
_GL_CXXALIAS_SYS (posix_spawnattr_init, int, (posix_spawnattr_t *__attr));
# endif
_GL_CXXALIASWARN (posix_spawnattr_init);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawnattr_init
# if HAVE_RAW_DECL_POSIX_SPAWNATTR_INIT
_GL_WARN_ON_USE (posix_spawnattr_init, "posix_spawnattr_init is unportable - "
                 "use gnulib module posix_spawnattr_init for portability");
# endif
#endif

#if 1
/* Free resources associated with ATTR.  */
# if 0
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawnattr_destroy rpl_posix_spawnattr_destroy
#  endif
_GL_FUNCDECL_RPL (posix_spawnattr_destroy, int, (posix_spawnattr_t *__attr)
                                                __THROW _GL_ARG_NONNULL ((1)));
_GL_CXXALIAS_RPL (posix_spawnattr_destroy, int, (posix_spawnattr_t *__attr));
# else
#  if !0
_GL_FUNCDECL_SYS (posix_spawnattr_destroy, int, (posix_spawnattr_t *__attr)
                                                __THROW _GL_ARG_NONNULL ((1)));
#  endif
_GL_CXXALIAS_SYS (posix_spawnattr_destroy, int, (posix_spawnattr_t *__attr));
# endif
_GL_CXXALIASWARN (posix_spawnattr_destroy);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawnattr_destroy
# if HAVE_RAW_DECL_POSIX_SPAWNATTR_DESTROY
_GL_WARN_ON_USE (posix_spawnattr_destroy,
                 "posix_spawnattr_destroy is unportable - "
                 "use gnulib module posix_spawnattr_destroy for portability");
# endif
#endif

#if 0
/* Store signal mask for signals with default handling from ATTR in
   SIGDEFAULT.  */
# if 0
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawnattr_getsigdefault rpl_posix_spawnattr_getsigdefault
#  endif
_GL_FUNCDECL_RPL (posix_spawnattr_getsigdefault, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   sigset_t *_Restrict_ __sigdefault)
                  __THROW _GL_ARG_NONNULL ((1, 2)));
_GL_CXXALIAS_RPL (posix_spawnattr_getsigdefault, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   sigset_t *_Restrict_ __sigdefault));
# else
#  if !0
_GL_FUNCDECL_SYS (posix_spawnattr_getsigdefault, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   sigset_t *_Restrict_ __sigdefault)
                  __THROW _GL_ARG_NONNULL ((1, 2)));
#  endif
_GL_CXXALIAS_SYS (posix_spawnattr_getsigdefault, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   sigset_t *_Restrict_ __sigdefault));
# endif
_GL_CXXALIASWARN (posix_spawnattr_getsigdefault);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawnattr_getsigdefault
# if HAVE_RAW_DECL_POSIX_SPAWNATTR_GETSIGDEFAULT
_GL_WARN_ON_USE (posix_spawnattr_getsigdefault,
                 "posix_spawnattr_getsigdefault is unportable - "
                 "use gnulib module posix_spawnattr_getsigdefault for portability");
# endif
#endif

#if 0
/* Set signal mask for signals with default handling in ATTR to SIGDEFAULT.  */
# if 0
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawnattr_setsigdefault rpl_posix_spawnattr_setsigdefault
#  endif
_GL_FUNCDECL_RPL (posix_spawnattr_setsigdefault, int,
                  (posix_spawnattr_t *_Restrict_ __attr,
                   const sigset_t *_Restrict_ __sigdefault)
                  __THROW _GL_ARG_NONNULL ((1, 2)));
_GL_CXXALIAS_RPL (posix_spawnattr_setsigdefault, int,
                  (posix_spawnattr_t *_Restrict_ __attr,
                   const sigset_t *_Restrict_ __sigdefault));
# else
#  if !0
_GL_FUNCDECL_SYS (posix_spawnattr_setsigdefault, int,
                  (posix_spawnattr_t *_Restrict_ __attr,
                   const sigset_t *_Restrict_ __sigdefault)
                  __THROW _GL_ARG_NONNULL ((1, 2)));
#  endif
_GL_CXXALIAS_SYS (posix_spawnattr_setsigdefault, int,
                  (posix_spawnattr_t *_Restrict_ __attr,
                   const sigset_t *_Restrict_ __sigdefault));
# endif
_GL_CXXALIASWARN (posix_spawnattr_setsigdefault);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawnattr_setsigdefault
# if HAVE_RAW_DECL_POSIX_SPAWNATTR_SETSIGDEFAULT
_GL_WARN_ON_USE (posix_spawnattr_setsigdefault,
                 "posix_spawnattr_setsigdefault is unportable - "
                 "use gnulib module posix_spawnattr_setsigdefault for portability");
# endif
#endif

#if 0
/* Store signal mask for the new process from ATTR in SIGMASK.  */
# if 0
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawnattr_getsigmask rpl_posix_spawnattr_getsigmask
#  endif
_GL_FUNCDECL_RPL (posix_spawnattr_getsigmask, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   sigset_t *_Restrict_ __sigmask)
                  __THROW _GL_ARG_NONNULL ((1, 2)));
_GL_CXXALIAS_RPL (posix_spawnattr_getsigmask, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   sigset_t *_Restrict_ __sigmask));
# else
#  if !0
_GL_FUNCDECL_SYS (posix_spawnattr_getsigmask, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   sigset_t *_Restrict_ __sigmask)
                  __THROW _GL_ARG_NONNULL ((1, 2)));
#  endif
_GL_CXXALIAS_SYS (posix_spawnattr_getsigmask, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   sigset_t *_Restrict_ __sigmask));
# endif
_GL_CXXALIASWARN (posix_spawnattr_getsigmask);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawnattr_getsigmask
# if HAVE_RAW_DECL_POSIX_SPAWNATTR_GETSIGMASK
_GL_WARN_ON_USE (posix_spawnattr_getsigmask,
                 "posix_spawnattr_getsigmask is unportable - "
                 "use gnulib module posix_spawnattr_getsigmask for portability");
# endif
#endif

#if 1
/* Set signal mask for the new process in ATTR to SIGMASK.  */
# if 0
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawnattr_setsigmask rpl_posix_spawnattr_setsigmask
#  endif
_GL_FUNCDECL_RPL (posix_spawnattr_setsigmask, int,
                  (posix_spawnattr_t *_Restrict_ __attr,
                   const sigset_t *_Restrict_ __sigmask)
                  __THROW _GL_ARG_NONNULL ((1, 2)));
_GL_CXXALIAS_RPL (posix_spawnattr_setsigmask, int,
                  (posix_spawnattr_t *_Restrict_ __attr,
                   const sigset_t *_Restrict_ __sigmask));
# else
#  if !0
_GL_FUNCDECL_SYS (posix_spawnattr_setsigmask, int,
                  (posix_spawnattr_t *_Restrict_ __attr,
                   const sigset_t *_Restrict_ __sigmask)
                  __THROW _GL_ARG_NONNULL ((1, 2)));
#  endif
_GL_CXXALIAS_SYS (posix_spawnattr_setsigmask, int,
                  (posix_spawnattr_t *_Restrict_ __attr,
                   const sigset_t *_Restrict_ __sigmask));
# endif
_GL_CXXALIASWARN (posix_spawnattr_setsigmask);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawnattr_setsigmask
# if HAVE_RAW_DECL_POSIX_SPAWNATTR_SETSIGMASK
_GL_WARN_ON_USE (posix_spawnattr_setsigmask,
                 "posix_spawnattr_setsigmask is unportable - "
                 "use gnulib module posix_spawnattr_setsigmask for portability");
# endif
#endif

#if 0
/* Get flag word from the attribute structure.  */
# if 0
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawnattr_getflags rpl_posix_spawnattr_getflags
#  endif
_GL_FUNCDECL_RPL (posix_spawnattr_getflags, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   short int *_Restrict_ __flags)
                  __THROW _GL_ARG_NONNULL ((1, 2)));
_GL_CXXALIAS_RPL (posix_spawnattr_getflags, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   short int *_Restrict_ __flags));
# else
#  if !0
_GL_FUNCDECL_SYS (posix_spawnattr_getflags, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   short int *_Restrict_ __flags)
                  __THROW _GL_ARG_NONNULL ((1, 2)));
#  endif
_GL_CXXALIAS_SYS (posix_spawnattr_getflags, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   short int *_Restrict_ __flags));
# endif
_GL_CXXALIASWARN (posix_spawnattr_getflags);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawnattr_getflags
# if HAVE_RAW_DECL_POSIX_SPAWNATTR_GETFLAGS
_GL_WARN_ON_USE (posix_spawnattr_getflags,
                 "posix_spawnattr_getflags is unportable - "
                 "use gnulib module posix_spawnattr_getflags for portability");
# endif
#endif

#if 1
/* Store flags in the attribute structure.  */
# if 0
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawnattr_setflags rpl_posix_spawnattr_setflags
#  endif
_GL_FUNCDECL_RPL (posix_spawnattr_setflags, int,
                  (posix_spawnattr_t *__attr, short int __flags)
                  __THROW _GL_ARG_NONNULL ((1)));
_GL_CXXALIAS_RPL (posix_spawnattr_setflags, int,
                  (posix_spawnattr_t *__attr, short int __flags));
# else
#  if !0
_GL_FUNCDECL_SYS (posix_spawnattr_setflags, int,
                  (posix_spawnattr_t *__attr, short int __flags)
                  __THROW _GL_ARG_NONNULL ((1)));
#  endif
_GL_CXXALIAS_SYS (posix_spawnattr_setflags, int,
                  (posix_spawnattr_t *__attr, short int __flags));
# endif
_GL_CXXALIASWARN (posix_spawnattr_setflags);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawnattr_setflags
# if HAVE_RAW_DECL_POSIX_SPAWNATTR_SETFLAGS
_GL_WARN_ON_USE (posix_spawnattr_setflags,
                 "posix_spawnattr_setflags is unportable - "
                 "use gnulib module posix_spawnattr_setflags for portability");
# endif
#endif

#if 0
/* Get process group ID from the attribute structure.  */
# if 0
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawnattr_getpgroup rpl_posix_spawnattr_getpgroup
#  endif
_GL_FUNCDECL_RPL (posix_spawnattr_getpgroup, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   pid_t *_Restrict_ __pgroup)
                  __THROW _GL_ARG_NONNULL ((1, 2)));
_GL_CXXALIAS_RPL (posix_spawnattr_getpgroup, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   pid_t *_Restrict_ __pgroup));
# else
#  if !0
_GL_FUNCDECL_SYS (posix_spawnattr_getpgroup, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   pid_t *_Restrict_ __pgroup)
                  __THROW _GL_ARG_NONNULL ((1, 2)));
#  endif
_GL_CXXALIAS_SYS (posix_spawnattr_getpgroup, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   pid_t *_Restrict_ __pgroup));
# endif
_GL_CXXALIASWARN (posix_spawnattr_getpgroup);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawnattr_getpgroup
# if HAVE_RAW_DECL_POSIX_SPAWNATTR_GETPGROUP
_GL_WARN_ON_USE (posix_spawnattr_getpgroup,
                 "posix_spawnattr_getpgroup is unportable - "
                 "use gnulib module posix_spawnattr_getpgroup for portability");
# endif
#endif

#if 0
/* Store process group ID in the attribute structure.  */
# if 0
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawnattr_setpgroup rpl_posix_spawnattr_setpgroup
#  endif
_GL_FUNCDECL_RPL (posix_spawnattr_setpgroup, int,
                  (posix_spawnattr_t *__attr, pid_t __pgroup)
                  __THROW _GL_ARG_NONNULL ((1)));
_GL_CXXALIAS_RPL (posix_spawnattr_setpgroup, int,
                  (posix_spawnattr_t *__attr, pid_t __pgroup));
# else
#  if !0
_GL_FUNCDECL_SYS (posix_spawnattr_setpgroup, int,
                  (posix_spawnattr_t *__attr, pid_t __pgroup)
                  __THROW _GL_ARG_NONNULL ((1)));
#  endif
_GL_CXXALIAS_SYS (posix_spawnattr_setpgroup, int,
                  (posix_spawnattr_t *__attr, pid_t __pgroup));
# endif
_GL_CXXALIASWARN (posix_spawnattr_setpgroup);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawnattr_setpgroup
# if HAVE_RAW_DECL_POSIX_SPAWNATTR_SETPGROUP
_GL_WARN_ON_USE (posix_spawnattr_setpgroup,
                 "posix_spawnattr_setpgroup is unportable - "
                 "use gnulib module posix_spawnattr_setpgroup for portability");
# endif
#endif

#if 0
/* Get scheduling policy from the attribute structure.  */
# if 0
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawnattr_getschedpolicy rpl_posix_spawnattr_getschedpolicy
#  endif
_GL_FUNCDECL_RPL (posix_spawnattr_getschedpolicy, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   int *_Restrict_ __schedpolicy)
                  __THROW _GL_ARG_NONNULL ((1, 2)));
_GL_CXXALIAS_RPL (posix_spawnattr_getschedpolicy, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   int *_Restrict_ __schedpolicy));
# else
#  if !0 || POSIX_SPAWN_SETSCHEDULER == 0
_GL_FUNCDECL_SYS (posix_spawnattr_getschedpolicy, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   int *_Restrict_ __schedpolicy)
                  __THROW _GL_ARG_NONNULL ((1, 2)));
#  endif
_GL_CXXALIAS_SYS (posix_spawnattr_getschedpolicy, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   int *_Restrict_ __schedpolicy));
# endif
_GL_CXXALIASWARN (posix_spawnattr_getschedpolicy);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawnattr_getschedpolicy
# if HAVE_RAW_DECL_POSIX_SPAWNATTR_GETSCHEDPOLICY
_GL_WARN_ON_USE (posix_spawnattr_getschedpolicy,
                 "posix_spawnattr_getschedpolicy is unportable - "
                 "use gnulib module posix_spawnattr_getschedpolicy for portability");
# endif
#endif

#if 0
/* Store scheduling policy in the attribute structure.  */
# if 0
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawnattr_setschedpolicy rpl_posix_spawnattr_setschedpolicy
#  endif
_GL_FUNCDECL_RPL (posix_spawnattr_setschedpolicy, int,
                  (posix_spawnattr_t *__attr, int __schedpolicy)
                  __THROW _GL_ARG_NONNULL ((1)));
_GL_CXXALIAS_RPL (posix_spawnattr_setschedpolicy, int,
                  (posix_spawnattr_t *__attr, int __schedpolicy));
# else
#  if !0 || POSIX_SPAWN_SETSCHEDULER == 0
_GL_FUNCDECL_SYS (posix_spawnattr_setschedpolicy, int,
                  (posix_spawnattr_t *__attr, int __schedpolicy)
                  __THROW _GL_ARG_NONNULL ((1)));
#  endif
_GL_CXXALIAS_SYS (posix_spawnattr_setschedpolicy, int,
                  (posix_spawnattr_t *__attr, int __schedpolicy));
# endif
_GL_CXXALIASWARN (posix_spawnattr_setschedpolicy);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawnattr_setschedpolicy
# if HAVE_RAW_DECL_POSIX_SPAWNATTR_SETSCHEDPOLICY
_GL_WARN_ON_USE (posix_spawnattr_setschedpolicy,
                 "posix_spawnattr_setschedpolicy is unportable - "
                 "use gnulib module posix_spawnattr_setschedpolicy for portability");
# endif
#endif

#if 0
/* Get scheduling parameters from the attribute structure.  */
# if 0
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawnattr_getschedparam rpl_posix_spawnattr_getschedparam
#  endif
_GL_FUNCDECL_RPL (posix_spawnattr_getschedparam, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   struct sched_param *_Restrict_ __schedparam)
                  __THROW _GL_ARG_NONNULL ((1, 2)));
_GL_CXXALIAS_RPL (posix_spawnattr_getschedparam, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   struct sched_param *_Restrict_ __schedparam));
# else
#  if !0 || POSIX_SPAWN_SETSCHEDPARAM == 0
_GL_FUNCDECL_SYS (posix_spawnattr_getschedparam, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   struct sched_param *_Restrict_ __schedparam)
                  __THROW _GL_ARG_NONNULL ((1, 2)));
#  endif
_GL_CXXALIAS_SYS (posix_spawnattr_getschedparam, int,
                  (const posix_spawnattr_t *_Restrict_ __attr,
                   struct sched_param *_Restrict_ __schedparam));
# endif
_GL_CXXALIASWARN (posix_spawnattr_getschedparam);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawnattr_getschedparam
# if HAVE_RAW_DECL_POSIX_SPAWNATTR_GETSCHEDPARAM
_GL_WARN_ON_USE (posix_spawnattr_getschedparam,
                 "posix_spawnattr_getschedparam is unportable - "
                 "use gnulib module posix_spawnattr_getschedparam for portability");
# endif
#endif

#if 0
/* Store scheduling parameters in the attribute structure.  */
# if 0
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawnattr_setschedparam rpl_posix_spawnattr_setschedparam
#  endif
_GL_FUNCDECL_RPL (posix_spawnattr_setschedparam, int,
                  (posix_spawnattr_t *_Restrict_ __attr,
                   const struct sched_param *_Restrict_ __schedparam)
                  __THROW _GL_ARG_NONNULL ((1, 2)));
_GL_CXXALIAS_RPL (posix_spawnattr_setschedparam, int,
                  (posix_spawnattr_t *_Restrict_ __attr,
                   const struct sched_param *_Restrict_ __schedparam));
# else
#  if !0 || POSIX_SPAWN_SETSCHEDPARAM == 0
_GL_FUNCDECL_SYS (posix_spawnattr_setschedparam, int,
                  (posix_spawnattr_t *_Restrict_ __attr,
                   const struct sched_param *_Restrict_ __schedparam)
                  __THROW _GL_ARG_NONNULL ((1, 2)));
#  endif
_GL_CXXALIAS_SYS (posix_spawnattr_setschedparam, int,
                  (posix_spawnattr_t *_Restrict_ __attr,
                   const struct sched_param *_Restrict_ __schedparam));
# endif
_GL_CXXALIASWARN (posix_spawnattr_setschedparam);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawnattr_setschedparam
# if HAVE_RAW_DECL_POSIX_SPAWNATTR_SETSCHEDPARAM
_GL_WARN_ON_USE (posix_spawnattr_setschedparam,
                 "posix_spawnattr_setschedparam is unportable - "
                 "use gnulib module posix_spawnattr_setschedparam for portability");
# endif
#endif


#if 1
/* Initialize data structure for file attribute for 'spawn' call.  */
# if 0
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawn_file_actions_init rpl_posix_spawn_file_actions_init
#  endif
_GL_FUNCDECL_RPL (posix_spawn_file_actions_init, int,
                  (posix_spawn_file_actions_t *__file_actions)
                  __THROW _GL_ARG_NONNULL ((1)));
_GL_CXXALIAS_RPL (posix_spawn_file_actions_init, int,
                  (posix_spawn_file_actions_t *__file_actions));
# else
#  if !0
_GL_FUNCDECL_SYS (posix_spawn_file_actions_init, int,
                  (posix_spawn_file_actions_t *__file_actions)
                  __THROW _GL_ARG_NONNULL ((1)));
#  endif
_GL_CXXALIAS_SYS (posix_spawn_file_actions_init, int,
                  (posix_spawn_file_actions_t *__file_actions));
# endif
_GL_CXXALIASWARN (posix_spawn_file_actions_init);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawn_file_actions_init
# if HAVE_RAW_DECL_POSIX_SPAWN_FILE_ACTIONS_INIT
_GL_WARN_ON_USE (posix_spawn_file_actions_init,
                 "posix_spawn_file_actions_init is unportable - "
                 "use gnulib module posix_spawn_file_actions_init for portability");
# endif
#endif

#if 1
/* Free resources associated with FILE-ACTIONS.  */
# if 0
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawn_file_actions_destroy rpl_posix_spawn_file_actions_destroy
#  endif
_GL_FUNCDECL_RPL (posix_spawn_file_actions_destroy, int,
                  (posix_spawn_file_actions_t *__file_actions)
                  __THROW _GL_ARG_NONNULL ((1)));
_GL_CXXALIAS_RPL (posix_spawn_file_actions_destroy, int,
                  (posix_spawn_file_actions_t *__file_actions));
# else
#  if !0
_GL_FUNCDECL_SYS (posix_spawn_file_actions_destroy, int,
                  (posix_spawn_file_actions_t *__file_actions)
                  __THROW _GL_ARG_NONNULL ((1)));
#  endif
_GL_CXXALIAS_SYS (posix_spawn_file_actions_destroy, int,
                  (posix_spawn_file_actions_t *__file_actions));
# endif
_GL_CXXALIASWARN (posix_spawn_file_actions_destroy);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawn_file_actions_destroy
# if HAVE_RAW_DECL_POSIX_SPAWN_FILE_ACTIONS_DESTROY
_GL_WARN_ON_USE (posix_spawn_file_actions_destroy,
                 "posix_spawn_file_actions_destroy is unportable - "
                 "use gnulib module posix_spawn_file_actions_destroy for portability");
# endif
#endif

#if 1
/* Add an action to FILE-ACTIONS which tells the implementation to call
   'open' for the given file during the 'spawn' call.  */
# if 1
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawn_file_actions_addopen rpl_posix_spawn_file_actions_addopen
#  endif
_GL_FUNCDECL_RPL (posix_spawn_file_actions_addopen, int,
                  (posix_spawn_file_actions_t *_Restrict_ __file_actions,
                   int __fd,
                   const char *_Restrict_ __path, int __oflag, mode_t __mode)
                  __THROW _GL_ARG_NONNULL ((1, 3)));
_GL_CXXALIAS_RPL (posix_spawn_file_actions_addopen, int,
                  (posix_spawn_file_actions_t *_Restrict_ __file_actions,
                   int __fd,
                   const char *_Restrict_ __path, int __oflag, mode_t __mode));
# else
#  if !0
_GL_FUNCDECL_SYS (posix_spawn_file_actions_addopen, int,
                  (posix_spawn_file_actions_t *_Restrict_ __file_actions,
                   int __fd,
                   const char *_Restrict_ __path, int __oflag, mode_t __mode)
                  __THROW _GL_ARG_NONNULL ((1, 3)));
#  endif
_GL_CXXALIAS_SYS (posix_spawn_file_actions_addopen, int,
                  (posix_spawn_file_actions_t *_Restrict_ __file_actions,
                   int __fd,
                   const char *_Restrict_ __path, int __oflag, mode_t __mode));
# endif
_GL_CXXALIASWARN (posix_spawn_file_actions_addopen);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawn_file_actions_addopen
# if HAVE_RAW_DECL_POSIX_SPAWN_FILE_ACTIONS_ADDOPEN
_GL_WARN_ON_USE (posix_spawn_file_actions_addopen,
                 "posix_spawn_file_actions_addopen is unportable - "
                 "use gnulib module posix_spawn_file_actions_addopen for portability");
# endif
#endif

#if 1
/* Add an action to FILE-ACTIONS which tells the implementation to call
   'close' for the given file descriptor during the 'spawn' call.  */
# if 1
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawn_file_actions_addclose rpl_posix_spawn_file_actions_addclose
#  endif
_GL_FUNCDECL_RPL (posix_spawn_file_actions_addclose, int,
                  (posix_spawn_file_actions_t *__file_actions, int __fd)
                  __THROW _GL_ARG_NONNULL ((1)));
_GL_CXXALIAS_RPL (posix_spawn_file_actions_addclose, int,
                  (posix_spawn_file_actions_t *__file_actions, int __fd));
# else
#  if !0
_GL_FUNCDECL_SYS (posix_spawn_file_actions_addclose, int,
                  (posix_spawn_file_actions_t *__file_actions, int __fd)
                  __THROW _GL_ARG_NONNULL ((1)));
#  endif
_GL_CXXALIAS_SYS (posix_spawn_file_actions_addclose, int,
                  (posix_spawn_file_actions_t *__file_actions, int __fd));
# endif
_GL_CXXALIASWARN (posix_spawn_file_actions_addclose);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawn_file_actions_addclose
# if HAVE_RAW_DECL_POSIX_SPAWN_FILE_ACTIONS_ADDCLOSE
_GL_WARN_ON_USE (posix_spawn_file_actions_addclose,
                 "posix_spawn_file_actions_addclose is unportable - "
                 "use gnulib module posix_spawn_file_actions_addclose for portability");
# endif
#endif

#if 1
/* Add an action to FILE-ACTIONS which tells the implementation to call
   'dup2' for the given file descriptors during the 'spawn' call.  */
# if 1
#  if !(defined __cplusplus && defined GNULIB_NAMESPACE)
#   define posix_spawn_file_actions_adddup2 rpl_posix_spawn_file_actions_adddup2
#  endif
_GL_FUNCDECL_RPL (posix_spawn_file_actions_adddup2, int,
                  (posix_spawn_file_actions_t *__file_actions,
                   int __fd, int __newfd)
                  __THROW _GL_ARG_NONNULL ((1)));
_GL_CXXALIAS_RPL (posix_spawn_file_actions_adddup2, int,
                  (posix_spawn_file_actions_t *__file_actions,
                   int __fd, int __newfd));
# else
#  if !0
_GL_FUNCDECL_SYS (posix_spawn_file_actions_adddup2, int,
                  (posix_spawn_file_actions_t *__file_actions,
                   int __fd, int __newfd)
                  __THROW _GL_ARG_NONNULL ((1)));
#  endif
_GL_CXXALIAS_SYS (posix_spawn_file_actions_adddup2, int,
                  (posix_spawn_file_actions_t *__file_actions,
                   int __fd, int __newfd));
# endif
_GL_CXXALIASWARN (posix_spawn_file_actions_adddup2);
#elif defined GNULIB_POSIXCHECK
# undef posix_spawn_file_actions_adddup2
# if HAVE_RAW_DECL_POSIX_SPAWN_FILE_ACTIONS_ADDDUP2
_GL_WARN_ON_USE (posix_spawn_file_actions_adddup2,
                 "posix_spawn_file_actions_adddup2 is unportable - "
                 "use gnulib module posix_spawn_file_actions_adddup2 for portability");
# endif
#endif


#endif /* _GL_M4_SPAWN_H */
#endif /* _GL_M4_SPAWN_H */
