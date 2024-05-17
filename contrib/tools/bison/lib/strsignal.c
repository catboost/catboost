/* Copyright (C) 1991, 1994-2002, 2005, 2008-2013 Free Software Foundation,
   Inc.
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

#ifndef _LIBC
# include <config.h>
#endif

/* Specification.  */
#include <string.h>

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _LIBC
# include <libintl.h>
#else /* !_LIBC */
# include "gettext.h"
# define _(msgid) gettext (msgid)
# define N_(msgid) gettext_noop (msgid)
#endif /* _LIBC */

#ifdef _LIBC
# include <bits/libc-lock.h>
#else /* !_LIBC */
# include "glthread/lock.h"
# include "glthread/tls.h"
# define __libc_once_define(CLASS, NAME) gl_once_define (CLASS, NAME)
# define __libc_once(NAME, INIT) gl_once ((NAME), (INIT))
# define __libc_key_t gl_tls_key_t
# define __libc_getspecific(NAME) gl_tls_get ((NAME))
# define __libc_setspecific(NAME, POINTER) gl_tls_set ((NAME), (POINTER))
#if defined(_MSC_VER)
# define __snprintf _snprintf
#else
# define __snprintf snprintf
#endif
#endif /* _LIBC */

#ifdef _LIBC

/* Defined in siglist.c.  */
extern const char *const _sys_siglist[];
extern const char *const _sys_siglist_internal[] attribute_hidden;

#else /* !_LIBC */

/* NetBSD declares sys_siglist in unistd.h. */
# if HAVE_UNISTD_H
#  include <unistd.h>
# endif

# define INTUSE(x) (x)

# if HAVE_DECL_SYS_SIGLIST
#  undef _sys_siglist
#  define _sys_siglist sys_siglist
# else /* !HAVE_DECL_SYS_SIGLIST */
#  ifndef NSIG
#   define NSIG 32
#  endif /* NSIG */
#  if !HAVE_DECL__SYS_SIGLIST
static const char *_sys_siglist[NSIG];
#  endif
# endif /* !HAVE_DECL_SYS_SIGLIST */

#endif /* _LIBC */

static __libc_key_t key;

/* If nonzero the key allocation failed and we should better use a
   static buffer than fail.  */
#define BUFFERSIZ       100
static char local_buf[BUFFERSIZ];
static char *static_buf;

/* Destructor for the thread-specific data.  */
static void init (void);
static void free_key_mem (void *mem);
static char *getbuffer (void);


/* Return a string describing the meaning of the signal number SIGNUM.  */
char *
strsignal (int signum)
{
  const char *desc;
  __libc_once_define (static, once);

  /* If we have not yet initialized the buffer do it now.  */
  __libc_once (once, init);

  if (
#ifdef SIGRTMIN
      (signum >= SIGRTMIN && signum <= SIGRTMAX) ||
#endif
      signum < 0 || signum >= NSIG
      || (desc = INTUSE(_sys_siglist)[signum]) == NULL)
    {
      char *buffer = getbuffer ();
      int len;
#ifdef SIGRTMIN
      if (signum >= SIGRTMIN && signum <= SIGRTMAX)
        len = __snprintf (buffer, BUFFERSIZ - 1, _("Real-time signal %d"),
                          signum - (int) SIGRTMIN);
      else
#endif
        len = __snprintf (buffer, BUFFERSIZ - 1, _("Unknown signal %d"),
                          signum);
      if (len >= BUFFERSIZ)
        buffer = NULL;
      else
        buffer[len] = '\0';

      return buffer;
    }

  return (char *) _(desc);
}


/* Initialize buffer.  */
static void
init (void)
{
#ifdef _LIBC
  if (__libc_key_create (&key, free_key_mem))
    /* Creating the key failed.  This means something really went
       wrong.  In any case use a static buffer which is better than
       nothing.  */
    static_buf = local_buf;
#else /* !_LIBC */
  gl_tls_key_init (key, free_key_mem);

# if !HAVE_DECL_SYS_SIGLIST
  memset (_sys_siglist, 0, NSIG * sizeof *_sys_siglist);

  /* No need to use a do {} while (0) here since init_sig(...) must expand
     to a complete statement.  (We cannot use the ISO C99 designated array
     initializer syntax since it is not supported by ANSI C compilers and
     since some signal numbers might exceed NSIG.)  */
#  define init_sig(sig, abbrev, desc) \
  if (sig >= 0 && sig < NSIG) \
    _sys_siglist[sig] = desc;

#  include "siglist.h"

#  undef init_sig

# endif /* !HAVE_DECL_SYS_SIGLIST */
#endif /* !_LIBC */
}


/* Free the thread specific data, this is done if a thread terminates.  */
static void
free_key_mem (void *mem)
{
  free (mem);
  __libc_setspecific (key, NULL);
}


/* Return the buffer to be used.  */
static char *
getbuffer (void)
{
  char *result;

  if (static_buf != NULL)
    result = static_buf;
  else
    {
      /* We don't use the static buffer and so we have a key.  Use it
         to get the thread-specific buffer.  */
      result = __libc_getspecific (key);
      if (result == NULL)
        {
          /* No buffer allocated so far.  */
          result = malloc (BUFFERSIZ);
          if (result == NULL)
            /* No more memory available.  We use the static buffer.  */
            result = local_buf;
          else
            __libc_setspecific (key, result);
        }
    }

  return result;
}
