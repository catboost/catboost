/* Copyright (C) 1999, 2001-2002, 2006, 2009-2013 Free Software Foundation,
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

/* Extracted from sysdeps/posix/tempname.c.  */

#include <config.h>

/* Specification.  */
#include "tmpdir.h"

#include <stdbool.h>
#include "stdlib--.h"
#include <string.h>

#include <errno.h>
#ifndef __set_errno
# define __set_errno(Val) errno = (Val)
#endif

#include <stdio.h>
#ifndef P_tmpdir
# ifdef _P_tmpdir /* native Windows */
#  define P_tmpdir _P_tmpdir
# else
#  define P_tmpdir "/tmp"
# endif
#endif

#include <sys/stat.h>

#if (defined _WIN32 || defined __WIN32__) && ! defined __CYGWIN__
# define WIN32_LEAN_AND_MEAN  /* avoid including junk */
# include <windows.h>
#endif

#include "pathmax.h"

#if _LIBC
# define struct_stat64 struct stat64
#else
# define struct_stat64 struct stat
# define __libc_secure_getenv secure_getenv
# define __xstat64(version, path, buf) stat (path, buf)
#endif

/* Pathname support.
   ISSLASH(C)           tests whether C is a directory separator character.
 */
#if defined _WIN32 || defined __WIN32__ || defined __CYGWIN__ || defined __EMX__ || defined __DJGPP__
  /* Native Windows, Cygwin, OS/2, DOS */
# define ISSLASH(C) ((C) == '/' || (C) == '\\')
#else
  /* Unix */
# define ISSLASH(C) ((C) == '/')
#endif


/* Return nonzero if DIR is an existent directory.  */
static bool
direxists (const char *dir)
{
  struct_stat64 buf;
  return __xstat64 (_STAT_VER, dir, &buf) == 0 && S_ISDIR (buf.st_mode);
}

/* Path search algorithm, for tmpnam, tmpfile, etc.  If DIR is
   non-null and exists, uses it; otherwise uses the first of $TMPDIR,
   P_tmpdir, /tmp that exists.  Copies into TMPL a template suitable
   for use with mk[s]temp.  Will fail (-1) if DIR is non-null and
   doesn't exist, none of the searched dirs exists, or there's not
   enough space in TMPL. */
int
path_search (char *tmpl, size_t tmpl_len, const char *dir, const char *pfx,
             bool try_tmpdir)
{
  const char *d;
  size_t dlen, plen;
  bool add_slash;

  if (!pfx || !pfx[0])
    {
      pfx = "file";
      plen = 4;
    }
  else
    {
      plen = strlen (pfx);
      if (plen > 5)
        plen = 5;
    }

  if (try_tmpdir)
    {
      d = __libc_secure_getenv ("TMPDIR");
      if (d != NULL && direxists (d))
        dir = d;
      else if (dir != NULL && direxists (dir))
        /* nothing */ ;
      else
        dir = NULL;
    }
  if (dir == NULL)
    {
#if (defined _WIN32 || defined __WIN32__) && ! defined __CYGWIN__
      char dirbuf[PATH_MAX];
      DWORD retval;

      /* Find Windows temporary file directory.
         We try this before P_tmpdir because Windows defines P_tmpdir to "\\"
         and will therefore try to put all temporary files in the root
         directory (unless $TMPDIR is set).  */
      retval = GetTempPath (PATH_MAX, dirbuf);
      if (retval > 0 && retval < PATH_MAX && direxists (dirbuf))
        dir = dirbuf;
      else
#endif
      if (direxists (P_tmpdir))
        dir = P_tmpdir;
      else if (strcmp (P_tmpdir, "/tmp") != 0 && direxists ("/tmp"))
        dir = "/tmp";
      else
        {
          __set_errno (ENOENT);
          return -1;
        }
    }

  dlen = strlen (dir);
#ifdef __VMS
  add_slash = 0;
#else
  add_slash = dlen != 0 && !ISSLASH (dir[dlen - 1]);
#endif

  /* check we have room for "${dir}/${pfx}XXXXXX\0" */
  if (tmpl_len < dlen + add_slash + plen + 6 + 1)
    {
      __set_errno (EINVAL);
      return -1;
    }

  memcpy (tmpl, dir, dlen);
  sprintf (tmpl + dlen, &"/%.*sXXXXXX"[!add_slash], (int) plen, pfx);
  return 0;
}
