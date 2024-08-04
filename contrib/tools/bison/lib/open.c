/* Open a descriptor to a file.
   Copyright (C) 2007-2020 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* Written by Bruno Haible <bruno@clisp.org>, 2007.  */

/* If the user's config.h happens to include <fcntl.h>, let it include only
   the system's <fcntl.h> here, so that orig_open doesn't recurse to
   rpl_open.  */
#define __need_system_fcntl_h
#include <config.h>

/* Get the original definition of open.  It might be defined as a macro.  */
#include <fcntl.h>
#include <sys/types.h>
#undef __need_system_fcntl_h

static int
orig_open (const char *filename, int flags, mode_t mode)
{
  return open (filename, flags, mode);
}

/* Specification.  */
/* Write "fcntl.h" here, not <fcntl.h>, otherwise OSF/1 5.1 DTK cc eliminates
   this include because of the preliminary #include <fcntl.h> above.  */
#include "fcntl.h"

#include "cloexec.h"

#include <errno.h>
#include <stdarg.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef REPLACE_OPEN_DIRECTORY
# define REPLACE_OPEN_DIRECTORY 0
#endif

int
open (const char *filename, int flags, ...)
{
  /* 0 = unknown, 1 = yes, -1 = no.  */
#if GNULIB_defined_O_CLOEXEC
  int have_cloexec = -1;
#else
  static int have_cloexec;
#endif

  mode_t mode;
  int fd;

  mode = 0;
  if (flags & O_CREAT)
    {
      va_list arg;
      va_start (arg, flags);

      /* We have to use PROMOTED_MODE_T instead of mode_t, otherwise GCC 4
         creates crashing code when 'mode_t' is smaller than 'int'.  */
      mode = va_arg (arg, PROMOTED_MODE_T);

      va_end (arg);
    }

#if GNULIB_defined_O_NONBLOCK
  /* The only known platform that lacks O_NONBLOCK is mingw, but it
     also lacks named pipes and Unix sockets, which are the only two
     file types that require non-blocking handling in open().
     Therefore, it is safe to ignore O_NONBLOCK here.  It is handy
     that mingw also lacks openat(), so that is also covered here.  */
  flags &= ~O_NONBLOCK;
#endif

#if defined _WIN32 && ! defined __CYGWIN__
  if (strcmp (filename, "/dev/null") == 0)
    filename = "NUL";
#endif

#if OPEN_TRAILING_SLASH_BUG
  /* Fail if one of O_CREAT, O_WRONLY, O_RDWR is specified and the filename
     ends in a slash, as POSIX says such a filename must name a directory
     <https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap04.html#tag_04_13>:
       "A pathname that contains at least one non-<slash> character and that
        ends with one or more trailing <slash> characters shall not be resolved
        successfully unless the last pathname component before the trailing
        <slash> characters names an existing directory"
     If the named file already exists as a directory, then
       - if O_CREAT is specified, open() must fail because of the semantics
         of O_CREAT,
       - if O_WRONLY or O_RDWR is specified, open() must fail because POSIX
         <https://pubs.opengroup.org/onlinepubs/9699919799/functions/open.html>
         says that it fails with errno = EISDIR in this case.
     If the named file does not exist or does not name a directory, then
       - if O_CREAT is specified, open() must fail since open() cannot create
         directories,
       - if O_WRONLY or O_RDWR is specified, open() must fail because the
         file does not contain a '.' directory.  */
  if (flags & (O_CREAT | O_WRONLY | O_RDWR))
    {
      size_t len = strlen (filename);
      if (len > 0 && filename[len - 1] == '/')
        {
          errno = EISDIR;
          return -1;
        }
    }
#endif

  fd = orig_open (filename,
                  flags & ~(have_cloexec <= 0 ? O_CLOEXEC : 0), mode);

  if (flags & O_CLOEXEC)
    {
      if (! have_cloexec)
        {
          if (0 <= fd)
            have_cloexec = 1;
          else if (errno == EINVAL)
            {
              fd = orig_open (filename, flags & ~O_CLOEXEC, mode);
              have_cloexec = -1;
            }
        }
      if (have_cloexec < 0 && 0 <= fd)
        set_cloexec_flag (fd, true);
    }


#if REPLACE_FCHDIR
  /* Implementing fchdir and fdopendir requires the ability to open a
     directory file descriptor.  If open doesn't support that (as on
     mingw), we use a dummy file that behaves the same as directories
     on Linux (ie. always reports EOF on attempts to read()), and
     override fstat() in fchdir.c to hide the fact that we have a
     dummy.  */
  if (REPLACE_OPEN_DIRECTORY && fd < 0 && errno == EACCES
      && ((flags & O_ACCMODE) == O_RDONLY
          || (O_SEARCH != O_RDONLY && (flags & O_ACCMODE) == O_SEARCH)))
    {
      struct stat statbuf;
      if (stat (filename, &statbuf) == 0 && S_ISDIR (statbuf.st_mode))
        {
          /* Maximum recursion depth of 1.  */
          fd = open ("/dev/null", flags, mode);
          if (0 <= fd)
            fd = _gl_register_fd (fd, filename);
        }
      else
        errno = EACCES;
    }
#endif

#if OPEN_TRAILING_SLASH_BUG
  /* If the filename ends in a slash and fd does not refer to a directory,
     then fail.
     Rationale: POSIX says such a filename must name a directory
     <https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap04.html#tag_04_13>:
       "A pathname that contains at least one non-<slash> character and that
        ends with one or more trailing <slash> characters shall not be resolved
        successfully unless the last pathname component before the trailing
        <slash> characters names an existing directory"
     If the named file without the slash is not a directory, open() must fail
     with ENOTDIR.  */
  if (fd >= 0)
    {
      /* We know len is positive, since open did not fail with ENOENT.  */
      size_t len = strlen (filename);
      if (filename[len - 1] == '/')
        {
          struct stat statbuf;

          if (fstat (fd, &statbuf) >= 0 && !S_ISDIR (statbuf.st_mode))
            {
              close (fd);
              errno = ENOTDIR;
              return -1;
            }
        }
    }
#endif

#if REPLACE_FCHDIR
  if (!REPLACE_OPEN_DIRECTORY && 0 <= fd)
    fd = _gl_register_fd (fd, filename);
#endif

  return fd;
}
