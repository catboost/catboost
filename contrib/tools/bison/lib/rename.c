/* Work around rename bugs in some systems.

   Copyright (C) 2001-2003, 2005-2006, 2009-2013 Free Software Foundation, Inc.

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

/* Written by Volker Borchert, Eric Blake.  */

#include <config.h>

#include <stdio.h>

#undef rename

#if (defined _WIN32 || defined __WIN32__) && ! defined __CYGWIN__
/* The mingw rename has problems with trailing slashes; it also
   requires use of native Windows calls to allow atomic renames over
   existing files.  */

# include <errno.h>
# include <stdbool.h>
# include <stdlib.h>
# include <sys/stat.h>
# include <unistd.h>

# define WIN32_LEAN_AND_MEAN
# include <windows.h>

# include "dirname.h"

/* Rename the file SRC to DST.  This replacement is necessary on
   Windows, on which the system rename function will not replace
   an existing DST.  */
int
rpl_rename (char const *src, char const *dst)
{
  int error;
  size_t src_len = strlen (src);
  size_t dst_len = strlen (dst);
  char *src_base = last_component (src);
  char *dst_base = last_component (dst);
  bool src_slash;
  bool dst_slash;
  bool dst_exists;
  struct stat src_st;
  struct stat dst_st;

  /* Filter out dot as last component.  */
  if (!src_len || !dst_len)
    {
      errno = ENOENT;
      return -1;
    }
  if (*src_base == '.')
    {
      size_t len = base_len (src_base);
      if (len == 1 || (len == 2 && src_base[1] == '.'))
        {
          errno = EINVAL;
          return -1;
        }
    }
  if (*dst_base == '.')
    {
      size_t len = base_len (dst_base);
      if (len == 1 || (len == 2 && dst_base[1] == '.'))
        {
          errno = EINVAL;
          return -1;
        }
    }

  /* Presence of a trailing slash requires directory semantics.  If
     the source does not exist, or if the destination cannot be turned
     into a directory, give up now.  Otherwise, strip trailing slashes
     before calling rename.  There are no symlinks on mingw, so stat
     works instead of lstat.  */
  src_slash = ISSLASH (src[src_len - 1]);
  dst_slash = ISSLASH (dst[dst_len - 1]);
  if (stat (src, &src_st))
    return -1;
  if (stat (dst, &dst_st))
    {
      if (errno != ENOENT || (!S_ISDIR (src_st.st_mode) && dst_slash))
        return -1;
      dst_exists = false;
    }
  else
    {
      if (S_ISDIR (dst_st.st_mode) != S_ISDIR (src_st.st_mode))
        {
          errno = S_ISDIR (dst_st.st_mode) ? EISDIR : ENOTDIR;
          return -1;
        }
      dst_exists = true;
    }

  /* There are no symlinks, so if a file existed with a trailing
     slash, it must be a directory, and we don't have to worry about
     stripping strip trailing slash.  However, mingw refuses to
     replace an existing empty directory, so we have to help it out.
     And canonicalize_file_name is not yet ported to mingw; however,
     for directories, getcwd works as a viable alternative.  Ensure
     that we can get back to where we started before using it; later
     attempts to return are fatal.  Note that we can end up losing a
     directory if rename then fails, but it was empty, so not much
     damage was done.  */
  if (dst_exists && S_ISDIR (dst_st.st_mode))
    {
      char *cwd = getcwd (NULL, 0);
      char *src_temp;
      char *dst_temp;
      if (!cwd || chdir (cwd))
        return -1;
      if (IS_ABSOLUTE_FILE_NAME (src))
        {
          dst_temp = chdir (dst) ? NULL : getcwd (NULL, 0);
          src_temp = chdir (src) ? NULL : getcwd (NULL, 0);
        }
      else
        {
          src_temp = chdir (src) ? NULL : getcwd (NULL, 0);
          if (!IS_ABSOLUTE_FILE_NAME (dst) && chdir (cwd))
            abort ();
          dst_temp = chdir (dst) ? NULL : getcwd (NULL, 0);
        }
      if (chdir (cwd))
        abort ();
      free (cwd);
      if (!src_temp || !dst_temp)
        {
          free (src_temp);
          free (dst_temp);
          errno = ENOMEM;
          return -1;
        }
      src_len = strlen (src_temp);
      if (strncmp (src_temp, dst_temp, src_len) == 0
          && (ISSLASH (dst_temp[src_len]) || dst_temp[src_len] == '\0'))
        {
          error = dst_temp[src_len];
          free (src_temp);
          free (dst_temp);
          if (error)
            {
              errno = EINVAL;
              return -1;
            }
          return 0;
        }
      if (rmdir (dst))
        {
          error = errno;
          free (src_temp);
          free (dst_temp);
          errno = error;
          return -1;
        }
      free (src_temp);
      free (dst_temp);
    }

  /* MoveFileEx works if SRC is a directory without any flags, but
     fails with MOVEFILE_REPLACE_EXISTING, so try without flags first.
     Thankfully, MoveFileEx handles hard links correctly, even though
     rename() does not.  */
  if (MoveFileEx (src, dst, 0))
    return 0;

  /* Retry with MOVEFILE_REPLACE_EXISTING if the move failed
     due to the destination already existing.  */
  error = GetLastError ();
  if (error == ERROR_FILE_EXISTS || error == ERROR_ALREADY_EXISTS)
    {
      if (MoveFileEx (src, dst, MOVEFILE_REPLACE_EXISTING))
        return 0;

      error = GetLastError ();
    }

  switch (error)
    {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
    case ERROR_BAD_PATHNAME:
    case ERROR_DIRECTORY:
      errno = ENOENT;
      break;

    case ERROR_ACCESS_DENIED:
    case ERROR_SHARING_VIOLATION:
      errno = EACCES;
      break;

    case ERROR_OUTOFMEMORY:
      errno = ENOMEM;
      break;

    case ERROR_CURRENT_DIRECTORY:
      errno = EBUSY;
      break;

    case ERROR_NOT_SAME_DEVICE:
      errno = EXDEV;
      break;

    case ERROR_WRITE_PROTECT:
      errno = EROFS;
      break;

    case ERROR_WRITE_FAULT:
    case ERROR_READ_FAULT:
    case ERROR_GEN_FAILURE:
      errno = EIO;
      break;

    case ERROR_HANDLE_DISK_FULL:
    case ERROR_DISK_FULL:
    case ERROR_DISK_TOO_FRAGMENTED:
      errno = ENOSPC;
      break;

    case ERROR_FILE_EXISTS:
    case ERROR_ALREADY_EXISTS:
      errno = EEXIST;
      break;

    case ERROR_BUFFER_OVERFLOW:
    case ERROR_FILENAME_EXCED_RANGE:
      errno = ENAMETOOLONG;
      break;

    case ERROR_INVALID_NAME:
    case ERROR_DELETE_PENDING:
      errno = EPERM;        /* ? */
      break;

# ifndef ERROR_FILE_TOO_LARGE
/* This value is documented but not defined in all versions of windows.h.  */
#  define ERROR_FILE_TOO_LARGE 223
# endif
    case ERROR_FILE_TOO_LARGE:
      errno = EFBIG;
      break;

    default:
      errno = EINVAL;
      break;
    }

  return -1;
}

#else /* ! W32 platform */

# include <errno.h>
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <sys/stat.h>
# include <unistd.h>

# include "dirname.h"
# include "same-inode.h"

/* Rename the file SRC to DST, fixing any trailing slash bugs.  */

int
rpl_rename (char const *src, char const *dst)
{
  size_t src_len = strlen (src);
  size_t dst_len = strlen (dst);
  char *src_temp = (char *) src;
  char *dst_temp = (char *) dst;
  bool src_slash;
  bool dst_slash;
  bool dst_exists;
  int ret_val = -1;
  int rename_errno = ENOTDIR;
  struct stat src_st;
  struct stat dst_st;

  if (!src_len || !dst_len)
    return rename (src, dst); /* Let strace see the ENOENT failure.  */

# if RENAME_DEST_EXISTS_BUG
  {
    char *src_base = last_component (src);
    char *dst_base = last_component (dst);
    if (*src_base == '.')
      {
        size_t len = base_len (src_base);
        if (len == 1 || (len == 2 && src_base[1] == '.'))
          {
            errno = EINVAL;
            return -1;
          }
      }
    if (*dst_base == '.')
      {
        size_t len = base_len (dst_base);
        if (len == 1 || (len == 2 && dst_base[1] == '.'))
          {
            errno = EINVAL;
            return -1;
          }
      }
  }
# endif /* RENAME_DEST_EXISTS_BUG */

  src_slash = src[src_len - 1] == '/';
  dst_slash = dst[dst_len - 1] == '/';

# if !RENAME_HARD_LINK_BUG && !RENAME_DEST_EXISTS_BUG
  /* If there are no trailing slashes, then trust the native
     implementation unless we also suspect issues with hard link
     detection or file/directory conflicts.  */
  if (!src_slash && !dst_slash)
    return rename (src, dst);
# endif /* !RENAME_HARD_LINK_BUG && !RENAME_DEST_EXISTS_BUG */

  /* Presence of a trailing slash requires directory semantics.  If
     the source does not exist, or if the destination cannot be turned
     into a directory, give up now.  Otherwise, strip trailing slashes
     before calling rename.  */
  if (lstat (src, &src_st))
    return -1;
  if (lstat (dst, &dst_st))
    {
      if (errno != ENOENT || (!S_ISDIR (src_st.st_mode) && dst_slash))
        return -1;
      dst_exists = false;
    }
  else
    {
      if (S_ISDIR (dst_st.st_mode) != S_ISDIR (src_st.st_mode))
        {
          errno = S_ISDIR (dst_st.st_mode) ? EISDIR : ENOTDIR;
          return -1;
        }
# if RENAME_HARD_LINK_BUG
      if (SAME_INODE (src_st, dst_st))
        return 0;
# endif /* RENAME_HARD_LINK_BUG */
      dst_exists = true;
    }

# if (RENAME_TRAILING_SLASH_SOURCE_BUG || RENAME_DEST_EXISTS_BUG        \
      || RENAME_HARD_LINK_BUG)
  /* If the only bug was that a trailing slash was allowed on a
     non-existing file destination, as in Solaris 10, then we've
     already covered that situation.  But if there is any problem with
     a trailing slash on an existing source or destination, as in
     Solaris 9, or if a directory can overwrite a symlink, as on
     Cygwin 1.5, or if directories cannot be created with trailing
     slash, as on NetBSD 1.6, then we must strip the offending slash
     and check that we have not encountered a symlink instead of a
     directory.

     Stripping a trailing slash interferes with POSIX semantics, where
     rename behavior on a symlink with a trailing slash operates on
     the corresponding target directory.  We prefer the GNU semantics
     of rejecting any use of a symlink with trailing slash, but do not
     enforce them, since Solaris 10 is able to obey POSIX semantics
     and there might be clients expecting it, as counter-intuitive as
     those semantics are.

     Technically, we could also follow the POSIX behavior by chasing a
     readlink trail, but that is harder to implement.  */
  if (src_slash)
    {
      src_temp = strdup (src);
      if (!src_temp)
        {
          /* Rather than rely on strdup-posix, we set errno ourselves.  */
          rename_errno = ENOMEM;
          goto out;
        }
      strip_trailing_slashes (src_temp);
      if (lstat (src_temp, &src_st))
        {
          rename_errno = errno;
          goto out;
        }
      if (S_ISLNK (src_st.st_mode))
        goto out;
    }
  if (dst_slash)
    {
      dst_temp = strdup (dst);
      if (!dst_temp)
        {
          rename_errno = ENOMEM;
          goto out;
        }
      strip_trailing_slashes (dst_temp);
      if (lstat (dst_temp, &dst_st))
        {
          if (errno != ENOENT)
            {
              rename_errno = errno;
              goto out;
            }
        }
      else if (S_ISLNK (dst_st.st_mode))
        goto out;
    }
# endif /* RENAME_TRAILING_SLASH_SOURCE_BUG || RENAME_DEST_EXISTS_BUG
           || RENAME_HARD_LINK_BUG */

# if RENAME_DEST_EXISTS_BUG
  /* Cygwin 1.5 sometimes behaves oddly when moving a non-empty
     directory on top of an empty one (the old directory name can
     reappear if the new directory tree is removed).  Work around this
     by removing the target first, but don't remove the target if it
     is a subdirectory of the source.  Note that we can end up losing
     a directory if rename then fails, but it was empty, so not much
     damage was done.  */
  if (dst_exists && S_ISDIR (dst_st.st_mode))
    {
      if (src_st.st_dev != dst_st.st_dev)
        {
          rename_errno = EXDEV;
          goto out;
        }
      if (src_temp != src)
        free (src_temp);
      src_temp = canonicalize_file_name (src);
      if (dst_temp != dst)
        free (dst_temp);
      dst_temp = canonicalize_file_name (dst);
      if (!src_temp || !dst_temp)
        {
          rename_errno = ENOMEM;
          goto out;
        }
      src_len = strlen (src_temp);
      if (strncmp (src_temp, dst_temp, src_len) == 0
          && dst_temp[src_len] == '/')
        {
          rename_errno = EINVAL;
          goto out;
        }
      if (rmdir (dst))
        {
          rename_errno = errno;
          goto out;
        }
    }
# endif /* RENAME_DEST_EXISTS_BUG */

  ret_val = rename (src_temp, dst_temp);
  rename_errno = errno;
 out:
  if (src_temp != src)
    free (src_temp);
  if (dst_temp != dst)
    free (dst_temp);
  errno = rename_errno;
  return ret_val;
}
#endif /* ! W32 platform */
