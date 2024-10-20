/* Temporary directories and temporary files with automatic cleanup.
   Copyright (C) 2001, 2003, 2006-2007, 2009-2016 Free Software Foundation,
   Inc.
   Written by Bruno Haible <bruno@clisp.org>, 2006.

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


#include <config.h>

/* Specification.  */
#include "clean-temp.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if (defined _WIN32 || defined __WIN32__) && ! defined __CYGWIN__
# define WIN32_LEAN_AND_MEAN  /* avoid including junk */
# include <windows.h>
#endif

#include "error.h"
#include "fatal-signal.h"
#include "pathmax.h"
#include "tmpdir.h"
#include "xalloc.h"
#include "xmalloca.h"
#include "gl_xlist.h"
#include "gl_linkedhash_list.h"
#include "gettext.h"
#if GNULIB_FWRITEERROR
# include "fwriteerror.h"
#endif
#if GNULIB_CLOSE_STREAM
# include "close-stream.h"
#endif
#if GNULIB_FCNTL_SAFER
# include "fcntl--.h"
#endif
#if GNULIB_FOPEN_SAFER
# include "stdio--.h"
#endif

#define _(str) gettext (str)

/* GNU Hurd doesn't have PATH_MAX.  Use a fallback.
   Temporary directory names are usually not that long.  */
#ifndef PATH_MAX
# define PATH_MAX 1024
#endif

#ifndef uintptr_t
# define uintptr_t unsigned long
#endif


/* The use of 'volatile' in the types below (and ISO C 99 section 5.1.2.3.(5))
   ensure that while constructing or modifying the data structures, the field
   values are written to memory in the order of the C statements.  So the
   signal handler can rely on these field values to be up to date.  */


/* Registry for a single temporary directory.
   'struct temp_dir' from the public header file overlaps with this.  */
struct tempdir
{
  /* The absolute pathname of the directory.  */
  char * volatile dirname;
  /* Whether errors during explicit cleanup are reported to standard error.  */
  bool cleanup_verbose;
  /* Absolute pathnames of subdirectories.  */
  gl_list_t /* <char *> */ volatile subdirs;
  /* Absolute pathnames of files.  */
  gl_list_t /* <char *> */ volatile files;
};

/* List of all temporary directories.  */
static struct
{
  struct tempdir * volatile * volatile tempdir_list;
  size_t volatile tempdir_count;
  size_t tempdir_allocated;
} cleanup_list /* = { NULL, 0, 0 } */;

/* List of all open file descriptors to temporary files.  */
static gl_list_t /* <int> */ volatile descriptors;


/* For the subdirs and for the files, we use a gl_list_t of type LINKEDHASH.
   Why?  We need a data structure that

     1) Can contain an arbitrary number of 'char *' values.  The strings
        are compared via strcmp, not pointer comparison.
     2) Has insertion and deletion operations that are fast: ideally O(1),
        or possibly O(log n).  This is important for GNU sort, which may
        create a large number of temporary files.
     3) Allows iteration through all elements from within a signal handler.
     4) May or may not allow duplicates.  It doesn't matter here, since
        any file or subdir can only be removed once.

   Criterion 1) would allow any gl_list_t or gl_oset_t implementation.

   Criterion 2) leaves only GL_LINKEDHASH_LIST, GL_TREEHASH_LIST, or
   GL_TREE_OSET.

   Criterion 3) puts at disadvantage GL_TREEHASH_LIST and GL_TREE_OSET.
   Namely, iteration through the elements of a binary tree requires access
   to many ->left, ->right, ->parent pointers. However, the rebalancing
   code for insertion and deletion in an AVL or red-black tree is so
   complicated that we cannot assume that >left, ->right, ->parent pointers
   are in a consistent state throughout these operations.  Therefore, to
   avoid a crash in the signal handler, all destructive operations to the
   lists would have to be protected by a
       block_fatal_signals ();
       ...
       unblock_fatal_signals ();
   pair.  Which causes extra system calls.

   Criterion 3) would also discourage GL_ARRAY_LIST and GL_CARRAY_LIST,
   if they were not already excluded.  Namely, these implementations use
   xrealloc(), leaving a time window in which in the list->elements pointer
   points to already deallocated memory.  To avoid a crash in the signal
   handler at such a moment, all destructive operations would have to
   protected by block/unblock_fatal_signals (), in this case too.

   A list of type GL_LINKEDHASH_LIST without duplicates fulfills all
   requirements:
     2) Insertion and deletion are O(1) on average.
     3) The gl_list_iterator, gl_list_iterator_next implementations do
        not trigger memory allocations, nor other system calls, and are
        therefore safe to be called from a signal handler.
        Furthermore, since SIGNAL_SAFE_LIST is defined, the implementation
        of the destructive functions ensures that the list structure is
        safe to be traversed at any moment, even when interrupted by an
        asynchronous signal.
 */

/* String equality and hash code functions used by the lists.  */

static bool
string_equals (const void *x1, const void *x2)
{
  const char *s1 = (const char *) x1;
  const char *s2 = (const char *) x2;
  return strcmp (s1, s2) == 0;
}

#define SIZE_BITS (sizeof (size_t) * CHAR_BIT)

/* A hash function for NUL-terminated char* strings using
   the method described by Bruno Haible.
   See http://www.haible.de/bruno/hashfunc.html.  */
static size_t
string_hash (const void *x)
{
  const char *s = (const char *) x;
  size_t h = 0;

  for (; *s; s++)
    h = *s + ((h << 9) | (h >> (SIZE_BITS - 9)));

  return h;
}


/* The signal handler.  It gets called asynchronously.  */
static void
cleanup ()
{
  size_t i;

  /* First close all file descriptors to temporary files.  */
  {
    gl_list_t fds = descriptors;

    if (fds != NULL)
      {
        gl_list_iterator_t iter;
        const void *element;

        iter = gl_list_iterator (fds);
        while (gl_list_iterator_next (&iter, &element, NULL))
          {
            int fd = (int) (uintptr_t) element;
            close (fd);
          }
        gl_list_iterator_free (&iter);
      }
  }

  for (i = 0; i < cleanup_list.tempdir_count; i++)
    {
      struct tempdir *dir = cleanup_list.tempdir_list[i];

      if (dir != NULL)
        {
          gl_list_iterator_t iter;
          const void *element;

          /* First cleanup the files in the subdirectories.  */
          iter = gl_list_iterator (dir->files);
          while (gl_list_iterator_next (&iter, &element, NULL))
            {
              const char *file = (const char *) element;
              unlink (file);
            }
          gl_list_iterator_free (&iter);

          /* Then cleanup the subdirectories.  */
          iter = gl_list_iterator (dir->subdirs);
          while (gl_list_iterator_next (&iter, &element, NULL))
            {
              const char *subdir = (const char *) element;
              rmdir (subdir);
            }
          gl_list_iterator_free (&iter);

          /* Then cleanup the temporary directory itself.  */
          rmdir (dir->dirname);
        }
    }
}

/* Create a temporary directory.
   PREFIX is used as a prefix for the name of the temporary directory. It
   should be short and still give an indication about the program.
   PARENTDIR can be used to specify the parent directory; if NULL, a default
   parent directory is used (either $TMPDIR or /tmp or similar).
   CLEANUP_VERBOSE determines whether errors during explicit cleanup are
   reported to standard error.
   Return a fresh 'struct temp_dir' on success.  Upon error, an error message
   is shown and NULL is returned.  */
struct temp_dir *
create_temp_dir (const char *prefix, const char *parentdir,
                 bool cleanup_verbose)
{
  struct tempdir * volatile *tmpdirp = NULL;
  struct tempdir *tmpdir;
  size_t i;
  char *xtemplate;
  char *tmpdirname;

  /* See whether it can take the slot of an earlier temporary directory
     already cleaned up.  */
  for (i = 0; i < cleanup_list.tempdir_count; i++)
    if (cleanup_list.tempdir_list[i] == NULL)
      {
        tmpdirp = &cleanup_list.tempdir_list[i];
        break;
      }
  if (tmpdirp == NULL)
    {
      /* See whether the array needs to be extended.  */
      if (cleanup_list.tempdir_count == cleanup_list.tempdir_allocated)
        {
          /* Note that we cannot use xrealloc(), because then the cleanup()
             function could access an already deallocated array.  */
          struct tempdir * volatile *old_array = cleanup_list.tempdir_list;
          size_t old_allocated = cleanup_list.tempdir_allocated;
          size_t new_allocated = 2 * cleanup_list.tempdir_allocated + 1;
          struct tempdir * volatile *new_array =
            XNMALLOC (new_allocated, struct tempdir * volatile);

          if (old_allocated == 0)
            /* First use of this facility.  Register the cleanup handler.  */
            at_fatal_signal (&cleanup);
          else
            {
              /* Don't use memcpy() here, because memcpy takes non-volatile
                 arguments and is therefore not guaranteed to complete all
                 memory stores before the next statement.  */
              size_t k;

              for (k = 0; k < old_allocated; k++)
                new_array[k] = old_array[k];
            }

          cleanup_list.tempdir_list = new_array;
          cleanup_list.tempdir_allocated = new_allocated;

          /* Now we can free the old array.  */
          if (old_array != NULL)
            free ((struct tempdir **) old_array);
        }

      tmpdirp = &cleanup_list.tempdir_list[cleanup_list.tempdir_count];
      /* Initialize *tmpdirp before incrementing tempdir_count, so that
         cleanup() will skip this entry before it is fully initialized.  */
      *tmpdirp = NULL;
      cleanup_list.tempdir_count++;
    }

  /* Initialize a 'struct tempdir'.  */
  tmpdir = XMALLOC (struct tempdir);
  tmpdir->dirname = NULL;
  tmpdir->cleanup_verbose = cleanup_verbose;
  tmpdir->subdirs = gl_list_create_empty (GL_LINKEDHASH_LIST,
                                          string_equals, string_hash, NULL,
                                          false);
  tmpdir->files = gl_list_create_empty (GL_LINKEDHASH_LIST,
                                        string_equals, string_hash, NULL,
                                        false);

  /* Create the temporary directory.  */
  xtemplate = (char *) xmalloca (PATH_MAX);
  if (path_search (xtemplate, PATH_MAX, parentdir, prefix, parentdir == NULL))
    {
      error (0, errno,
             _("cannot find a temporary directory, try setting $TMPDIR"));
      goto quit;
    }
  block_fatal_signals ();
  tmpdirname = mkdtemp (xtemplate);
  if (tmpdirname != NULL)
    {
      tmpdir->dirname = tmpdirname;
      *tmpdirp = tmpdir;
    }
  unblock_fatal_signals ();
  if (tmpdirname == NULL)
    {
      error (0, errno,
             _("cannot create a temporary directory using template \"%s\""),
             xtemplate);
      goto quit;
    }
  /* Replace tmpdir->dirname with a copy that has indefinite extent.
     We cannot do this inside the block_fatal_signals/unblock_fatal_signals
     block because then the cleanup handler would not remove the directory
     if xstrdup fails.  */
  tmpdir->dirname = xstrdup (tmpdirname);
  freea (xtemplate);
  return (struct temp_dir *) tmpdir;

 quit:
  freea (xtemplate);
  return NULL;
}

/* Register the given ABSOLUTE_FILE_NAME as being a file inside DIR, that
   needs to be removed before DIR can be removed.
   Should be called before the file ABSOLUTE_FILE_NAME is created.  */
void
register_temp_file (struct temp_dir *dir,
                    const char *absolute_file_name)
{
  struct tempdir *tmpdir = (struct tempdir *)dir;

  /* Add absolute_file_name to tmpdir->files, without duplicates.  */
  if (gl_list_search (tmpdir->files, absolute_file_name) == NULL)
    gl_list_add_first (tmpdir->files, xstrdup (absolute_file_name));
}

/* Unregister the given ABSOLUTE_FILE_NAME as being a file inside DIR, that
   needs to be removed before DIR can be removed.
   Should be called when the file ABSOLUTE_FILE_NAME could not be created.  */
void
unregister_temp_file (struct temp_dir *dir,
                      const char *absolute_file_name)
{
  struct tempdir *tmpdir = (struct tempdir *)dir;
  gl_list_t list = tmpdir->files;
  gl_list_node_t node;

  node = gl_list_search (list, absolute_file_name);
  if (node != NULL)
    {
      char *old_string = (char *) gl_list_node_value (list, node);

      gl_list_remove_node (list, node);
      free (old_string);
    }
}

/* Register the given ABSOLUTE_DIR_NAME as being a subdirectory inside DIR,
   that needs to be removed before DIR can be removed.
   Should be called before the subdirectory ABSOLUTE_DIR_NAME is created.  */
void
register_temp_subdir (struct temp_dir *dir,
                      const char *absolute_dir_name)
{
  struct tempdir *tmpdir = (struct tempdir *)dir;

  /* Add absolute_dir_name to tmpdir->subdirs, without duplicates.  */
  if (gl_list_search (tmpdir->subdirs, absolute_dir_name) == NULL)
    gl_list_add_first (tmpdir->subdirs, xstrdup (absolute_dir_name));
}

/* Unregister the given ABSOLUTE_DIR_NAME as being a subdirectory inside DIR,
   that needs to be removed before DIR can be removed.
   Should be called when the subdirectory ABSOLUTE_DIR_NAME could not be
   created.  */
void
unregister_temp_subdir (struct temp_dir *dir,
                        const char *absolute_dir_name)
{
  struct tempdir *tmpdir = (struct tempdir *)dir;
  gl_list_t list = tmpdir->subdirs;
  gl_list_node_t node;

  node = gl_list_search (list, absolute_dir_name);
  if (node != NULL)
    {
      char *old_string = (char *) gl_list_node_value (list, node);

      gl_list_remove_node (list, node);
      free (old_string);
    }
}

/* Remove a file, with optional error message.
   Return 0 upon success, or -1 if there was some problem.  */
static int
do_unlink (struct temp_dir *dir, const char *absolute_file_name)
{
  if (unlink (absolute_file_name) < 0 && dir->cleanup_verbose
      && errno != ENOENT)
    {
      error (0, errno, _("cannot remove temporary file %s"), absolute_file_name);
      return -1;
    }
  return 0;
}

/* Remove a directory, with optional error message.
   Return 0 upon success, or -1 if there was some problem.  */
static int
do_rmdir (struct temp_dir *dir, const char *absolute_dir_name)
{
  if (rmdir (absolute_dir_name) < 0 && dir->cleanup_verbose
      && errno != ENOENT)
    {
      error (0, errno,
             _("cannot remove temporary directory %s"), absolute_dir_name);
      return -1;
    }
  return 0;
}

/* Remove the given ABSOLUTE_FILE_NAME and unregister it.
   Return 0 upon success, or -1 if there was some problem.  */
int
cleanup_temp_file (struct temp_dir *dir,
                   const char *absolute_file_name)
{
  int err;

  err = do_unlink (dir, absolute_file_name);
  unregister_temp_file (dir, absolute_file_name);

  return err;
}

/* Remove the given ABSOLUTE_DIR_NAME and unregister it.
   Return 0 upon success, or -1 if there was some problem.  */
int
cleanup_temp_subdir (struct temp_dir *dir,
                     const char *absolute_dir_name)
{
  int err;

  err = do_rmdir (dir, absolute_dir_name);
  unregister_temp_subdir (dir, absolute_dir_name);

  return err;
}

/* Remove all registered files and subdirectories inside DIR.
   Return 0 upon success, or -1 if there was some problem.  */
int
cleanup_temp_dir_contents (struct temp_dir *dir)
{
  struct tempdir *tmpdir = (struct tempdir *)dir;
  int err = 0;
  gl_list_t list;
  gl_list_iterator_t iter;
  const void *element;
  gl_list_node_t node;

  /* First cleanup the files in the subdirectories.  */
  list = tmpdir->files;
  iter = gl_list_iterator (list);
  while (gl_list_iterator_next (&iter, &element, &node))
    {
      char *file = (char *) element;

      err |= do_unlink (dir, file);
      gl_list_remove_node (list, node);
      /* Now only we can free file.  */
      free (file);
    }
  gl_list_iterator_free (&iter);

  /* Then cleanup the subdirectories.  */
  list = tmpdir->subdirs;
  iter = gl_list_iterator (list);
  while (gl_list_iterator_next (&iter, &element, &node))
    {
      char *subdir = (char *) element;

      err |= do_rmdir (dir, subdir);
      gl_list_remove_node (list, node);
      /* Now only we can free subdir.  */
      free (subdir);
    }
  gl_list_iterator_free (&iter);

  return err;
}

/* Remove all registered files and subdirectories inside DIR and DIR itself.
   DIR cannot be used any more after this call.
   Return 0 upon success, or -1 if there was some problem.  */
int
cleanup_temp_dir (struct temp_dir *dir)
{
  struct tempdir *tmpdir = (struct tempdir *)dir;
  int err = 0;
  size_t i;

  err |= cleanup_temp_dir_contents (dir);
  err |= do_rmdir (dir, tmpdir->dirname);

  for (i = 0; i < cleanup_list.tempdir_count; i++)
    if (cleanup_list.tempdir_list[i] == tmpdir)
      {
        /* Remove cleanup_list.tempdir_list[i].  */
        if (i + 1 == cleanup_list.tempdir_count)
          {
            while (i > 0 && cleanup_list.tempdir_list[i - 1] == NULL)
              i--;
            cleanup_list.tempdir_count = i;
          }
        else
          cleanup_list.tempdir_list[i] = NULL;
        /* Now only we can free the tmpdir->dirname, tmpdir->subdirs,
           tmpdir->files, and tmpdir itself.  */
        gl_list_free (tmpdir->files);
        gl_list_free (tmpdir->subdirs);
        free (tmpdir->dirname);
        free (tmpdir);
        return err;
      }

  /* The user passed an invalid DIR argument.  */
  abort ();
}


#if (defined _WIN32 || defined __WIN32__) && ! defined __CYGWIN__

/* On Windows, opening a file with _O_TEMPORARY has the effect of passing
   the FILE_FLAG_DELETE_ON_CLOSE flag to CreateFile(), which has the effect
   of deleting the file when it is closed - even when the program crashes.
   But (according to the Cygwin sources) it works only on Windows NT or newer.
   So we cache the info whether we are running on Windows NT or newer.  */

static bool
supports_delete_on_close ()
{
  static int known; /* 1 = yes, -1 = no, 0 = unknown */
  /* M4 wants to close and later reopen a temporary file, so
     delete-on-close must not be used.  */
  known = -1;
  if (!known)
    {
      OSVERSIONINFO v;

      /* According to
         <http://msdn.microsoft.com/en-us/library/windows/desktop/ms724451(v=vs.85).aspx>
         this structure must be initialized as follows:  */
      v.dwOSVersionInfoSize = sizeof (OSVERSIONINFO);

      if (GetVersionEx (&v))
        known = (v.dwPlatformId == VER_PLATFORM_WIN32_NT ? 1 : -1);
      else
        known = -1;
    }
  return (known > 0);
}

#endif


/* Register a file descriptor to be closed.  */
static void
register_fd (int fd)
{
  if (descriptors == NULL)
    descriptors = gl_list_create_empty (GL_LINKEDHASH_LIST, NULL, NULL, NULL,
                                        false);
  gl_list_add_first (descriptors, (void *) (uintptr_t) fd);
}

/* Unregister a file descriptor to be closed.  */
static void
unregister_fd (int fd)
{
  gl_list_t fds = descriptors;
  gl_list_node_t node;

  if (fds == NULL)
    /* descriptors should already contain fd.  */
    abort ();
  node = gl_list_search (fds, (void *) (uintptr_t) fd);
  if (node == NULL)
    /* descriptors should already contain fd.  */
    abort ();
  gl_list_remove_node (fds, node);
}

/* Open a temporary file in a temporary directory.
   Registers the resulting file descriptor to be closed.  */
int
open_temp (const char *file_name, int flags, mode_t mode)
{
  int fd;
  int saved_errno;

  block_fatal_signals ();
  /* Note: 'open' here is actually open() or open_safer().  */
#if (defined _WIN32 || defined __WIN32__) && ! defined __CYGWIN__
  /* Use _O_TEMPORARY when possible, to increase the chances that the
     temporary file is removed when the process crashes.  */
  if (supports_delete_on_close ())
    fd = open (file_name, flags | _O_TEMPORARY, mode);
  else
#endif
    fd = open (file_name, flags, mode);
  saved_errno = errno;
  if (fd >= 0)
    register_fd (fd);
  unblock_fatal_signals ();
  errno = saved_errno;
  return fd;
}

/* Open a temporary file in a temporary directory.
   Registers the resulting file descriptor to be closed.  */
FILE *
fopen_temp (const char *file_name, const char *mode)
{
  FILE *fp;
  int saved_errno;

  block_fatal_signals ();
  /* Note: 'fopen' here is actually fopen() or fopen_safer().  */
#if (defined _WIN32 || defined __WIN32__) && ! defined __CYGWIN__
  /* Use _O_TEMPORARY when possible, to increase the chances that the
     temporary file is removed when the process crashes.  */
  if (supports_delete_on_close ())
    {
      size_t mode_len = strlen (mode);
      char *augmented_mode = (char *) xmalloca (mode_len + 2);
      memcpy (augmented_mode, mode, mode_len);
      memcpy (augmented_mode + mode_len, "D", 2);

      fp = fopen (file_name, augmented_mode);
      saved_errno = errno;

      freea (augmented_mode);
    }
  else
#endif
    {
      fp = fopen (file_name, mode);
      saved_errno = errno;
    }
  if (fp != NULL)
    {
      /* It is sufficient to register fileno (fp) instead of the entire fp,
         because at cleanup time there is no need to do an fflush (fp); a
         close (fileno (fp)) will be enough.  */
      int fd = fileno (fp);
      if (!(fd >= 0))
        abort ();
      register_fd (fd);
    }
  unblock_fatal_signals ();
  errno = saved_errno;
  return fp;
}

/* Close a temporary file in a temporary directory.
   Unregisters the previously registered file descriptor.  */
int
close_temp (int fd)
{
  if (fd >= 0)
    {
      /* No blocking of signals is needed here, since a double close of a
         file descriptor is harmless.  */
      int result = close (fd);
      int saved_errno = errno;

      /* No race condition here: we assume a single-threaded program, hence
         fd cannot be re-opened here.  */

      unregister_fd (fd);

      errno = saved_errno;
      return result;
    }
  else
    return close (fd);
}

/* Close a temporary file in a temporary directory.
   Unregisters the previously registered file descriptor.  */
int
fclose_temp (FILE *fp)
{
  int fd = fileno (fp);
  /* No blocking of signals is needed here, since a double close of a
     file descriptor is harmless.  */
  int result = fclose (fp);
  int saved_errno = errno;

  /* No race condition here: we assume a single-threaded program, hence
     fd cannot be re-opened here.  */

  unregister_fd (fd);

  errno = saved_errno;
  return result;
}

#if GNULIB_FWRITEERROR
/* Like fwriteerror.
   Unregisters the previously registered file descriptor.  */
int
fwriteerror_temp (FILE *fp)
{
  int fd = fileno (fp);
  /* No blocking of signals is needed here, since a double close of a
     file descriptor is harmless.  */
  int result = fwriteerror (fp);
  int saved_errno = errno;

  /* No race condition here: we assume a single-threaded program, hence
     fd cannot be re-opened here.  */

  unregister_fd (fd);

  errno = saved_errno;
  return result;
}
#endif

#if GNULIB_CLOSE_STREAM
/* Like close_stream.
   Unregisters the previously registered file descriptor.  */
int
close_stream_temp (FILE *fp)
{
  int fd = fileno (fp);
  /* No blocking of signals is needed here, since a double close of a
     file descriptor is harmless.  */
  int result = close_stream (fp);
  int saved_errno = errno;

  /* No race condition here: we assume a single-threaded program, hence
     fd cannot be re-opened here.  */

  unregister_fd (fd);

  errno = saved_errno;
  return result;
}
#endif
