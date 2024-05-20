/* Temporary directories and temporary files with automatic cleanup.
   Copyright (C) 2006, 2011-2013 Free Software Foundation, Inc.
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

#ifndef _CLEAN_TEMP_H
#define _CLEAN_TEMP_H

#include <stdbool.h>
#include <stdio.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif


/* Temporary directories and temporary files should be automatically removed
   when the program exits either normally or through a fatal signal.  We can't
   rely on the "unlink before close" idiom, because it works only on Unix and
   also - if no signal blocking is used - leaves a time window where a fatal
   signal would not clean up the temporary file.

   Also, open file descriptors need to be closed before the temporary files
   and the temporary directories can be removed, because only on Unix
   (excluding Cygwin) can one remove directories containing open files.

   This module provides support for temporary directories and temporary files
   inside these temporary directories.  Temporary files without temporary
   directories are not supported here.  The temporary directories and files
   are automatically cleaned up (at the latest) when the program exits or
   dies from a fatal signal such as SIGINT, SIGTERM, SIGHUP, but not if it
   dies from a fatal signal such as SIGQUIT, SIGKILL, or SIGABRT, SIGSEGV,
   SIGBUS, SIGILL, SIGFPE.

   For the cleanup in the normal case, programs that use this module need to
   call 'cleanup_temp_dir' for each successful return of 'create_temp_dir'.
   The cleanup in the case of a fatal signal such as SIGINT, SIGTERM, SIGHUP,
   is done entirely automatically by the functions of this module.  */

struct temp_dir
{
  /* The absolute pathname of the directory.  */
  const char * const dir_name;
  /* Whether errors during explicit cleanup are reported to standard error.  */
  bool cleanup_verbose;
  /* More fields are present here, but not public.  */
};

/* Create a temporary directory.
   PREFIX is used as a prefix for the name of the temporary directory. It
   should be short and still give an indication about the program.
   PARENTDIR can be used to specify the parent directory; if NULL, a default
   parent directory is used (either $TMPDIR or /tmp or similar).
   CLEANUP_VERBOSE determines whether errors during explicit cleanup are
   reported to standard error.
   Return a fresh 'struct temp_dir' on success.  Upon error, an error message
   is shown and NULL is returned.  */
extern struct temp_dir * create_temp_dir (const char *prefix,
                                          const char *parentdir,
                                          bool cleanup_verbose);

/* Register the given ABSOLUTE_FILE_NAME as being a file inside DIR, that
   needs to be removed before DIR can be removed.
   Should be called before the file ABSOLUTE_FILE_NAME is created.  */
extern void register_temp_file (struct temp_dir *dir,
                                const char *absolute_file_name);

/* Unregister the given ABSOLUTE_FILE_NAME as being a file inside DIR, that
   needs to be removed before DIR can be removed.
   Should be called when the file ABSOLUTE_FILE_NAME could not be created.  */
extern void unregister_temp_file (struct temp_dir *dir,
                                  const char *absolute_file_name);

/* Register the given ABSOLUTE_DIR_NAME as being a subdirectory inside DIR,
   that needs to be removed before DIR can be removed.
   Should be called before the subdirectory ABSOLUTE_DIR_NAME is created.  */
extern void register_temp_subdir (struct temp_dir *dir,
                                  const char *absolute_dir_name);

/* Unregister the given ABSOLUTE_DIR_NAME as being a subdirectory inside DIR,
   that needs to be removed before DIR can be removed.
   Should be called when the subdirectory ABSOLUTE_DIR_NAME could not be
   created.  */
extern void unregister_temp_subdir (struct temp_dir *dir,
                                    const char *absolute_dir_name);

/* Remove the given ABSOLUTE_FILE_NAME and unregister it.
   Return 0 upon success, or -1 if there was some problem.  */
extern int cleanup_temp_file (struct temp_dir *dir,
                              const char *absolute_file_name);

/* Remove the given ABSOLUTE_DIR_NAME and unregister it.
   Return 0 upon success, or -1 if there was some problem.  */
extern int cleanup_temp_subdir (struct temp_dir *dir,
                                const char *absolute_dir_name);

/* Remove all registered files and subdirectories inside DIR.
   Return 0 upon success, or -1 if there was some problem.  */
extern int cleanup_temp_dir_contents (struct temp_dir *dir);

/* Remove all registered files and subdirectories inside DIR and DIR itself.
   DIR cannot be used any more after this call.
   Return 0 upon success, or -1 if there was some problem.  */
extern int cleanup_temp_dir (struct temp_dir *dir);

/* Open a temporary file in a temporary directory.
   Registers the resulting file descriptor to be closed.  */
extern int open_temp (const char *file_name, int flags, mode_t mode);
extern FILE * fopen_temp (const char *file_name, const char *mode);

/* Close a temporary file in a temporary directory.
   Unregisters the previously registered file descriptor.  */
extern int close_temp (int fd);
extern int fclose_temp (FILE *fp);

/* Like fwriteerror.
   Unregisters the previously registered file descriptor.  */
extern int fwriteerror_temp (FILE *fp);

/* Like close_stream.
   Unregisters the previously registered file descriptor.  */
extern int close_stream_temp (FILE *fp);


#ifdef __cplusplus
}
#endif

#endif /* _CLEAN_TEMP_H */
