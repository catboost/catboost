/* Creation of autonomous subprocesses.
   Copyright (C) 2001-2004, 2006-2013 Free Software Foundation, Inc.
   Written by Bruno Haible <haible@clisp.cons.org>, 2001.

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
#include "execute.h"

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

#include "penviron.h"
#include "error.h"
#include "fatal-signal.h"
#include "wait-process.h"
#include "gettext.h"

#define _(str) gettext (str)

#if (defined _WIN32 || defined __WIN32__) && ! defined __CYGWIN__

/* Native Windows API.  */
# include <process.h>
# include "w32spawn.h"

#else

/* Unix API.  */
# include <spawn.h>

#endif

/* The results of open() in this file are not used with fchdir,
   therefore save some unnecessary work in fchdir.c.  */
#undef open
#undef close


#if defined EINTR && ((defined _WIN32 || defined __WIN32__) && ! defined __CYGWIN__)

/* EINTR handling for close(), open().
   These functions can return -1/EINTR even though we don't have any
   signal handlers set up, namely when we get interrupted via SIGSTOP.  */

static int
nonintr_close (int fd)
{
  int retval;

  do
    retval = close (fd);
  while (retval < 0 && errno == EINTR);

  return retval;
}
#define close nonintr_close

static int
nonintr_open (const char *pathname, int oflag, mode_t mode)
{
  int retval;

  do
    retval = open (pathname, oflag, mode);
  while (retval < 0 && errno == EINTR);

  return retval;
}
#undef open /* avoid warning on VMS */
#define open nonintr_open

#endif


/* Execute a command, optionally redirecting any of the three standard file
   descriptors to /dev/null.  Return its exit code.
   If it didn't terminate correctly, exit if exit_on_error is true, otherwise
   return 127.
   If slave_process is true, the child process will be terminated when its
   creator receives a catchable fatal signal.  */
int
execute (const char *progname,
         const char *prog_path, char **prog_argv,
         bool ignore_sigpipe,
         bool null_stdin, bool null_stdout, bool null_stderr,
         bool slave_process, bool exit_on_error,
         int *termsigp)
{
#if (defined _WIN32 || defined __WIN32__) && ! defined __CYGWIN__

  /* Native Windows API.  */
  int orig_stdin;
  int orig_stdout;
  int orig_stderr;
  int exitcode;
  int nullinfd;
  int nulloutfd;

  /* FIXME: Need to free memory allocated by prepare_spawn.  */
  prog_argv = prepare_spawn (prog_argv);

  /* Save standard file handles of parent process.  */
  if (null_stdin)
    orig_stdin = dup_safer_noinherit (STDIN_FILENO);
  if (null_stdout)
    orig_stdout = dup_safer_noinherit (STDOUT_FILENO);
  if (null_stderr)
    orig_stderr = dup_safer_noinherit (STDERR_FILENO);
  exitcode = -1;

  /* Create standard file handles of child process.  */
  nullinfd = -1;
  nulloutfd = -1;
  if ((!null_stdin
       || ((nullinfd = open ("NUL", O_RDONLY, 0)) >= 0
           && (nullinfd == STDIN_FILENO
               || (dup2 (nullinfd, STDIN_FILENO) >= 0
                   && close (nullinfd) >= 0))))
      && (!(null_stdout || null_stderr)
          || ((nulloutfd = open ("NUL", O_RDWR, 0)) >= 0
              && (!null_stdout
                  || nulloutfd == STDOUT_FILENO
                  || dup2 (nulloutfd, STDOUT_FILENO) >= 0)
              && (!null_stderr
                  || nulloutfd == STDERR_FILENO
                  || dup2 (nulloutfd, STDERR_FILENO) >= 0)
              && ((null_stdout && nulloutfd == STDOUT_FILENO)
                  || (null_stderr && nulloutfd == STDERR_FILENO)
                  || close (nulloutfd) >= 0))))
    /* Use spawnvpe and pass the environment explicitly.  This is needed if
       the program has modified the environment using putenv() or [un]setenv().
       On Windows, programs have two environments, one in the "environment
       block" of the process and managed through SetEnvironmentVariable(), and
       one inside the process, in the location retrieved by the 'environ'
       macro.  When using spawnvp() without 'e', the child process inherits a
       copy of the environment block - ignoring the effects of putenv() and
       [un]setenv().  */
    {
      exitcode = spawnvpe (P_WAIT, prog_path, (const char **) prog_argv,
                           (const char **) environ);
      if (exitcode < 0 && errno == ENOEXEC)
        {
          /* prog is not a native executable.  Try to execute it as a
             shell script.  Note that prepare_spawn() has already prepended
             a hidden element "sh.exe" to prog_argv.  */
          --prog_argv;
          exitcode = spawnvpe (P_WAIT, prog_argv[0], (const char **) prog_argv,
                               (const char **) environ);
        }
    }
  if (nulloutfd >= 0)
    close (nulloutfd);
  if (nullinfd >= 0)
    close (nullinfd);

  /* Restore standard file handles of parent process.  */
  if (null_stderr)
    undup_safer_noinherit (orig_stderr, STDERR_FILENO);
  if (null_stdout)
    undup_safer_noinherit (orig_stdout, STDOUT_FILENO);
  if (null_stdin)
    undup_safer_noinherit (orig_stdin, STDIN_FILENO);

  if (termsigp != NULL)
    *termsigp = 0;

  if (exitcode == -1)
    {
      if (exit_on_error || !null_stderr)
        error (exit_on_error ? EXIT_FAILURE : 0, errno,
               _("%s subprocess failed"), progname);
      return 127;
    }

  return exitcode;

#else

  /* Unix API.  */
  /* Note about 127: Some errors during posix_spawnp() cause the function
     posix_spawnp() to return an error code; some other errors cause the
     subprocess to exit with return code 127.  It is implementation
     dependent which error is reported which way.  We treat both cases as
     equivalent.  */
  sigset_t blocked_signals;
  posix_spawn_file_actions_t actions;
  bool actions_allocated;
  posix_spawnattr_t attrs;
  bool attrs_allocated;
  int err;
  pid_t child;

  if (slave_process)
    {
      sigprocmask (SIG_SETMASK, NULL, &blocked_signals);
      block_fatal_signals ();
    }
  actions_allocated = false;
  attrs_allocated = false;
  if ((err = posix_spawn_file_actions_init (&actions)) != 0
      || (actions_allocated = true,
          (null_stdin
            && (err = posix_spawn_file_actions_addopen (&actions,
                                                        STDIN_FILENO,
                                                        "/dev/null", O_RDONLY,
                                                        0))
               != 0)
          || (null_stdout
              && (err = posix_spawn_file_actions_addopen (&actions,
                                                          STDOUT_FILENO,
                                                          "/dev/null", O_RDWR,
                                                          0))
                 != 0)
          || (null_stderr
              && (err = posix_spawn_file_actions_addopen (&actions,
                                                          STDERR_FILENO,
                                                          "/dev/null", O_RDWR,
                                                          0))
                 != 0)
          || (slave_process
              && ((err = posix_spawnattr_init (&attrs)) != 0
                  || (attrs_allocated = true,
                      (err = posix_spawnattr_setsigmask (&attrs,
                                                         &blocked_signals))
                      != 0
                      || (err = posix_spawnattr_setflags (&attrs,
                                                        POSIX_SPAWN_SETSIGMASK))
                         != 0)))
          || (err = posix_spawnp (&child, prog_path, &actions,
                                  attrs_allocated ? &attrs : NULL, prog_argv,
                                  environ))
             != 0))
    {
      if (actions_allocated)
        posix_spawn_file_actions_destroy (&actions);
      if (attrs_allocated)
        posix_spawnattr_destroy (&attrs);
      if (slave_process)
        unblock_fatal_signals ();
      if (termsigp != NULL)
        *termsigp = 0;
      if (exit_on_error || !null_stderr)
        error (exit_on_error ? EXIT_FAILURE : 0, err,
               _("%s subprocess failed"), progname);
      return 127;
    }
  posix_spawn_file_actions_destroy (&actions);
  if (attrs_allocated)
    posix_spawnattr_destroy (&attrs);
  if (slave_process)
    {
      register_slave_subprocess (child);
      unblock_fatal_signals ();
    }

  return wait_subprocess (child, progname, ignore_sigpipe, null_stderr,
                          slave_process, exit_on_error, termsigp);

#endif
}
