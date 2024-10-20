/* Program name management.
   Copyright (C) 2016 Free Software Foundation, Inc.

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
#include "getprogname.h"

#include <errno.h> /* get program_invocation_name declaration */
#include <stdlib.h> /* get __argv declaration */

#ifdef _AIX
# include <unistd.h>
# include <procinfo.h>
# include <string.h>
#endif

#ifdef __MVS__
# ifndef _OPEN_SYS
#  define _OPEN_SYS
# endif
# include <string.h>
# error #include <sys/ps.h>
#endif

#ifdef __hpux
# include <unistd.h>
# include <sys/param.h>
# include <sys/pstat.h>
# include <string.h>
#endif

#include "dirname.h"

#ifndef HAVE_GETPROGNAME             /* not Mac OS X, FreeBSD, NetBSD, OpenBSD >= 5.4, Cygwin */
char const *
getprogname (void)
{
# if HAVE_DECL_PROGRAM_INVOCATION_SHORT_NAME                /* glibc, BeOS */
  /* https://www.gnu.org/software/libc/manual/html_node/Error-Messages.html */
  return program_invocation_short_name;
# elif HAVE_DECL_PROGRAM_INVOCATION_NAME                    /* glibc, BeOS */
  /* https://www.gnu.org/software/libc/manual/html_node/Error-Messages.html */
  return last_component (program_invocation_name);
# elif HAVE_GETEXECNAME                                     /* Solaris */
  /* http://docs.oracle.com/cd/E19253-01/816-5168/6mbb3hrb1/index.html */
  const char *p = getexecname ();
  if (!p)
    p = "?";
  return last_component (p);
# elif HAVE_DECL___ARGV                                     /* mingw, MSVC */
  /* https://msdn.microsoft.com/en-us/library/dn727674.aspx */
  const char *p = __argv && __argv[0] ? __argv[0] : "?";
  return last_component (p);
# elif HAVE_VAR___PROGNAME                                  /* OpenBSD, QNX */
  /* http://man.openbsd.org/style.9 */
  /* http://www.qnx.de/developers/docs/6.5.0/index.jsp?topic=%2Fcom.qnx.doc.neutrino_lib_ref%2Fp%2F__progname.html */
  /* Be careful to declare this only when we absolutely need it
     (OpenBSD 5.1), rather than when it's available.  Otherwise,
     its mere declaration makes program_invocation_short_name
     malfunction (have zero length) with Fedora 25's glibc.  */
  extern char *__progname;
  const char *p = __progname;
  return p && p[0] ? p : "?";
# elif _AIX                                                 /* AIX */
  /* Idea by Bastien ROUCARIÈS,
     http://lists.gnu.org/archive/html/bug-gnulib/2010-12/msg00095.html
     Reference: http://
   ibm.biz/knowctr#ssw_aix_53/com.ibm.aix.basetechref/doc/basetrf1/getprocs.htm
  */
  static char *p;
  static int first = 1;
  if (first)
    {
      first = 0;
      pid_t pid = getpid ();
      struct procentry64 procs;
      p = (0 < getprocs64 (&procs, sizeof procs, NULL, 0, &pid, 1)
           ? strdup (procs.pi_comm)
           : NULL);
      if (!p)
        p = "?";
    }
  return p;
# elif defined __hpux
  static char *p;
  static int first = 1;
  if (first)
    {
      first = 0;
      pid_t pid = getpid ();
      struct pst_status status;
      p = (0 < pstat_getproc (&status, sizeof status, 0, pid)
           ? strdup (status.pst_ucomm)
           : NULL);
      if (!p)
        p = "?";
    }
  return p;
# elif __MVS__                                              /* z/OS */
  /* https://www.ibm.com/support/knowledgecenter/SSLTBW_2.1.0/com.ibm.zos.v2r1.bpxbd00/rtwgetp.htm */
  static char *p = "?";
  static int first = 1;
  if (first)
    {
      pid_t pid = getpid ();
      int token;
      W_PSPROC buf;
      first = 0;
      memset (&buf, 0, sizeof(buf));
      buf.ps_cmdptr    = (char *) malloc (buf.ps_cmdlen    = PS_CMDBLEN_LONG);
      buf.ps_conttyptr = (char *) malloc (buf.ps_conttylen = PS_CONTTYBLEN);
      buf.ps_pathptr   = (char *) malloc (buf.ps_pathlen   = PS_PATHBLEN);
      if (buf.ps_cmdptr && buf.ps_conttyptr && buf.ps_pathptr)
        {
          for (token = 0; token >= 0;
               token = w_getpsent (token, &buf, sizeof(buf)))
            {
              if (token > 0 && buf.ps_pid == pid)
                {
                  char *s = strdup (last_component (buf.ps_pathptr));
                  if (s)
                    p = s;
                  break;
                }
            }
        }
      free (buf.ps_cmdptr);
      free (buf.ps_conttyptr);
      free (buf.ps_pathptr);
    }
  return p;
# else
#  error "getprogname module not ported to this OS"
# endif
}

#endif
