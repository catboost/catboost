/* Canonical list of all signal names.
   Copyright (C) 1996-1999, 2008-2016 Free Software Foundation, Inc.
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

/* This file should be usable for any platform, since it just associates
   the SIG* macros with text names and descriptions.  The actual values
   come from <bits/signum.h> (via <signal.h>).  For any signal macros do not
   exist on every platform, we can use #ifdef tests here and still use
   this single common file for all platforms.  */

/* This file is included multiple times.  */

/* Duplicate values (such as SIGBUS==SIGSEGV on Haiku) favor the last
   list entry.  */

/* Standard signals  */
#ifdef SIGHUP
  init_sig (SIGHUP, "HUP", N_("Hangup"))
#endif
#ifdef SIGINT
  init_sig (SIGINT, "INT", N_("Interrupt"))
#endif
#ifdef SIGQUIT
  init_sig (SIGQUIT, "QUIT", N_("Quit"))
#endif
#ifdef SIGILL
  init_sig (SIGILL, "ILL", N_("Illegal instruction"))
#endif
#ifdef SIGTRAP
  init_sig (SIGTRAP, "TRAP", N_("Trace/breakpoint trap"))
#endif
#ifdef SIGABRT
  init_sig (SIGABRT, "ABRT", N_("Aborted"))
#endif
#ifdef SIGFPE
  init_sig (SIGFPE, "FPE", N_("Floating point exception"))
#endif
#ifdef SIGKILL
  init_sig (SIGKILL, "KILL", N_("Killed"))
#endif
#ifdef SIGBUS
  init_sig (SIGBUS, "BUS", N_("Bus error"))
#endif
#ifdef SIGSEGV
  init_sig (SIGSEGV, "SEGV", N_("Segmentation fault"))
#endif
#ifdef SIGPIPE
  init_sig (SIGPIPE, "PIPE", N_("Broken pipe"))
#endif
#ifdef SIGALRM
  init_sig (SIGALRM, "ALRM", N_("Alarm clock"))
#endif
#ifdef SIGTERM
  init_sig (SIGTERM, "TERM", N_("Terminated"))
#endif
#ifdef SIGURG
  init_sig (SIGURG, "URG", N_("Urgent I/O condition"))
#endif
#ifdef SIGSTOP
  init_sig (SIGSTOP, "STOP", N_("Stopped (signal)"))
#endif
#ifdef SIGTSTP
  init_sig (SIGTSTP, "TSTP", N_("Stopped"))
#endif
#ifdef SIGCONT
  init_sig (SIGCONT, "CONT", N_("Continued"))
#endif
#ifdef SIGCHLD
  init_sig (SIGCHLD, "CHLD", N_("Child exited"))
#endif
#ifdef SIGTTIN
  init_sig (SIGTTIN, "TTIN", N_("Stopped (tty input)"))
#endif
#ifdef SIGTTOU
  init_sig (SIGTTOU, "TTOU", N_("Stopped (tty output)"))
#endif
#ifdef SIGIO
  init_sig (SIGIO, "IO", N_("I/O possible"))
#endif
#ifdef SIGXCPU
  init_sig (SIGXCPU, "XCPU", N_("CPU time limit exceeded"))
#endif
#ifdef SIGXFSZ
  init_sig (SIGXFSZ, "XFSZ", N_("File size limit exceeded"))
#endif
#ifdef SIGVTALRM
  init_sig (SIGVTALRM, "VTALRM", N_("Virtual timer expired"))
#endif
#ifdef SIGPROF
  init_sig (SIGPROF, "PROF", N_("Profiling timer expired"))
#endif
#ifdef SIGWINCH
  init_sig (SIGWINCH, "WINCH", N_("Window changed"))
#endif
#ifdef SIGUSR1
  init_sig (SIGUSR1, "USR1", N_("User defined signal 1"))
#endif
#ifdef SIGUSR2
  init_sig (SIGUSR2, "USR2", N_("User defined signal 2"))
#endif

/* Variations  */
#ifdef SIGEMT
  init_sig (SIGEMT, "EMT", N_("EMT trap"))
#endif
#ifdef SIGSYS
  init_sig (SIGSYS, "SYS", N_("Bad system call"))
#endif
#ifdef SIGSTKFLT
  init_sig (SIGSTKFLT, "STKFLT", N_("Stack fault"))
#endif
#ifdef SIGINFO
  init_sig (SIGINFO, "INFO", N_("Information request"))
#elif defined(SIGPWR) && (!defined(SIGLOST) || (SIGPWR != SIGLOST))
  init_sig (SIGPWR, "PWR", N_("Power failure"))
#endif
#ifdef SIGLOST
  init_sig (SIGLOST, "LOST", N_("Resource lost"))
#endif
