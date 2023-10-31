/* Stack overflow handling.

   Copyright (C) 2002, 2004, 2006, 2008-2013 Free Software Foundation, Inc.

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

/* Written by Paul Eggert.  */

/* NOTES:

   A program that uses alloca, dynamic arrays, or large local
   variables may extend the stack by more than a page at a time.  If
   so, when the stack overflows the operating system may not detect
   the overflow until the program uses the array, and this module may
   incorrectly report a program error instead of a stack overflow.

   To avoid this problem, allocate only small objects on the stack; a
   program should be OK if it limits single allocations to a page or
   less.  Allocate larger arrays in static storage, or on the heap
   (e.g., with malloc).  Yes, this is a pain, but we don't know of any
   better solution that is portable.

   No attempt has been made to deal with multithreaded applications.  */

#include <config.h>

#ifndef __attribute__
# if __GNUC__ < 3
#  define __attribute__(x)
# endif
#endif

#include "gettext.h"
#define _(msgid) gettext (msgid)

#include <errno.h>

#include <signal.h>
#if ! HAVE_STACK_T && ! defined stack_t
typedef struct sigaltstack stack_t;
#endif
#ifndef SIGSTKSZ
# define SIGSTKSZ 16384
#elif defined __USE_DYNAMIC_STACK_SIZE
/* Redefining SIGSTKSZ here as dynamic stack size is not supported in this version of bison */
# undef SIGSTKSZ
# define SIGSTKSZ 16384
#elif HAVE_LIBSIGSEGV && SIGSTKSZ < 16384
/* libsigsegv 2.6 through 2.8 have a bug where some architectures use
   more than the Linux default of an 8k alternate stack when deciding
   if a fault was caused by stack overflow.  */
# undef SIGSTKSZ
# define SIGSTKSZ 16384
#endif

#include <stdlib.h>
#include <string.h>

/* Posix 2001 declares ucontext_t in <ucontext.h>, Posix 200x in
   <signal.h>.  */
#if HAVE_UCONTEXT_H
# include <ucontext.h>
#endif

#include <unistd.h>

#if HAVE_LIBSIGSEGV
# include <sigsegv.h>
#endif

#include "c-stack.h"
#include "exitfail.h"
#include "ignore-value.h"

#if defined SA_ONSTACK && defined SA_SIGINFO
# define SIGINFO_WORKS 1
#else
# define SIGINFO_WORKS 0
# ifndef SA_ONSTACK
#  define SA_ONSTACK 0
# endif
#endif

extern char *program_name;

/* The user-specified action to take when a SEGV-related program error
   or stack overflow occurs.  */
static void (* volatile segv_action) (int);

/* Translated messages for program errors and stack overflow.  Do not
   translate them in the signal handler, since gettext is not
   async-signal-safe.  */
static char const * volatile program_error_message;
static char const * volatile stack_overflow_message;

/* Output an error message, then exit with status EXIT_FAILURE if it
   appears to have been a stack overflow, or with a core dump
   otherwise.  This function is async-signal-safe.  */

static _Noreturn void
die (int signo)
{
  char const *message;
#if !SIGINFO_WORKS && !HAVE_LIBSIGSEGV
  /* We can't easily determine whether it is a stack overflow; so
     assume that the rest of our program is perfect (!) and that
     this segmentation violation is a stack overflow.  */
  signo = 0;
#endif /* !SIGINFO_WORKS && !HAVE_LIBSIGSEGV */
  segv_action (signo);
  message = signo ? program_error_message : stack_overflow_message;
  ignore_value (write (STDERR_FILENO, program_name, strlen (program_name)));
  ignore_value (write (STDERR_FILENO, ": ", 2));
  ignore_value (write (STDERR_FILENO, message, strlen (message)));
  ignore_value (write (STDERR_FILENO, "\n", 1));
  if (! signo)
    _exit (exit_failure);
  raise (signo);
  abort ();
}

#if (HAVE_SIGALTSTACK && HAVE_DECL_SIGALTSTACK \
     && HAVE_STACK_OVERFLOW_HANDLING) || HAVE_LIBSIGSEGV

/* Storage for the alternate signal stack.  */
static union
{
  char buffer[SIGSTKSZ];

  /* These other members are for proper alignment.  There's no
     standard way to guarantee stack alignment, but this seems enough
     in practice.  */
  long double ld;
  long l;
  void *p;
} alternate_signal_stack;

static void
null_action (int signo __attribute__ ((unused)))
{
}

#endif /* SIGALTSTACK || LIBSIGSEGV */

/* Only use libsigsegv if we need it; platforms like Solaris can
   detect stack overflow without the overhead of an external
   library.  */
#if HAVE_LIBSIGSEGV && ! HAVE_XSI_STACK_OVERFLOW_HEURISTIC

/* Nonzero if general segv handler could not be installed.  */
static volatile int segv_handler_missing;

/* Handle a segmentation violation and exit if it cannot be stack
   overflow.  This function is async-signal-safe.  */

static int segv_handler (void *address __attribute__ ((unused)),
                         int serious)
{
# if DEBUG
  {
    char buf[1024];
    sprintf (buf, "segv_handler serious=%d\n", serious);
    write (STDERR_FILENO, buf, strlen (buf));
  }
# endif

  /* If this fault is not serious, return 0 to let the stack overflow
     handler take a shot at it.  */
  if (!serious)
    return 0;
  die (SIGSEGV);
}

/* Handle a segmentation violation that is likely to be a stack
   overflow and exit.  This function is async-signal-safe.  */

static _Noreturn void
overflow_handler (int emergency,
                  stackoverflow_context_t context __attribute__ ((unused)))
{
# if DEBUG
  {
    char buf[1024];
    sprintf (buf, "overflow_handler emergency=%d segv_handler_missing=%d\n",
             emergency, segv_handler_missing);
    write (STDERR_FILENO, buf, strlen (buf));
  }
# endif

  die ((!emergency || segv_handler_missing) ? 0 : SIGSEGV);
}

int
c_stack_action (void (*action) (int))
{
  segv_action = action ? action : null_action;
  program_error_message = _("program error");
  stack_overflow_message = _("stack overflow");

  /* Always install the overflow handler.  */
  if (stackoverflow_install_handler (overflow_handler,
                                     alternate_signal_stack.buffer,
                                     sizeof alternate_signal_stack.buffer))
    {
      errno = ENOTSUP;
      return -1;
    }
  /* Try installing a general handler; if it fails, then treat all
     segv as stack overflow.  */
  segv_handler_missing = sigsegv_install_handler (segv_handler);
  return 0;
}

#elif HAVE_SIGALTSTACK && HAVE_DECL_SIGALTSTACK && HAVE_STACK_OVERFLOW_HANDLING

# if SIGINFO_WORKS

/* Handle a segmentation violation and exit.  This function is
   async-signal-safe.  */

static _Noreturn void
segv_handler (int signo, siginfo_t *info,
              void *context __attribute__ ((unused)))
{
  /* Clear SIGNO if it seems to have been a stack overflow.  */
#  if ! HAVE_XSI_STACK_OVERFLOW_HEURISTIC
  /* We can't easily determine whether it is a stack overflow; so
     assume that the rest of our program is perfect (!) and that
     this segmentation violation is a stack overflow.

     Note that although both Linux and Solaris provide
     sigaltstack, SA_ONSTACK, and SA_SIGINFO, currently only
     Solaris satisfies the XSI heuristic.  This is because
     Solaris populates uc_stack with the details of the
     interrupted stack, while Linux populates it with the details
     of the current stack.  */
  signo = 0;
#  else
  if (0 < info->si_code)
    {
      /* If the faulting address is within the stack, or within one
         page of the stack, assume that it is a stack overflow.  */
      ucontext_t const *user_context = context;
      char const *stack_base = user_context->uc_stack.ss_sp;
      size_t stack_size = user_context->uc_stack.ss_size;
      char const *faulting_address = info->si_addr;
      size_t page_size = sysconf (_SC_PAGESIZE);
      size_t s = faulting_address - stack_base + page_size;
      if (s < stack_size + 2 * page_size)
        signo = 0;

#   if DEBUG
      {
        char buf[1024];
        sprintf (buf,
                 "segv_handler fault=%p base=%p size=%lx page=%lx signo=%d\n",
                 faulting_address, stack_base, (unsigned long) stack_size,
                 (unsigned long) page_size, signo);
        write (STDERR_FILENO, buf, strlen (buf));
      }
#   endif
    }
#  endif

  die (signo);
}
# endif

int
c_stack_action (void (*action) (int))
{
  int r;
  stack_t st;
  struct sigaction act;
  st.ss_flags = 0;
# if SIGALTSTACK_SS_REVERSED
  /* Irix mistakenly treats ss_sp as the upper bound, rather than
     lower bound, of the alternate stack.  */
  st.ss_sp = alternate_signal_stack.buffer + SIGSTKSZ - sizeof (void *);
  st.ss_size = sizeof alternate_signal_stack.buffer - sizeof (void *);
# else
  st.ss_sp = alternate_signal_stack.buffer;
  st.ss_size = sizeof alternate_signal_stack.buffer;
# endif
  r = sigaltstack (&st, NULL);
  if (r != 0)
    return r;

  segv_action = action ? action : null_action;
  program_error_message = _("program error");
  stack_overflow_message = _("stack overflow");

  sigemptyset (&act.sa_mask);

# if SIGINFO_WORKS
  /* POSIX 1003.1-2001 says SA_RESETHAND implies SA_NODEFER, but
     this is not true on Solaris 8 at least.  It doesn't hurt to use
     SA_NODEFER here, so leave it in.  */
  act.sa_flags = SA_NODEFER | SA_ONSTACK | SA_RESETHAND | SA_SIGINFO;
  act.sa_sigaction = segv_handler;
# else
  act.sa_flags = SA_NODEFER | SA_ONSTACK | SA_RESETHAND;
  act.sa_handler = die;
# endif

# if FAULT_YIELDS_SIGBUS
  if (sigaction (SIGBUS, &act, NULL) < 0)
    return -1;
# endif
  return sigaction (SIGSEGV, &act, NULL);
}

#else /* ! ((HAVE_SIGALTSTACK && HAVE_DECL_SIGALTSTACK
             && HAVE_STACK_OVERFLOW_HANDLING) || HAVE_LIBSIGSEGV) */

int
c_stack_action (void (*action) (int)  __attribute__ ((unused)))
{
#if (defined _MSC_VER) && (_MSC_VER < 1800)
#else
  errno = ENOTSUP;
#endif
  return -1;
}

#endif
