/* Stack overflow handling.

   Copyright (C) 2002, 2004, 2008-2013 Free Software Foundation, Inc.

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


/* Set up ACTION so that it is invoked on C stack overflow and on other,
   stack-unrelated, segmentation violation.
   Return -1 (setting errno) if this cannot be done.

   When a stack overflow or segmentation violation occurs:
   1) ACTION is called.  It is passed an argument equal to
        - 0, for a stack overflow,
        - SIGSEGV, for a segmentation violation that does not appear related
          to stack overflow.
      On many platforms the two cases are hard to distinguish; when in doubt,
      zero is passed.
   2) If ACTION returns, a message is written to standard error, and the
      program is terminated: in the case of stack overflow, with exit code
      exit_failure (see "exitfail.h"), otherwise through a signal SIGSEGV.

   A null ACTION acts like an action that does nothing.

   ACTION must be async-signal-safe.  ACTION together with its callees
   must not require more than SIGSTKSZ bytes of stack space.  Also,
   ACTION should not call longjmp, because this implementation does
   not guarantee that it is safe to return to the original stack.

   This function may install a handler for the SIGSEGV signal or for the SIGBUS
   signal or exercise other system dependent exception handling APIs.  */

extern int c_stack_action (void (* /*action*/) (int));
