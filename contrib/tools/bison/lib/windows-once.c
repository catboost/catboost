/* Once-only control (native Windows implementation).
   Copyright (C) 2005-2020 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

/* Written by Bruno Haible <bruno@clisp.org>, 2005.
   Based on GCC's gthr-win32.h.  */

#include <config.h>

/* Specification.  */
#include "windows-once.h"

#include <stdlib.h>

void
glwthread_once (glwthread_once_t *once_control, void (*initfunction) (void))
{
  if (once_control->inited <= 0)
    {
      if (InterlockedIncrement (&once_control->started) == 0)
        {
          /* This thread is the first one to come to this once_control.  */
          InitializeCriticalSection (&once_control->lock);
          EnterCriticalSection (&once_control->lock);
          once_control->inited = 0;
          initfunction ();
          once_control->inited = 1;
          LeaveCriticalSection (&once_control->lock);
        }
      else
        {
          /* Don't let once_control->started grow and wrap around.  */
          InterlockedDecrement (&once_control->started);
          /* Some other thread has already started the initialization.
             Yield the CPU while waiting for the other thread to finish
             initializing and taking the lock.  */
          while (once_control->inited < 0)
            Sleep (0);
          if (once_control->inited <= 0)
            {
              /* Take the lock.  This blocks until the other thread has
                 finished calling the initfunction.  */
              EnterCriticalSection (&once_control->lock);
              LeaveCriticalSection (&once_control->lock);
              if (!(once_control->inited > 0))
                abort ();
            }
        }
    }
}
