/* Plain mutexes (native Windows implementation).
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
#include "windows-mutex.h"

#include <errno.h>

void
glwthread_mutex_init (glwthread_mutex_t *mutex)
{
  InitializeCriticalSection (&mutex->lock);
  mutex->guard.done = 1;
}

int
glwthread_mutex_lock (glwthread_mutex_t *mutex)
{
  if (!mutex->guard.done)
    {
      if (InterlockedIncrement (&mutex->guard.started) == 0)
        /* This thread is the first one to need this mutex.  Initialize it.  */
        glwthread_mutex_init (mutex);
      else
        {
          /* Don't let mutex->guard.started grow and wrap around.  */
          InterlockedDecrement (&mutex->guard.started);
          /* Yield the CPU while waiting for another thread to finish
             initializing this mutex.  */
          while (!mutex->guard.done)
            Sleep (0);
        }
    }
  EnterCriticalSection (&mutex->lock);
  return 0;
}

int
glwthread_mutex_trylock (glwthread_mutex_t *mutex)
{
  if (!mutex->guard.done)
    {
      if (InterlockedIncrement (&mutex->guard.started) == 0)
        /* This thread is the first one to need this mutex.  Initialize it.  */
        glwthread_mutex_init (mutex);
      else
        {
          /* Don't let mutex->guard.started grow and wrap around.  */
          InterlockedDecrement (&mutex->guard.started);
          /* Let another thread finish initializing this mutex, and let it also
             lock this mutex.  */
          return EBUSY;
        }
    }
  if (!TryEnterCriticalSection (&mutex->lock))
    return EBUSY;
  return 0;
}

int
glwthread_mutex_unlock (glwthread_mutex_t *mutex)
{
  if (!mutex->guard.done)
    return EINVAL;
  LeaveCriticalSection (&mutex->lock);
  return 0;
}

int
glwthread_mutex_destroy (glwthread_mutex_t *mutex)
{
  if (!mutex->guard.done)
    return EINVAL;
  DeleteCriticalSection (&mutex->lock);
  mutex->guard.done = 0;
  return 0;
}
