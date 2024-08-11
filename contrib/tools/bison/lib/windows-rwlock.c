/* Read-write locks (native Windows implementation).
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
#include "windows-rwlock.h"

#include <errno.h>
#include <stdlib.h>

/* Don't assume that UNICODE is not defined.  */
#undef CreateEvent
#define CreateEvent CreateEventA

/* In this file, the waitqueues are implemented as circular arrays.  */
#define glwthread_waitqueue_t glwthread_carray_waitqueue_t

static void
glwthread_waitqueue_init (glwthread_waitqueue_t *wq)
{
  wq->array = NULL;
  wq->count = 0;
  wq->alloc = 0;
  wq->offset = 0;
}

/* Enqueues the current thread, represented by an event, in a wait queue.
   Returns INVALID_HANDLE_VALUE if an allocation failure occurs.  */
static HANDLE
glwthread_waitqueue_add (glwthread_waitqueue_t *wq)
{
  HANDLE event;
  unsigned int index;

  if (wq->count == wq->alloc)
    {
      unsigned int new_alloc = 2 * wq->alloc + 1;
      HANDLE *new_array =
        (HANDLE *) realloc (wq->array, new_alloc * sizeof (HANDLE));
      if (new_array == NULL)
        /* No more memory.  */
        return INVALID_HANDLE_VALUE;
      /* Now is a good opportunity to rotate the array so that its contents
         starts at offset 0.  */
      if (wq->offset > 0)
        {
          unsigned int old_count = wq->count;
          unsigned int old_alloc = wq->alloc;
          unsigned int old_offset = wq->offset;
          unsigned int i;
          if (old_offset + old_count > old_alloc)
            {
              unsigned int limit = old_offset + old_count - old_alloc;
              for (i = 0; i < limit; i++)
                new_array[old_alloc + i] = new_array[i];
            }
          for (i = 0; i < old_count; i++)
            new_array[i] = new_array[old_offset + i];
          wq->offset = 0;
        }
      wq->array = new_array;
      wq->alloc = new_alloc;
    }
  /* Whether the created event is a manual-reset one or an auto-reset one,
     does not matter, since we will wait on it only once.  */
  event = CreateEvent (NULL, TRUE, FALSE, NULL);
  if (event == INVALID_HANDLE_VALUE)
    /* No way to allocate an event.  */
    return INVALID_HANDLE_VALUE;
  index = wq->offset + wq->count;
  if (index >= wq->alloc)
    index -= wq->alloc;
  wq->array[index] = event;
  wq->count++;
  return event;
}

/* Notifies the first thread from a wait queue and dequeues it.  */
static void
glwthread_waitqueue_notify_first (glwthread_waitqueue_t *wq)
{
  SetEvent (wq->array[wq->offset + 0]);
  wq->offset++;
  wq->count--;
  if (wq->count == 0 || wq->offset == wq->alloc)
    wq->offset = 0;
}

/* Notifies all threads from a wait queue and dequeues them all.  */
static void
glwthread_waitqueue_notify_all (glwthread_waitqueue_t *wq)
{
  unsigned int i;

  for (i = 0; i < wq->count; i++)
    {
      unsigned int index = wq->offset + i;
      if (index >= wq->alloc)
        index -= wq->alloc;
      SetEvent (wq->array[index]);
    }
  wq->count = 0;
  wq->offset = 0;
}

void
glwthread_rwlock_init (glwthread_rwlock_t *lock)
{
  InitializeCriticalSection (&lock->lock);
  glwthread_waitqueue_init (&lock->waiting_readers);
  glwthread_waitqueue_init (&lock->waiting_writers);
  lock->runcount = 0;
  lock->guard.done = 1;
}

int
glwthread_rwlock_rdlock (glwthread_rwlock_t *lock)
{
  if (!lock->guard.done)
    {
      if (InterlockedIncrement (&lock->guard.started) == 0)
        /* This thread is the first one to need this lock.  Initialize it.  */
        glwthread_rwlock_init (lock);
      else
        {
          /* Don't let lock->guard.started grow and wrap around.  */
          InterlockedDecrement (&lock->guard.started);
          /* Yield the CPU while waiting for another thread to finish
             initializing this lock.  */
          while (!lock->guard.done)
            Sleep (0);
        }
    }
  EnterCriticalSection (&lock->lock);
  /* Test whether only readers are currently running, and whether the runcount
     field will not overflow, and whether no writer is waiting.  The latter
     condition is because POSIX recommends that "write locks shall take
     precedence over read locks", to avoid "writer starvation".  */
  if (!(lock->runcount + 1 > 0 && lock->waiting_writers.count == 0))
    {
      /* This thread has to wait for a while.  Enqueue it among the
         waiting_readers.  */
      HANDLE event = glwthread_waitqueue_add (&lock->waiting_readers);
      if (event != INVALID_HANDLE_VALUE)
        {
          DWORD result;
          LeaveCriticalSection (&lock->lock);
          /* Wait until another thread signals this event.  */
          result = WaitForSingleObject (event, INFINITE);
          if (result == WAIT_FAILED || result == WAIT_TIMEOUT)
            abort ();
          CloseHandle (event);
          /* The thread which signalled the event already did the bookkeeping:
             removed us from the waiting_readers, incremented lock->runcount.  */
          if (!(lock->runcount > 0))
            abort ();
          return 0;
        }
      else
        {
          /* Allocation failure.  Weird.  */
          do
            {
              LeaveCriticalSection (&lock->lock);
              Sleep (1);
              EnterCriticalSection (&lock->lock);
            }
          while (!(lock->runcount + 1 > 0));
        }
    }
  lock->runcount++;
  LeaveCriticalSection (&lock->lock);
  return 0;
}

int
glwthread_rwlock_wrlock (glwthread_rwlock_t *lock)
{
  if (!lock->guard.done)
    {
      if (InterlockedIncrement (&lock->guard.started) == 0)
        /* This thread is the first one to need this lock.  Initialize it.  */
        glwthread_rwlock_init (lock);
      else
        {
          /* Don't let lock->guard.started grow and wrap around.  */
          InterlockedDecrement (&lock->guard.started);
          /* Yield the CPU while waiting for another thread to finish
             initializing this lock.  */
          while (!lock->guard.done)
            Sleep (0);
        }
    }
  EnterCriticalSection (&lock->lock);
  /* Test whether no readers or writers are currently running.  */
  if (!(lock->runcount == 0))
    {
      /* This thread has to wait for a while.  Enqueue it among the
         waiting_writers.  */
      HANDLE event = glwthread_waitqueue_add (&lock->waiting_writers);
      if (event != INVALID_HANDLE_VALUE)
        {
          DWORD result;
          LeaveCriticalSection (&lock->lock);
          /* Wait until another thread signals this event.  */
          result = WaitForSingleObject (event, INFINITE);
          if (result == WAIT_FAILED || result == WAIT_TIMEOUT)
            abort ();
          CloseHandle (event);
          /* The thread which signalled the event already did the bookkeeping:
             removed us from the waiting_writers, set lock->runcount = -1.  */
          if (!(lock->runcount == -1))
            abort ();
          return 0;
        }
      else
        {
          /* Allocation failure.  Weird.  */
          do
            {
              LeaveCriticalSection (&lock->lock);
              Sleep (1);
              EnterCriticalSection (&lock->lock);
            }
          while (!(lock->runcount == 0));
        }
    }
  lock->runcount--; /* runcount becomes -1 */
  LeaveCriticalSection (&lock->lock);
  return 0;
}

int
glwthread_rwlock_tryrdlock (glwthread_rwlock_t *lock)
{
  if (!lock->guard.done)
    {
      if (InterlockedIncrement (&lock->guard.started) == 0)
        /* This thread is the first one to need this lock.  Initialize it.  */
        glwthread_rwlock_init (lock);
      else
        {
          /* Don't let lock->guard.started grow and wrap around.  */
          InterlockedDecrement (&lock->guard.started);
          /* Yield the CPU while waiting for another thread to finish
             initializing this lock.  */
          while (!lock->guard.done)
            Sleep (0);
        }
    }
  /* It's OK to wait for this critical section, because it is never taken for a
     long time.  */
  EnterCriticalSection (&lock->lock);
  /* Test whether only readers are currently running, and whether the runcount
     field will not overflow, and whether no writer is waiting.  The latter
     condition is because POSIX recommends that "write locks shall take
     precedence over read locks", to avoid "writer starvation".  */
  if (!(lock->runcount + 1 > 0 && lock->waiting_writers.count == 0))
    {
      /* This thread would have to wait for a while.  Return instead.  */
      LeaveCriticalSection (&lock->lock);
      return EBUSY;
    }
  lock->runcount++;
  LeaveCriticalSection (&lock->lock);
  return 0;
}

int
glwthread_rwlock_trywrlock (glwthread_rwlock_t *lock)
{
  if (!lock->guard.done)
    {
      if (InterlockedIncrement (&lock->guard.started) == 0)
        /* This thread is the first one to need this lock.  Initialize it.  */
        glwthread_rwlock_init (lock);
      else
        {
          /* Don't let lock->guard.started grow and wrap around.  */
          InterlockedDecrement (&lock->guard.started);
          /* Yield the CPU while waiting for another thread to finish
             initializing this lock.  */
          while (!lock->guard.done)
            Sleep (0);
        }
    }
  /* It's OK to wait for this critical section, because it is never taken for a
     long time.  */
  EnterCriticalSection (&lock->lock);
  /* Test whether no readers or writers are currently running.  */
  if (!(lock->runcount == 0))
    {
      /* This thread would have to wait for a while.  Return instead.  */
      LeaveCriticalSection (&lock->lock);
      return EBUSY;
    }
  lock->runcount--; /* runcount becomes -1 */
  LeaveCriticalSection (&lock->lock);
  return 0;
}

int
glwthread_rwlock_unlock (glwthread_rwlock_t *lock)
{
  if (!lock->guard.done)
    return EINVAL;
  EnterCriticalSection (&lock->lock);
  if (lock->runcount < 0)
    {
      /* Drop a writer lock.  */
      if (!(lock->runcount == -1))
        abort ();
      lock->runcount = 0;
    }
  else
    {
      /* Drop a reader lock.  */
      if (!(lock->runcount > 0))
        {
          LeaveCriticalSection (&lock->lock);
          return EPERM;
        }
      lock->runcount--;
    }
  if (lock->runcount == 0)
    {
      /* POSIX recommends that "write locks shall take precedence over read
         locks", to avoid "writer starvation".  */
      if (lock->waiting_writers.count > 0)
        {
          /* Wake up one of the waiting writers.  */
          lock->runcount--;
          glwthread_waitqueue_notify_first (&lock->waiting_writers);
        }
      else
        {
          /* Wake up all waiting readers.  */
          lock->runcount += lock->waiting_readers.count;
          glwthread_waitqueue_notify_all (&lock->waiting_readers);
        }
    }
  LeaveCriticalSection (&lock->lock);
  return 0;
}

int
glwthread_rwlock_destroy (glwthread_rwlock_t *lock)
{
  if (!lock->guard.done)
    return EINVAL;
  if (lock->runcount != 0)
    return EBUSY;
  DeleteCriticalSection (&lock->lock);
  if (lock->waiting_readers.array != NULL)
    free (lock->waiting_readers.array);
  if (lock->waiting_writers.array != NULL)
    free (lock->waiting_writers.array);
  lock->guard.done = 0;
  return 0;
}
