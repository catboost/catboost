/* Thread-local storage (native Windows implementation).
   Copyright (C) 2005-2020 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* Written by Bruno Haible <bruno@clisp.org>, 2005.  */

#include <config.h>

/* Specification.  */
#include "windows-tls.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>

#include "windows-once.h"

void *
glwthread_tls_get (glwthread_tls_key_t key)
{
  return TlsGetValue (key);
}

int
glwthread_tls_set (glwthread_tls_key_t key, void *value)
{
  if (!TlsSetValue (key, value))
    return EINVAL;
  return 0;
}

/* The following variables keep track of TLS keys with non-NULL destructor.  */

static glwthread_once_t dtor_table_init_once = GLWTHREAD_ONCE_INIT;

static CRITICAL_SECTION dtor_table_lock;

struct dtor { glwthread_tls_key_t key; void (*destructor) (void *); };

/* The table of dtors.  */
static struct dtor *dtor_table;
/* Number of active entries in the dtor_table.  */
static unsigned int dtors_count;
/* Valid indices into dtor_table are 0..dtors_used-1.  */
static unsigned int dtors_used;
/* Allocation size of dtor_table.  */
static unsigned int dtors_allocated;
/* Invariant: 0 <= dtors_count <= dtors_used <= dtors_allocated.  */

/* Number of threads that are currently processing destructors.  */
static unsigned int dtor_processing_threads;

static void
dtor_table_initialize (void)
{
  InitializeCriticalSection (&dtor_table_lock);
  /* The other variables are already initialized to NULL or 0, respectively.  */
}

static void
dtor_table_ensure_initialized (void)
{
  glwthread_once (&dtor_table_init_once, dtor_table_initialize);
}

/* Shrinks dtors_used down to dtors_count, by replacing inactive entries
   with active ones.  */
static void
dtor_table_shrink_used (void)
{
  unsigned int i = 0;
  unsigned int j = dtors_used;

  for (;;)
    {
      BOOL i_found = FALSE;
      BOOL j_found = FALSE;
      /* Find the next inactive entry, from the left.  */
      for (; i < dtors_count;)
        {
          if (dtor_table[i].destructor == NULL)
            {
              i_found = TRUE;
              break;
            }
          i++;
        }

      /* Find the next active entry, from the right.  */
      for (; j > dtors_count;)
        {
          j--;
          if (dtor_table[j].destructor != NULL)
            {
              j_found = TRUE;
              break;
            }
        }

      if (i_found != j_found)
        /* dtors_count was apparently wrong.  */
        abort ();

      if (!i_found)
        break;

      /* i_found and j_found are TRUE.  Swap the two entries.  */
      dtor_table[i] = dtor_table[j];

      i++;
    }

  dtors_used = dtors_count;
}

void
glwthread_tls_process_destructors (void)
{
  unsigned int repeat;

  dtor_table_ensure_initialized ();

  EnterCriticalSection (&dtor_table_lock);
  if (dtor_processing_threads == 0)
    {
      /* Now it's the appropriate time for shrinking dtors_used.  */
      if (dtors_used > dtors_count)
        dtor_table_shrink_used ();
    }
  dtor_processing_threads++;

  for (repeat = GLWTHREAD_DESTRUCTOR_ITERATIONS; repeat > 0; repeat--)
    {
      unsigned int destructors_run = 0;

      /* Iterate across dtor_table.  We don't need to make a copy of dtor_table,
         because
           * When another thread calls glwthread_tls_key_create with a non-NULL
             destructor argument, this will possibly reallocate the dtor_table
             array and increase dtors_allocated as well as dtors_used and
             dtors_count, but it will not change dtors_used nor the contents of
             the first dtors_used entries of dtor_table.
           * When another thread calls glwthread_tls_key_delete, this will
             possibly set some 'destructor' member to NULL, thus marking an
             entry as inactive, but it will not otherwise change dtors_used nor
             the contents of the first dtors_used entries of dtor_table.  */
      unsigned int i_limit = dtors_used;
      unsigned int i;

      for (i = 0; i < i_limit; i++)
        {
          struct dtor current = dtor_table[i];
          if (current.destructor != NULL)
            {
              /* The current dtor has not been deleted yet.  */
              void *current_value = glwthread_tls_get (current.key);
              if (current_value != NULL)
                {
                  /* The current value is non-NULL.  Run the destructor.  */
                  glwthread_tls_set (current.key, NULL);
                  LeaveCriticalSection (&dtor_table_lock);
                  current.destructor (current_value);
                  EnterCriticalSection (&dtor_table_lock);
                  destructors_run++;
                }
            }
        }

      /* When all TLS values were already NULL, no further iterations are
         needed.  */
      if (destructors_run == 0)
        break;
    }

  dtor_processing_threads--;
  LeaveCriticalSection (&dtor_table_lock);
}

int
glwthread_tls_key_create (glwthread_tls_key_t *keyp, void (*destructor) (void *))
{
  if (destructor != NULL)
    {
      dtor_table_ensure_initialized ();

      EnterCriticalSection (&dtor_table_lock);
      if (dtor_processing_threads == 0)
        {
          /* Now it's the appropriate time for shrinking dtors_used.  */
          if (dtors_used > dtors_count)
            dtor_table_shrink_used ();
        }

      while (dtors_used == dtors_allocated)
        {
          /* Need to grow the dtor_table.  */
          unsigned int new_allocated = 2 * dtors_allocated + 1;
          if (new_allocated < 7)
            new_allocated = 7;
          if (new_allocated <= dtors_allocated) /* overflow? */
            new_allocated = UINT_MAX;

          LeaveCriticalSection (&dtor_table_lock);
          {
            struct dtor *new_table =
              (struct dtor *) malloc (new_allocated * sizeof (struct dtor));
            if (new_table == NULL)
              return ENOMEM;
            EnterCriticalSection (&dtor_table_lock);
            /* Attention! dtors_used, dtors_allocated may have changed!  */
            if (dtors_used < new_allocated)
              {
                if (dtors_allocated < new_allocated)
                  {
                    /* The new_table is useful.  */
                    memcpy (new_table, dtor_table,
                            dtors_used * sizeof (struct dtor));
                    dtor_table = new_table;
                    dtors_allocated = new_allocated;
                  }
                else
                  {
                    /* The new_table is not useful, since another thread
                       meanwhile allocated a drop_table that is at least
                       as large.  */
                    free (new_table);
                  }
                break;
              }
            /* The new_table is not useful, since other threads increased
               dtors_used.  Free it any retry.  */
            free (new_table);
          }
        }
      /* Here dtors_used < dtors_allocated.  */
      {
        /* Allocate a new key.  */
        glwthread_tls_key_t key = TlsAlloc ();
        if (key == (DWORD)-1)
          {
            LeaveCriticalSection (&dtor_table_lock);
            return EAGAIN;
          }
        /* Store the new dtor in the dtor_table, after all used entries.
           Do not overwrite inactive entries with indices < dtors_used, in order
           not to disturb glwthread_tls_process_destructors invocations that may
           be executing in other threads.  */
        dtor_table[dtors_used].key = key;
        dtor_table[dtors_used].destructor = destructor;
        dtors_used++;
        dtors_count++;
        LeaveCriticalSection (&dtor_table_lock);
        *keyp = key;
      }
    }
  else
    {
      /* Allocate a new key.  */
      glwthread_tls_key_t key = TlsAlloc ();
      if (key == (DWORD)-1)
        return EAGAIN;
      *keyp = key;
    }
  return 0;
}

int
glwthread_tls_key_delete (glwthread_tls_key_t key)
{
  /* Should the destructor be called for all threads that are currently running?
     Probably not, because
       - ISO C does not specify when the destructor is to be invoked at all.
       - In POSIX, the destructor functions specified with pthread_key_create()
         are invoked at thread exit.
       - It would be hard to implement, because there are no primitives for
         accessing thread-specific values from a different thread.  */
  dtor_table_ensure_initialized ();

  EnterCriticalSection (&dtor_table_lock);
  if (dtor_processing_threads == 0)
    {
      /* Now it's the appropriate time for shrinking dtors_used.  */
      if (dtors_used > dtors_count)
        dtor_table_shrink_used ();
      /* Here dtors_used == dtors_count.  */

      /* Find the key in dtor_table.  */
      {
        unsigned int i_limit = dtors_used;
        unsigned int i;

        for (i = 0; i < i_limit; i++)
          if (dtor_table[i].key == key)
            {
              if (i < dtors_used - 1)
                /* Swap the entries i and dtors_used - 1.  */
                dtor_table[i] = dtor_table[dtors_used - 1];
              dtors_count = dtors_used = dtors_used - 1;
              break;
            }
      }
    }
  else
    {
      /* Be careful not to disturb the glwthread_tls_process_destructors
         invocations that are executing in other threads.  */
      unsigned int i_limit = dtors_used;
      unsigned int i;

      for (i = 0; i < i_limit; i++)
        if (dtor_table[i].destructor != NULL /* skip inactive entries */
            && dtor_table[i].key == key)
          {
            /* Mark this entry as inactive.  */
            dtor_table[i].destructor = NULL;
            dtors_count = dtors_count - 1;
            break;
          }
    }
  LeaveCriticalSection (&dtor_table_lock);
  /* Now we have ensured that glwthread_tls_process_destructors will no longer
     use this key.  */

  if (!TlsFree (key))
    return EINVAL;
  return 0;
}
