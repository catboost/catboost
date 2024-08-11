/* Hash table for sequential list, set, and map data type.
   Copyright (C) 2006, 2009-2020 Free Software Foundation, Inc.
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
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* Common code of
   gl_linkedhash_list.c, gl_avltreehash_list.c, gl_rbtreehash_list.c,
   gl_linkedhash_set.c, gl_hash_set.c,
   gl_linkedhash_map.c, gl_hash_map.c.  */

#include "gl_anyhash_primes.h"

/* Resizes the hash table with a new estimated size.  */
static void
hash_resize (CONTAINER_T container, size_t estimate)
{
  size_t new_size = next_prime (estimate);

  if (new_size > container->table_size)
    {
      gl_hash_entry_t *old_table = container->table;
      /* Allocate the new table.  */
      gl_hash_entry_t *new_table;
      size_t i;

      if (size_overflow_p (xtimes (new_size, sizeof (gl_hash_entry_t))))
        goto fail;
      new_table =
        (gl_hash_entry_t *) calloc (new_size, sizeof (gl_hash_entry_t));
      if (new_table == NULL)
        goto fail;

      /* Iterate through the entries of the old table.  */
      for (i = container->table_size; i > 0; )
        {
          gl_hash_entry_t node = old_table[--i];

          while (node != NULL)
            {
              gl_hash_entry_t next = node->hash_next;
              /* Add the entry to the new table.  */
              size_t bucket = node->hashcode % new_size;
              node->hash_next = new_table[bucket];
              new_table[bucket] = node;

              node = next;
            }
        }

      container->table = new_table;
      container->table_size = new_size;
      free (old_table);
    }
  return;

 fail:
  /* Just continue without resizing the table.  */
  return;
}

/* Resizes the hash table if needed, after CONTAINER_COUNT (container) was
   incremented.  */
static void
hash_resize_after_add (CONTAINER_T container)
{
  size_t count = CONTAINER_COUNT (container);
  size_t estimate = xsum (count, count / 2); /* 1.5 * count */
  if (estimate > container->table_size)
    hash_resize (container, estimate);
}
