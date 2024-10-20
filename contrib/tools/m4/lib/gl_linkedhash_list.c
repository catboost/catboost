/* Sequential list data type implemented by a hash table with a linked list.
   Copyright (C) 2006, 2008-2016 Free Software Foundation, Inc.
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
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#include <config.h>

/* Specification.  */
#include "gl_linkedhash_list.h"

#include <stdint.h> /* for SIZE_MAX */
#include <stdlib.h>

#include "xsize.h"

#ifndef uintptr_t
# define uintptr_t unsigned long
#endif

#define WITH_HASHTABLE 1

/* -------------------------- gl_list_t Data Type -------------------------- */

/* Generic hash-table code.  */
#include "gl_anyhash_list1.h"

/* Generic linked list code.  */
#include "gl_anylinked_list1.h"

/* Generic hash-table code.  */
#include "gl_anyhash_list2.h"

/* Resize the hash table if needed, after list->count was incremented.  */
static void
hash_resize_after_add (gl_list_t list)
{
  size_t count = list->count;
  size_t estimate = xsum (count, count / 2); /* 1.5 * count */
  if (estimate > list->table_size)
    hash_resize (list, estimate);
}

/* Add a node to the hash table structure.  */
static void
add_to_bucket (gl_list_t list, gl_list_node_t node)
{
  size_t bucket = node->h.hashcode % list->table_size;

  node->h.hash_next = list->table[bucket];
  list->table[bucket] = &node->h;
}
/* Tell all compilers that the return value is 0.  */
#define add_to_bucket(list,node)  ((add_to_bucket) (list, node), 0)

/* Remove a node from the hash table structure.  */
static void
remove_from_bucket (gl_list_t list, gl_list_node_t node)
{
  size_t bucket = node->h.hashcode % list->table_size;
  gl_hash_entry_t *p;

  for (p = &list->table[bucket]; ; p = &(*p)->hash_next)
    {
      if (*p == &node->h)
        {
          *p = node->h.hash_next;
          break;
        }
      if (*p == NULL)
        /* node is not in the right bucket.  Did the hash codes
           change inadvertently?  */
        abort ();
    }
}

/* Generic linked list code.  */
#include "gl_anylinked_list2.h"


const struct gl_list_implementation gl_linkedhash_list_implementation =
  {
    gl_linked_nx_create_empty,
    gl_linked_nx_create,
    gl_linked_size,
    gl_linked_node_value,
    gl_linked_node_nx_set_value,
    gl_linked_next_node,
    gl_linked_previous_node,
    gl_linked_get_at,
    gl_linked_nx_set_at,
    gl_linked_search_from_to,
    gl_linked_indexof_from_to,
    gl_linked_nx_add_first,
    gl_linked_nx_add_last,
    gl_linked_nx_add_before,
    gl_linked_nx_add_after,
    gl_linked_nx_add_at,
    gl_linked_remove_node,
    gl_linked_remove_at,
    gl_linked_remove,
    gl_linked_list_free,
    gl_linked_iterator,
    gl_linked_iterator_from_to,
    gl_linked_iterator_next,
    gl_linked_iterator_free,
    gl_linked_sortedlist_search,
    gl_linked_sortedlist_search_from_to,
    gl_linked_sortedlist_indexof,
    gl_linked_sortedlist_indexof_from_to,
    gl_linked_sortedlist_nx_add,
    gl_linked_sortedlist_remove
  };
