/* Sequential list data type implemented by a linked list.
   Copyright (C) 2006, 2009-2013 Free Software Foundation, Inc.
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

/* Common code of gl_linked_list.c and gl_linkedhash_list.c.  */

/* -------------------------- gl_list_t Data Type -------------------------- */

/* Concrete list node implementation, valid for this file only.  */
struct gl_list_node_impl
{
#if WITH_HASHTABLE
  struct gl_hash_entry h;           /* hash table entry fields; must be first */
#endif
  struct gl_list_node_impl *next;
  struct gl_list_node_impl *prev;
  const void *value;
};

/* Concrete gl_list_impl type, valid for this file only.  */
struct gl_list_impl
{
  struct gl_list_impl_base base;
#if WITH_HASHTABLE
  /* A hash table: managed as an array of collision lists.  */
  struct gl_hash_entry **table;
  size_t table_size;
#endif
  /* A circular list anchored at root.
     The first node is = root.next, the last node is = root.prev.
     The root's value is unused.  */
  struct gl_list_node_impl root;
  /* Number of list nodes, excluding the root.  */
  size_t count;
};
