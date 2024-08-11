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

/* Hash table entry.  */
struct gl_hash_entry
{
  struct gl_hash_entry *hash_next;  /* chain of entries in same bucket */
  size_t hashcode;                  /* cache of the hash code of
                                       - the key (for map data type) or
                                       - the value (for list, set data types) */
};
typedef struct gl_hash_entry * gl_hash_entry_t;
