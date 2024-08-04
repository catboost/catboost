/* hash - hashing table processing.

   Copyright (C) 2019-2020 Free Software Foundation, Inc.

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

#include <config.h>

/* Specification.  */
#include "hash.h"

#include "xalloc.h"

/* Same as hash_initialize, but invokes xalloc_die on memory
   exhaustion.  */

Hash_table *
hash_xinitialize (size_t candidate, const Hash_tuning *tuning,
                  Hash_hasher hasher, Hash_comparator comparator,
                  Hash_data_freer data_freer)
{
  Hash_table *res =
    hash_initialize (candidate, tuning, hasher, comparator, data_freer);
  if (!res)
    xalloc_die ();
  return res;
}
