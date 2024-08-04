/* Functions to support lbitsets.

   Copyright (C) 2002, 2004, 2009-2015, 2018-2020 Free Software Foundation,
   Inc.

   Contributed by Michael Hayes (m.hayes@elec.canterbury.ac.nz).

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

#ifndef _BITSET_LIST_H
#define _BITSET_LIST_H

#include "bitset.h"

size_t lbitset_bytes (bitset_bindex);

bitset lbitset_init (bitset, bitset_bindex);

void lbitset_release_memory (void);

#endif
