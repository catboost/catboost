/* Bitset vectors.

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

#ifndef _BITSETV_H
#define _BITSETV_H

#include "bitset.h"

typedef bitset * bitsetv;

/* Create a vector of N_VECS bitsets, each of N_BITS, and of
   type TYPE.  */
bitsetv bitsetv_alloc (bitset_bindex, bitset_bindex, enum bitset_type);

/* Create a vector of N_VECS bitsets, each of N_BITS, and with
   attribute hints specified by ATTR.  */
bitsetv bitsetv_create (bitset_bindex, bitset_bindex, unsigned);

/* Free vector of bitsets.  Do nothing if NULL.  */
void bitsetv_free (bitsetv);

/* Zero vector of bitsets.  */
void bitsetv_zero (bitsetv);

/* Set vector of bitsets.  */
void bitsetv_ones (bitsetv);

/* Given a vector BSETV of N bitsets of size N, modify its contents to
   be the transitive closure of what was given.  */
void bitsetv_transitive_closure (bitsetv);

/* Given a vector BSETV of N bitsets of size N, modify its contents to
   be the reflexive transitive closure of what was given.  This is
   the same as transitive closure but with all bits on the diagonal
   of the bit matrix set.  */
void bitsetv_reflexive_transitive_closure (bitsetv);

/* Dump vector of bitsets.  */
void bitsetv_dump (FILE *, const char *, const char *, bitsetv);

/* Function to debug vector of bitsets from debugger.  */
void debug_bitsetv (bitsetv);

/* Dump vector of bitsets as a matrix.  */
void bitsetv_matrix_dump (FILE *, const char *, bitsetv);

#endif  /* _BITSETV_H  */
