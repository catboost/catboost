/*
 *  Copyright 2002 Adrian Thurston <thurston@cs.queensu.ca>
 */

/*  This file is part of Aapl.
 *
 *  Aapl is free software; you can redistribute it and/or modify it under the
 *  terms of the GNU Lesser General Public License as published by the Free
 *  Software Foundation; either version 2.1 of the License, or (at your option)
 *  any later version.
 *
 *  Aapl is distributed in the hope that it will be useful, but WITHOUT ANY
 *  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 *  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
 *  more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with Aapl; if not, write to the Free Software Foundation, Inc., 59
 *  Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#ifndef _AAPL_AVLIBASIC_H
#define _AAPL_AVLIBASIC_H

#include "compare.h"

/**
 * \addtogroup avlitree 
 * @{
 */

/**
 * \class AvliBasic
 * \brief Linked AVL Tree in which the entire element structure is the key.
 *
 * AvliBasic is a linked AVL tree that does not distinguish between the
 * element that it contains and the key. The entire element structure is the
 * key that is used to compare the relative ordering of elements. This is
 * similar to the BstSet structure.
 *
 * AvliBasic does not assume ownership of elements in the tree. Items must be
 * explicitly de-allocated.
 */

/*@}*/

#define BASE_EL(name) name
#define BASEKEY(name) name
#define AVLMEL_CLASSDEF class Element, class Compare
#define AVLMEL_TEMPDEF class Element, class Compare
#define AVLMEL_TEMPUSE Element, Compare
#define AvlTree AvliBasic
#define AVL_BASIC
#define WALKABLE

#include "avlcommon.h"

#undef BASE_EL
#undef BASEKEY
#undef AVLMEL_CLASSDEF
#undef AVLMEL_TEMPDEF
#undef AVLMEL_TEMPUSE
#undef AvlTree
#undef AVL_BASIC
#undef WALKABLE

#endif /* _AAPL_AVLIBASIC_H */
