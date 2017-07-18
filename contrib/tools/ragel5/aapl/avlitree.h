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

#ifndef _AAPL_AVLITREE_H
#define _AAPL_AVLITREE_H

#include "compare.h"
#include "dlistmel.h"

/**
 * \addtogroup avlitree
 * @{
 */

/**
 * \class AvliTree
 * \brief Linked AVL tree.
 *
 * AvliTree is the standard linked by-structure AVL tree. To use this
 * structure the user must define an element type and give it the necessary
 * properties. At the very least it must have a getKey() function that will be
 * used to compare the relative ordering of elements and tree management data
 * necessary for the AVL algorithm. An element type can acquire the management
 * data by inheriting the AvliTreeEl class.
 *
 * AvliTree does not presume to manage the allocation of elements in the tree.
 * The destructor will not delete the items in the tree, instead the elements
 * must be explicitly de-allocated by the user if necessary and when it is
 * safe to do so. The empty() routine will traverse the tree and delete all
 * items. 
 *
 * Since the tree does not manage the elements, it can contain elements that
 * are allocated statically or that are part of another data structure.
 *
 * \include ex_avlitree.cpp
 */

/*@}*/

#define BASE_EL(name) name
#define BASEKEY(name) name
#define BASELIST DListMel< Element, AvliTreeEl<Element> >
#define AVLMEL_CLASSDEF class Element, class Key, class Compare = CmpOrd<Key>
#define AVLMEL_TEMPDEF class Element, class Key, class Compare
#define AVLMEL_TEMPUSE Element, Key, Compare
#define AvlTree AvliTree
#define WALKABLE

#include "avlcommon.h"

#undef BASE_EL
#undef BASEKEY
#undef BASELIST
#undef AVLMEL_CLASSDEF
#undef AVLMEL_TEMPDEF
#undef AVLMEL_TEMPUSE
#undef AvlTree
#undef WALKABLE

#endif /* _AAPL_AVLITREE_H */
