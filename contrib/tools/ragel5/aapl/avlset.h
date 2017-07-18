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

#ifndef _AAPL_AVLSET_H
#define _AAPL_AVLSET_H

#include "compare.h"

/**
 * \addtogroup avltree 
 * @{
 */

/**
 * \class AvlSet
 * \brief Key-only oriented tree.
 *
 * AvlSet stores only keys in elements that are managed by the tree. AvlSet
 * requires that a Key type and a class containing a compare() routine
 * for Key be given. Items are inserted with just a key value.
 *
 * AvlSet assumes all elements in the tree are allocated on the heap and are
 * to be managed by the tree. This means that the class destructor will delete
 * the contents of the tree. A deep copy will cause existing elements to be
 * deleted first.
 *
 * \include ex_avlset.cpp
 */

/*@}*/

#define AVLTREE_SET
#define BASE_EL(name) name
#define BASEKEY(name) name
#define AVLMEL_CLASSDEF class Key, class Compare = CmpOrd<Key>
#define AVLMEL_TEMPDEF class Key, class Compare
#define AVLMEL_TEMPUSE Key, Compare
#define AvlTree AvlSet
#define Element AvlSetEl<Key>

#include "avlcommon.h"

#undef AVLTREE_SET
#undef BASE_EL
#undef BASEKEY
#undef AVLMEL_CLASSDEF
#undef AVLMEL_TEMPDEF
#undef AVLMEL_TEMPUSE
#undef AvlTree
#undef Element

#endif /* _AAPL_AVLSET_H */
