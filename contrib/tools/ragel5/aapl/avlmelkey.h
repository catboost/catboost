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

#ifndef _AAPL_AVLMELKEY_H
#define _AAPL_AVLMELKEY_H

#include "compare.h"

/**
 * \addtogroup avltree 
 * @{
 */

/**
 * \class AvlMelKey
 * \brief AVL tree for elements appearing in multiple trees with different keys.
 *
 * AvlMelKey is similar to AvlMel, except that an additional template
 * parameter, BaseKey, is provided for resolving ambiguous references to
 * getKey(). This means that if an element is stored in multiple trees, each
 * tree can use a different key for ordering the elements in it. Using
 * AvlMelKey an array of data structures can be indexed with an O(log(n))
 * search on two or more of the values contained within it and without
 * allocating any additional data.
 *
 * AvlMelKey does not assume ownership of elements in the tree. The destructor
 * will not delete the elements. If the user wishes to explicitly deallocate
 * all the items in the tree the empty() routine is available. 
 *
 * \include ex_avlmelkey.cpp
 */

/*@}*/

#define BASE_EL(name) BaseEl::name
#define BASEKEY(name) BaseKey::name
#define AVLMEL_CLASSDEF class Element, class Key, class BaseEl, \
		class BaseKey, class Compare = CmpOrd<Key>
#define AVLMEL_TEMPDEF class Element, class Key, class BaseEl, \
		class BaseKey, class Compare
#define AVLMEL_TEMPUSE Element, Key, BaseEl, BaseKey, Compare
#define AvlTree AvlMelKey

#include "avlcommon.h"

#undef BASE_EL
#undef BASEKEY
#undef AVLMEL_CLASSDEF
#undef AVLMEL_TEMPDEF
#undef AVLMEL_TEMPUSE
#undef AvlTree

#endif /* _AAPL_AVLMELKEY_H */
