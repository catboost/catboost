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

#ifndef _AAPL_SBSTTABLE_H
#define _AAPL_SBSTTABLE_H

#include "compare.h"
#include "svector.h"

/**
 * \addtogroup bst 
 * @{
 */

/** 
 * \class SBstTable
 * \brief Copy-on-write binary search table for structures that contain a key.
 *
 * This is a basic binary search table that employs a copy-on-write data
 * storage mechanism. It can be used to contain a structure that has a key and
 * possibly some data. The key should be a member of the element class and
 * accessible with getKey(). A class containing the compare routine must be
 * supplied.
 */

/*@}*/

#define BST_TEMPL_DECLARE class Element, class Key, \
		class Compare = CmpOrd<Key>, class Resize = ResizeExpn
#define BST_TEMPL_DEF class Element, class Key, class Compare, class Resize
#define BST_TEMPL_USE Element, Key, Compare, Resize
#define GET_KEY(el) ((el).getKey())
#define BstTable SBstTable
#define Vector SVector
#define Table STable
#define BSTTABLE
#define SHARED_BST

#include "bstcommon.h"

#undef BST_TEMPL_DECLARE
#undef BST_TEMPL_DEF
#undef BST_TEMPL_USE
#undef GET_KEY
#undef BstTable
#undef Vector
#undef Table
#undef BSTTABLE
#undef SHARED_BST

/**
 * \fn SBstTable::insert(const Key &key, Element **lastFound)
 * \brief Insert a new element with the given key.
 *
 * If the given key does not already exist in the table a new element is
 * inserted with the given key. A constructor taking only const Key& is used
 * to initialize the new element. If lastFound is given, it is set to the new
 * element created. If the insert fails then lastFound is set to the existing
 * element with the same key. 
 *
 * \returns The new element created upon success, null upon failure.
 */

/**
 * \fn SBstTable::insertMulti(const Key &key)
 * \brief Insert a new element even if the key exists already.
 *
 * If the key exists already then the new element is placed next to some
 * element with the same key. InsertMulti cannot fail. A constructor taking
 * only const Key& is used to initialize the new element.
 *
 * \returns The new element created.
 */

#endif /* _AAPL_SBSTTABLE_H */
