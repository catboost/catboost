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

#ifndef _AAPL_BSTMAP_H
#define _AAPL_BSTMAP_H

#include "compare.h"
#include "vector.h"

#ifdef AAPL_NAMESPACE
namespace Aapl {
#endif

/**
 * \brief Element for BstMap.
 *
 * Stores the key and value pair. 
 */
template <class Key, class Value> struct BstMapEl
{
	BstMapEl() {}
	BstMapEl(const Key &key) : key(key) {}
	BstMapEl(const Key &key, const Value &val) : key(key), value(val) {}

	/** \brief The key */
	Key key;

	/** \brief The value. */
	Value value;
};

#ifdef AAPL_NAMESPACE
}
#endif

/**
 * \addtogroup bst 
 * @{
 */

/** 
 * \class BstMap
 * \brief Binary search table for key and value pairs.
 *
 * BstMap stores key and value pairs in each element. The key and value can be
 * any type. A compare class for the key must be supplied.
 */

/*@}*/

#define BST_TEMPL_DECLARE class Key, class Value, \
		class Compare = CmpOrd<Key>, class Resize = ResizeExpn
#define BST_TEMPL_DEF class Key, class Value, class Compare, class Resize
#define BST_TEMPL_USE Key, Value, Compare, Resize
#define GET_KEY(el) ((el).key)
#define BstTable BstMap
#define Element BstMapEl<Key, Value>
#define BSTMAP

#include "bstcommon.h"

#undef BST_TEMPL_DECLARE
#undef BST_TEMPL_DEF
#undef BST_TEMPL_USE
#undef GET_KEY
#undef BstTable
#undef Element
#undef BSTMAP

/**
 * \fn BstMap::insert(const Key &key, BstMapEl<Key, Value> **lastFound)
 * \brief Insert the given key.
 *
 * If the given key does not already exist in the table then a new element
 * having key is inserted. They key copy constructor and value default
 * constructor are used to place the pair in the table. If lastFound is given,
 * it is set to the new entry created. If the insert fails then lastFound is
 * set to the existing pair of the same key.
 *
 * \returns The new element created upon success, null upon failure.
 */

/**
 * \fn BstMap::insertMulti(const Key &key)
 * \brief Insert the given key even if it exists already.
 *
 * If the key exists already then the new element having key is placed next
 * to some other pair of the same key. InsertMulti cannot fail. The key copy
 * constructor and the value default constructor are used to place the pair in
 * the table.
 *
 * \returns The new element created.
 */

#endif /* _AAPL_BSTMAP_H */
