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

#ifndef _AAPL_BSTSET_H
#define _AAPL_BSTSET_H

/**
 * \addtogroup bst 
 * @{
 */

/** 
 * \class BstSet
 * \brief Binary search table for types that are the key.
 *
 * BstSet is suitable for types that comprise the entire key. Rather than look
 * into the element to retrieve the key, the element is the key. A class that
 * contains a comparison routine for the key must be given.
 */

/*@}*/

#include "compare.h"
#include "vector.h"

#define BST_TEMPL_DECLARE class Key, class Compare = CmpOrd<Key>, \
		class Resize = ResizeExpn
#define BST_TEMPL_DEF class Key, class Compare, class Resize
#define BST_TEMPL_USE Key, Compare, Resize
#define GET_KEY(el) (el)
#define BstTable BstSet
#define Element Key
#define BSTSET

#include "bstcommon.h"

#undef BST_TEMPL_DECLARE
#undef BST_TEMPL_DEF
#undef BST_TEMPL_USE
#undef GET_KEY
#undef BstTable
#undef Element
#undef BSTSET

/**
 * \fn BstSet::insert(const Key &key, Key **lastFound)
 * \brief Insert the given key.
 *
 * If the given key does not already exist in the table then it is inserted.
 * The key's copy constructor is used to place the item in the table. If
 * lastFound is given, it is set to the new entry created. If the insert fails
 * then lastFound is set to the existing key of the same value.
 *
 * \returns The new element created upon success, null upon failure.
 */

/**
 * \fn BstSet::insertMulti(const Key &key)
 * \brief Insert the given key even if it exists already.
 *
 * If the key exists already then it is placed next to some other key of the
 * same value. InsertMulti cannot fail. The key's copy constructor is used to
 * place the item in the table.
 *
 * \returns The new element created.
 */

#endif /* _AAPL_BSTSET_H */
