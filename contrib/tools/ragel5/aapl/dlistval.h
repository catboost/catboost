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

#ifndef _AAPL_DLISTVAL_H
#define _AAPL_DLISTVAL_H

/**
 * \addtogroup dlist
 * @{
 */

/**
 * \class DListVal
 * \brief By-value doubly linked list.
 *
 * This class is a doubly linked list that does not require a list element
 * type to be declared. The user instead gives a type that is to be stored in
 * the list element. When inserting a new data item, the value is copied into
 * a newly allocated element. This list is inteded to behave and be utilized
 * like the list template found in the STL.
 *
 * DListVal is different from the other lists in that it allocates elements
 * itself. The raw element insert interface is still exposed for convenience,
 * however, the list assumes all elements in the list are allocated on the
 * heap and are to be managed by the list. The destructor WILL delete the
 * contents of the list. If the list is ever copied in from another list, the
 * existing contents are deleted first. This is in contrast to DList and
 * DListMel, which will never delete their contents to allow for statically
 * allocated elements.
 *
 * \include ex_dlistval.cpp
 */

/*@}*/

#define BASE_EL(name) name
#define DLMEL_TEMPDEF class T
#define DLMEL_TEMPUSE T
#define DList DListVal
#define Element DListValEl<T>
#define DOUBLELIST_VALUE

#include "dlcommon.h"

#undef BASE_EL
#undef DLMEL_TEMPDEF
#undef DLMEL_TEMPUSE
#undef DList
#undef Element
#undef DOUBLELIST_VALUE

#endif /* _AAPL_DLISTVAL_H */
