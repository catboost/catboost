/*
 *  Copyright 2001 Adrian Thurston <thurston@cs.queensu.ca>
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

#ifndef _AAPL_DLISTMEL_H
#define _AAPL_DLISTMEL_H

/**
 * \addtogroup dlist
 * @{
 */

/**
 * \class DListMel
 * \brief Doubly linked list for elements that may appear in multiple lists.
 *
 * This class is similar to DList, except that the user defined list element
 * can inherit from multple DListEl classes and consequently be an element in
 * multiple lists. In other words, DListMel allows a single instance of a data
 * structure to be an element in multiple lists without the lists interfereing
 * with one another.
 *
 * For each list that an element class is to appear in, the element must have
 * unique next and previous pointers that can be unambiguously refered to with
 * some base class name. This name is given to DListMel as a template argument
 * so it can use the correct next and previous pointers in its list
 * operations.
 *
 * DListMel does not assume ownership of elements in the list. If the elements
 * are known to reside on the heap and are not contained in any other list or
 * data structure, the provided empty() routine can be used to delete all
 * elements, however the destructor will not call this routine, it will simply
 * abandon all the elements. It is up to the programmer to explicitly
 * de-allocate items when it is safe to do so.
 *
 * \include ex_dlistmel.cpp
 */

/*@}*/

#define BASE_EL(name) BaseEl::name
#define DLMEL_TEMPDEF class Element, class BaseEl
#define DLMEL_TEMPUSE Element, BaseEl
#define DList DListMel

#include "dlcommon.h"

#undef BASE_EL
#undef DLMEL_TEMPDEF
#undef DLMEL_TEMPUSE
#undef DList

#endif /* _AAPL_DLISTMEL_H */
