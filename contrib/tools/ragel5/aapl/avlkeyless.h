/*
 *  Copyright 2002, 2003 Adrian Thurston <thurston@cs.queensu.ca>
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

#ifndef _AAPL_AVLKEYLESS_H
#define _AAPL_AVLKEYLESS_H

#include "compare.h"

/**
 * \addtogroup avltree 
 * @{
 */

/**
 * \class AvlKeyless
 * \brief AVL tree that has no insert/find/remove functions that take a key.
 *
 * AvlKeyless is an implementation of the AVL tree rebalancing functionality
 * only. It provides the common code for the tiny AVL tree implementations.
 */

/*@}*/

#define BASE_EL(name) name
#define AVLMEL_CLASSDEF class Element
#define AVLMEL_TEMPDEF class Element
#define AVLMEL_TEMPUSE Element
#define AvlTree AvlKeyless
#define AVL_KEYLESS

#include "avlcommon.h"

#undef BASE_EL
#undef AVLMEL_CLASSDEF
#undef AVLMEL_TEMPDEF
#undef AVLMEL_TEMPUSE
#undef AvlTree
#undef AVL_KEYLESS

#endif /* _AAPL_AVLKEYLESS_H */
