/*
 *  Copyright 2001, 2002 Adrian Thurston <thurston@cs.queensu.ca>
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

#ifndef _AAPL_TABLE_H
#define _AAPL_TABLE_H

#ifdef AAPL_NAMESPACE
namespace Aapl {
#endif

/**
 * \addtogroup vector
 * @{
 */

/** \class Table
 * \brief Base class for dynamic arrays.
 *
 * Table is used as the common data storage class for vectors. It does not
 * provide any methods to operate on the data and as such it is not intended
 * to be used directly. It exists so that algorithms that operatate on dynamic
 * arrays can be written without knowing about the various vector classes that
 * my exist.
 */

/*@}*/

/* Table class. */
template <class T> class Table
{
public:
	/* Default Constructor. */
	inline Table();

	/**
	 * \brief Get the length of the vector.
	 *
	 * \returns the length of the vector.
	 */
	long length() const
		{ return tabLen; }

	/**
	 * \brief Table data.
	 *
	 * The pointer to the elements in the vector. Modifying the vector may
	 * cause this pointer to change.
	 */
	T *data;

	/**
	 * \brief Table length.
	 *
	 * The number of items of type T in the table.
	 */
	long tabLen;

	/**
	 * \brief Allocated length.
	 *
	 * The number of items for which there is room in the current allocation.
	 */
	long allocLen;
};

/**
 * \brief Default constructor
 *
 * Initialize table data to empty.
 */
template <class T> inline Table<T>::Table()
:
	data(0),
	tabLen(0),
	allocLen(0)
{
}

/* Default shared table header class. */
struct STabHead
{
	/**
	 * \brief Table length.
	 *
	 * The number of items of type T in the table.
	 */
	long tabLen;

	/**
	 * \brief Allocated length.
	 *
	 * The number of items for which there is room in the current allocation.
	 */
	long allocLen;

	/**
	 * \brief Ref Count.
	 *
	 * The number of shared vectors referencing this data.
	 */
	long refCount;
};

/**
 * \addtogroup vector
 * @{
 */

/** \class STable
 * \brief Base class for implicitly shared dynamic arrays.
 *
 * STable is used as the common data storage class for shared vectors. It does
 * not provide any methods to operate on the data and as such it is not
 * intended to be used directly. It exists so that algorithms that operatate
 * on dynamic arrays can be written without knowing about the various shared
 * vector classes that my exist.
 */

/*@}*/

/* STable class. */
template <class T> class STable
{
public:
	/* Default Constructor. */
	inline STable();

	/**
	 * \brief Get the length of the shared vector.
	 *
	 * \returns the length of the shared vector.
	 */
	long length() const
		{ return data == 0 ? 0 : (((STabHead*)data) - 1)->tabLen; }
	
	/**
	 * \brief Get header of the shared vector.
	 *
	 * \returns the header of the shared vector.
	 */
	STabHead *header() const
		{ return data == 0 ? 0 : (((STabHead*)data) - 1); }

	/**
	 * \brief Table data.
	 *
	 * The pointer to the elements in the vector. The shared table header is
	 * located just behind the data. Modifying the vector may cause this
	 * pointer to change.
	 */
	T *data;
};

/**
 * \brief Default constructor
 *
 * Initialize shared table data to empty.
 */
template <class T> inline STable<T>::STable()
:
	data(0)
{
}

/* If needed is greater than existing, give twice needed. */
#define EXPN_UP( existing, needed ) \
		needed > existing ? (needed<<1) : existing
	
/* If needed is less than 1 quarter existing, give twice needed. */
#define EXPN_DOWN( existing, needed ) \
		needed < (existing>>2) ? (needed<<1) : existing

/**
 * \addtogroup vector
 * @{
 */

/** \class ResizeExpn
 * \brief Exponential table resizer. 
 *
 * ResizeExpn is the default table resizer. When an up resize is needed, space
 * is doubled. When a down resize is needed, space is halved.  The result is
 * that when growing the vector in a linear fashion, the number of resizes of
 * the allocated space behaves logarithmically.
 *
 * If only up resizes are done, there will never be more than 2 times the
 * needed space allocated. If down resizes are done as well, there will never
 * be more than 4 times the needed space allocated.  ResizeExpn uses this 50%
 * usage policy on up resizing and 25% usage policy on down resizing to
 * improve performance when repeatedly inserting and removing a small number
 * of elements relative to the size of the array.  This scheme guarantees that
 * repetitive inserting and removing of a small number of elements will never
 * result in repetative reallocation.
 *
 * The sizes passed to the resizer from the vectors are in units of T.
 */

/*@}*/

/* Exponential resizer. */
class ResizeExpn
{
protected:
	/**
	 * \brief Determine the new table size when up resizing.
	 *
	 * If the existing size is insufficient for the space needed then allocate
	 * twice the space needed. Otherwise use the existing size.
	 *
	 * \returns The new table size.
	 */
	static inline long upResize( long existing, long needed )
		{ return EXPN_UP( existing, needed ); }

	/**
	 * \brief Determine the new table size when down resizing.
	 *
	 * If the space needed is less than one quarter of the existing size then
	 * allocate twice the space needed. Otherwise use the exitsing size.
	 *
	 * \returns The new table size.
	 */
	static inline long downResize( long existing, long needed )
		{ return EXPN_DOWN( existing, needed ); }
};

#undef EXPN_UP
#undef EXPN_DOWN

#ifdef AAPL_NAMESPACE
}
#endif

#endif /* _AAPL_TABLE_H */
