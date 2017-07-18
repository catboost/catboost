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

#ifndef _AAPL_BUBBLESORT_H
#define _AAPL_BUBBLESORT_H

#ifdef AAPL_NAMESPACE
namespace Aapl {
#endif

/**
 * \addtogroup sort
 * @{
 */

/**
 * \class BubbleSort
 * \brief Bubble sort an array of data.
 *
 * BubbleSort can be used to sort any array of objects of type T provided a
 * compare class is given. BubbleSort is in-place. It does not require any
 * temporary storage.
 *
 * Objects are not made aware that they are being moved around in memory.
 * Assignment operators, constructors and destructors are never invoked by the
 * sort.
 *
 * BubbleSort runs in O(n^2) time. It is most useful when sorting arrays that
 * are nearly sorted. It is best when neighbouring pairs are out of place.
 * BubbleSort is a stable sort, meaning that objects with the same key have
 * their relative ordering preserved.
 */

/*@}*/

/* BubbleSort. */
template <class T, class Compare> class BubbleSort
	: public Compare
{
public:
	/* Sorting interface routine. */
	void sort(T *data, long len);
};


/**
 * \brief Bubble sort an array of data.
 */
template <class T, class Compare> void BubbleSort<T,Compare>::
		sort(T *data, long len)
{
	bool changed = true;
	for ( long pass = 1; changed && pass < len; pass ++ ) {
		changed = false;
		for ( long i = 0; i < len-pass; i++ ) {
			/* Do we swap pos with the next one? */
			if ( this->compare( data[i], data[i+1] ) > 0 ) {
				char tmp[sizeof(T)];

				/* Swap the two items. */
				memcpy( tmp, data+i, sizeof(T) );
				memcpy( data+i, data+i+1, sizeof(T) );
				memcpy( data+i+1, tmp, sizeof(T) );

				/* Note that we made a change. */
				changed = true;
			}
		}
	}
}

#ifdef AAPL_NAMESPACE
}
#endif

#endif /* _AAPL_BUBBLESORT_H */
