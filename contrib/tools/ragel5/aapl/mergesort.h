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

#ifndef _AAPL_MERGESORT_H
#define _AAPL_MERGESORT_H

#include "bubblesort.h"

#ifdef AAPL_NAMESPACE
namespace Aapl {
#endif

/**
 * \addtogroup sort
 * @{
 */

/**
 * \class MergeSort
 * \brief Merge sort an array of data.
 *
 * MergeSort can be used to sort any array of objects of type T provided a
 * compare class is given. MergeSort is not in-place, it requires temporary
 * storage equal to the size of the array. The temporary storage is allocated
 * on the heap.
 *
 * Objects are not made aware that they are being moved around in memory.
 * Assignment operators, constructors and destructors are never invoked by the
 * sort.
 *
 * MergeSort runs in worst case O(n*log(n)) time. In most cases it is slower
 * than QuickSort because more copying is neccessary. But on the other hand,
 * it is a stable sort, meaning that objects with the same key have their
 * relative ordering preserved. Also, its worst case is better. MergeSort
 * switches to a BubbleSort when the size of the array being sorted is small.
 * This happens when directly sorting a small array or when MergeSort calls
 * itself recursively on a small portion of a larger array.
 */

/*@}*/


/* MergeSort. */
template <class T, class Compare> class MergeSort
		: public BubbleSort<T, Compare>
{
public:
	/* Sorting interface routine. */
	void sort(T *data, long len);

private:
	/* Recursive worker. */
	void doSort(T *tmpStor, T *data, long len);
};

#define _MS_BUBBLE_THRESH 16

/* Recursive mergesort worker. Split data, make recursive calls, merge
 * results. */
template< class T, class Compare> void MergeSort<T,Compare>::
		doSort(T *tmpStor, T *data, long len)
{
	if ( len <= 1 )
		return;

	if ( len <= _MS_BUBBLE_THRESH ) {
		BubbleSort<T, Compare>::sort( data, len );
		return;
	}

	long mid = len / 2;

	doSort( tmpStor, data, mid );
	doSort( tmpStor + mid, data + mid, len - mid );

	/* Merge the data. */
	T *endLower = data + mid, *lower = data;
	T *endUpper = data + len, *upper = data + mid;
	T *dest = tmpStor;
	while ( true ) {
		if ( lower == endLower ) {
			/* Possibly upper left. */
			if ( upper != endUpper )
				memcpy( dest, upper, (endUpper - upper) * sizeof(T) );
			break;
		}
		else if ( upper == endUpper ) {
			/* Only lower left. */
			if ( lower != endLower )
				memcpy( dest, lower, (endLower - lower) * sizeof(T) );
			break;
		}
		else {
			/* Both upper and lower left. */
			if ( this->compare(*lower, *upper) <= 0 )
				memcpy( dest++, lower++, sizeof(T) );
			else
				memcpy( dest++, upper++, sizeof(T) );
		}
	}

	/* Copy back from the tmpStor array. */
	memcpy( data, tmpStor, sizeof( T ) * len );
}

/**
 * \brief Merge sort an array of data.
 */
template< class T, class Compare>
	void MergeSort<T,Compare>::sort(T *data, long len)
{
	/* Allocate the tmp space needed by merge sort, sort and free. */
	T *tmpStor = (T*) new char[sizeof(T) * len];
	doSort( tmpStor, data, len );
	delete[] (char*) tmpStor;
}

#ifdef AAPL_NAMESPACE
}
#endif

#endif /* _AAPL_MERGESORT_H */
