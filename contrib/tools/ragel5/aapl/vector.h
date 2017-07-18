/*
 *  Copyright 2002, 2006 Adrian Thurston <thurston@cs.queensu.ca>
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

#ifndef _AAPL_VECTOR_H
#define _AAPL_VECTOR_H

#include <new>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "table.h"

#ifdef AAPL_NAMESPACE
namespace Aapl {
#endif

/**
 * \addtogroup vector
 * @{
 */

/** \class Vector
 * \brief Dynamic array.
 *
 * This is typical vector implementation. It is a dynamic array that can be
 * used to contain complex data structures that have constructors and
 * destructors as well as simple types such as integers and pointers.
 *
 * Vector supports inserting, overwriting, and removing single or multiple
 * elements at once. Constructors and destructors are called wherever
 * appropriate.  For example, before an element is overwritten, it's
 * destructor is called.
 *
 * Vector provides automatic resizing of allocated memory as needed and offers
 * different allocation schemes for controlling how the automatic allocation
 * is done.  Two senses of the the length of the data is maintained: the
 * amount of raw memory allocated to the vector and the number of actual
 * elements in the vector. The various allocation schemes control how the
 * allocated space is changed in relation to the number of elements in the
 * vector.
 *
 * \include ex_vector.cpp
 */

/*@}*/

template < class T, class Resize = ResizeExpn > class Vector
	: public Table<T>, public Resize
{
private:
	typedef Table<T> BaseTable;

public:
	/**
	 * \brief Initialize an empty vector with no space allocated.  
	 *
	 * If a linear resizer is used, the step defaults to 256 units of T. For a
	 * runtime vector both up and down allocation schemes default to
	 * Exponential.
	 */
	Vector() { }

	/**
	 * \brief Create a vector that contains an initial element.
	 *
	 * The vector becomes one element in length. The element's copy
	 * constructor is used to place the value in the vector.
	 */
	Vector(const T &val)             { setAs(&val, 1); }

	/**
	 * \brief Create a vector that contains an array of elements.
	 *
	 * The vector becomes len elements in length.  Copy constructors are used
	 * to place the new elements in the vector. 
	 */
	Vector(const T *val, long len)   { setAs(val, len); }

	/* Deep copy. */
	Vector( const Vector &v );

	/* Free all mem used by the vector. */
	~Vector() { empty(); }

	/* Delete all items. */
	void empty();

	/* Abandon the contents of the vector without deleteing. */
	void abandon();

	/* Transfers the elements of another vector into this vector. First emptys
	 * the current vector. */
	void transfer( Vector &v );

	/* Perform a deep copy of another vector into this vector. */
	Vector &operator=( const Vector &v );


	/*@{*/
	/**
	 * \brief Insert one element at position pos.
	 *
	 * Elements in the vector from pos onward are shifted one space to the
	 * right. The copy constructor is used to place the element into this
	 * vector. If pos is greater than the length of the vector then undefined
	 * behaviour results. If pos is negative then it is treated as an offset
	 * relative to the length of the vector.
	 */
	void insert(long pos, const T &val)    { insert(pos, &val, 1); }

	/* Insert an array of values. */
	void insert(long pos, const T *val, long len);

	/**
	 * \brief Insert all the elements from another vector at position pos.
	 *
	 * Elements in this vector from pos onward are shifted v.tabLen spaces to
	 * the right. The element's copy constructor is used to copy the items
	 * into this vector. The other vector is left unchanged. If pos is off the
	 * end of the vector, then undefined behaviour results. If pos is negative
	 * then it is treated as an offset relative to the length of the vector.
	 * Equivalent to vector.insert(pos, other.data, other.tabLen).
	 */
	void insert(long pos, const Vector &v) { insert(pos, v.data, v.tabLen); }

	/* Insert len copies of val into the vector. */
	void insertDup(long pos, const T &val, long len);

	/**
	 * \brief Insert one new element using the default constrcutor.
	 *
	 * Elements in the vector from pos onward are shifted one space to the
	 * right.  The default constructor is used to init the new element. If pos
	 * is greater than the length of the vector then undefined behaviour
	 * results. If pos is negative then it is treated as an offset relative to
	 * the length of the vector.
	 */
	void insertNew(long pos)               { insertNew(pos, 1); }

	/* Insert len new items using default constructor. */
	void insertNew(long pos, long len);
	/*@}*/

	/*@{*/
	/**
	 * \brief Remove one element at position pos.
	 *
	 * The element's destructor is called. Elements to the right of pos are
	 * shifted one space to the left to take up the free space. If pos is greater
	 * than or equal to the length of the vector then undefined behavior results.
	 * If pos is negative then it is treated as an offset relative to the length
	 * of the vector.
	 */
	void remove(long pos)                 { remove(pos, 1); }

	/* Delete a number of elements. */
	void remove(long pos, long len);
	/*@}*/

	/*@{*/
	/**
	 * \brief Replace one element at position pos.
	 *
	 * If there is an existing element at position pos (if pos is less than
	 * the length of the vector) then its destructor is called before the
	 * space is used. The copy constructor is used to place the element into
	 * the vector.  If pos is greater than the length of the vector then
	 * undefined behaviour results.  If pos is negative then it is treated as
	 * an offset relative to the length of the vector.
	 */
	void replace(long pos, const T &val)    { replace(pos, &val, 1); }

	/* Replace with an array of values. */
	void replace(long pos, const T *val, long len);

	/**
	 * \brief Replace at position pos with all the elements of another vector.
	 *
	 * Replace at position pos with all the elements of another vector. The
	 * other vector is left unchanged. If there are existing elements at the
	 * positions to be replaced, then destructors are called before the space
	 * is used. Copy constructors are used to place the elements into this
	 * vector. It is allowable for the pos and length of the other vector to
	 * specify a replacement that overwrites existing elements and creates new
	 * ones.  If pos is greater than the length of the vector then undefined
	 * behaviour results.  If pos is negative, then it is treated as an offset
	 * relative to the length of the vector.
	 */
	void replace(long pos, const Vector &v) { replace(pos, v.data, v.tabLen); }

	/* Replace len items with len copies of val. */
	void replaceDup(long pos, const T &val, long len);

	/**
	 * \brief Replace at position pos with one new element.
	 *
	 * If there is an existing element at the position to be replaced (pos is
	 * less than the length of the vector) then the element's destructor is
	 * called before the space is used. The default constructor is used to
	 * initialize the new element. If pos is greater than the length of the
	 * vector then undefined behaviour results. If pos is negative, then it is
	 * treated as an offset relative to the length of the vector.
	 */
	void replaceNew(long pos)               { replaceNew(pos, 1); }

	/* Replace len items at pos with newly constructed objects. */
	void replaceNew(long pos, long len);
	/*@}*/

	/*@{*/
	/**
	 * \brief Set the contents of the vector to be val exactly.
	 *
	 * The vector becomes one element in length. Destructors are called on any
	 * existing elements in the vector. The element's copy constructor is used
	 * to place the val in the vector.
	 */
	void setAs(const T &val)             { setAs(&val, 1); }

	/* Set to the contents of an array. */
	void setAs(const T *val, long len);

	/**
	 * \brief Set the vector to exactly the contents of another vector.
	 *
	 * The vector becomes v.tabLen elements in length. Destructors are called
	 * on any existing elements. Copy constructors are used to place the new
	 * elements in the vector.
	 */
	void setAs(const Vector &v)          { setAs(v.data, v.tabLen); }

	/* Set as len copies of item. */
	void setAsDup(const T &item, long len);

	/**
	 * \brief Set the vector to exactly one new item.
	 *
	 * The vector becomes one element in length. Destructors are called on any
	 * existing elements in the vector. The default constructor is used to
	 * init the new item.
	 */
	void setAsNew()                      { setAsNew(1); }

	/* Set as newly constructed objects using the default constructor. */
	void setAsNew(long len);
	/*@}*/

	/*@{*/
	/** 
	 * \brief Append one elment to the end of the vector.
	 *
	 * Copy constructor is used to place the element in the vector.
	 */
	void append(const T &val)                { replace(BaseTable::tabLen, &val, 1); }

	/**
	 * \brief Append len elements to the end of the vector. 
	 *
	 * Copy constructors are used to place the elements in the vector. 
	 */
	void append(const T *val, long len)       { replace(BaseTable::tabLen, val, len); }

	/**
	 * \brief Append the contents of another vector.
	 *
	 * The other vector is left unchanged. Copy constructors are used to place the
	 * elements in the vector.
	 */
	void append(const Vector &v)             { replace(BaseTable::tabLen, v.data, v.tabLen); }

	/**
	 * \brief Append len copies of item.
	 *
	 * The copy constructor is used to place the item in the vector.
	 */
	void appendDup(const T &item, long len)   { replaceDup(BaseTable::tabLen, item, len); }

	/**
	 * \brief Append a single newly created item. 
	 *
	 * The new element is initialized with the default constructor.
	 */
	void appendNew()                         { replaceNew(BaseTable::tabLen, 1); }

	/**
	 * \brief Append len newly created items.
	 *
	 * The new elements are initialized with the default constructor.
	 */
	void appendNew(long len)                  { replaceNew(BaseTable::tabLen, len); }
	/*@}*/
	
	/*@{*/
	/** \fn Vector::prepend(const T &val)
	 * \brief Prepend one elment to the front of the vector.
	 *
	 * Copy constructor is used to place the element in the vector.
	 */
	void prepend(const T &val)               { insert(0, &val, 1); }

	/**
	 * \brief Prepend len elements to the front of the vector. 
	 *
	 * Copy constructors are used to place the elements in the vector. 
	 */
	void prepend(const T *val, long len)      { insert(0, val, len); }

	/**
	 * \brief Prepend the contents of another vector.
	 *
	 * The other vector is left unchanged. Copy constructors are used to place the
	 * elements in the vector.
	 */
	void prepend(const Vector &v)            { insert(0, v.data, v.tabLen); }

	/**
	 * \brief Prepend len copies of item.
	 *
	 * The copy constructor is used to place the item in the vector.
	 */
	void prependDup(const T &item, long len)  { insertDup(0, item, len); }

	/**
	 * \brief Prepend a single newly created item. 
	 *
	 * The new element is initialized with the default constructor.
	 */
	void prependNew()                        { insertNew(0, 1); }

	/**
	 * \brief Prepend len newly created items.
	 *
	 * The new elements are initialized with the default constructor.
	 */
	void prependNew(long len)                 { insertNew(0, len); }
	/*@}*/

	/* Convenience access. */
	T &operator[](int i) const { return BaseTable::data[i]; }
	long size() const           { return BaseTable::tabLen; }

	/* Forward this so a ref can be used. */
	struct Iter;

	/* Various classes for setting the iterator */
	struct IterFirst { IterFirst( const Vector &v ) : v(v) { } const Vector &v; };
	struct IterLast { IterLast( const Vector &v ) : v(v) { } const Vector &v; };
	struct IterNext { IterNext( const Iter &i ) : i(i) { } const Iter &i; };
	struct IterPrev { IterPrev( const Iter &i ) : i(i) { } const Iter &i; };

	/** 
	 * \brief Vector Iterator.
	 * \ingroup iterators
	 */
	struct Iter
	{
		/* Construct, assign. */
		Iter() : ptr(0), ptrBeg(0), ptrEnd(0) { }

		/* Construct. */
		Iter( const Vector &v );
		Iter( const IterFirst &vf );
		Iter( const IterLast &vl );
		inline Iter( const IterNext &vn );
		inline Iter( const IterPrev &vp );

		/* Assign. */
		Iter &operator=( const Vector &v );
		Iter &operator=( const IterFirst &vf );
		Iter &operator=( const IterLast &vl );
		inline Iter &operator=( const IterNext &vf );
		inline Iter &operator=( const IterPrev &vl );

		/** \brief Less than end? */
		bool lte() const { return ptr != ptrEnd; }

		/** \brief At end? */
		bool end() const { return ptr == ptrEnd; }

		/** \brief Greater than beginning? */
		bool gtb() const { return ptr != ptrBeg; }

		/** \brief At beginning? */
		bool beg() const { return ptr == ptrBeg; }

		/** \brief At first element? */
		bool first() const { return ptr == ptrBeg+1; }

		/** \brief At last element? */
		bool last() const { return ptr == ptrEnd-1; }

		/* Return the position. */
		long pos() const { return ptr - ptrBeg - 1; }
		T &operator[](int i) const { return ptr[i]; }

		/** \brief Implicit cast to T*. */
		operator T*() const   { return ptr; }

		/** \brief Dereference operator returns T&. */
		T &operator *() const { return *ptr; }

		/** \brief Arrow operator returns T*. */
		T *operator->() const { return ptr; }

		/** \brief Move to next item. */
		T *operator++()       { return ++ptr; }

		/** \brief Move to next item. */
		T *operator++(int)    { return ptr++; }

		/** \brief Move to next item. */
		T *increment()        { return ++ptr; }

		/** \brief Move n items forward. */
		T *operator+=(long n)       { return ptr+=n; }

		/** \brief Move to previous item. */
		T *operator--()       { return --ptr; }

		/** \brief Move to previous item. */
		T *operator--(int)    { return ptr--; }

		/** \brief Move to previous item. */
		T *decrement()        { return --ptr; }
		
		/** \brief Move n items back. */
		T *operator-=(long n)       { return ptr-=n; }

		/** \brief Return the next item. Does not modify this. */
		inline IterNext next() const { return IterNext(*this); }

		/** \brief Return the previous item. Does not modify this. */
		inline IterPrev prev() const { return IterPrev(*this); }

		/** \brief The iterator is simply a pointer. */
		T *ptr;

		/* For testing endpoints. */
		T *ptrBeg, *ptrEnd;
	};

	/** \brief Return first element. */
	IterFirst first() { return IterFirst( *this ); }

	/** \brief Return last element. */
	IterLast last() { return IterLast( *this ); }

protected:
 	void makeRawSpaceFor(long pos, long len);

	void upResize(long len);
	void downResize(long len);
};

/* Init a vector iterator with just a vector. */
template <class T, class Resize> Vector<T, Resize>::Iter::Iter( const Vector &v ) 
{
	if ( v.tabLen == 0 )
		ptr = ptrBeg = ptrEnd = 0;
	else {
		ptr = v.data;
		ptrBeg = v.data-1;
		ptrEnd = v.data+v.tabLen;
	}
}

/* Init a vector iterator with the first of a vector. */
template <class T, class Resize> Vector<T, Resize>::Iter::Iter( 
		const IterFirst &vf ) 
{
	if ( vf.v.tabLen == 0 )
		ptr = ptrBeg = ptrEnd = 0;
	else {
		ptr = vf.v.data;
		ptrBeg = vf.v.data-1;
		ptrEnd = vf.v.data+vf.v.tabLen;
	}
}

/* Init a vector iterator with the last of a vector. */
template <class T, class Resize> Vector<T, Resize>::Iter::Iter( 
		const IterLast &vl ) 
{
	if ( vl.v.tabLen == 0 )
		ptr = ptrBeg = ptrEnd = 0;
	else {
		ptr = vl.v.data+vl.v.tabLen-1;
		ptrBeg = vl.v.data-1;
		ptrEnd = vl.v.data+vl.v.tabLen;
	}
}

/* Init a vector iterator with the next of some other iterator. */
template <class T, class Resize> Vector<T, Resize>::Iter::Iter( 
		const IterNext &vn ) 
:
	ptr(vn.i.ptr+1), 
	ptrBeg(vn.i.ptrBeg),
	ptrEnd(vn.i.ptrEnd)
{
}

/* Init a vector iterator with the prev of some other iterator. */
template <class T, class Resize> Vector<T, Resize>::Iter::Iter( 
		const IterPrev &vp ) 
:
	ptr(vp.i.ptr-1),
	ptrBeg(vp.i.ptrBeg),
	ptrEnd(vp.i.ptrEnd)
{
}

/* Set a vector iterator with some vector. */
template <class T, class Resize> typename Vector<T, Resize>::Iter &
		Vector<T, Resize>::Iter::operator=( const Vector &v )    
{
	if ( v.tabLen == 0 )
		ptr = ptrBeg = ptrEnd = 0;
	else {
		ptr = v.data; 
		ptrBeg = v.data-1; 
		ptrEnd = v.data+v.tabLen; 
	}
	return *this;
}

/* Set a vector iterator with the first element in a vector. */
template <class T, class Resize> typename Vector<T, Resize>::Iter &
		Vector<T, Resize>::Iter::operator=( const IterFirst &vf )    
{
	if ( vf.v.tabLen == 0 )
		ptr = ptrBeg = ptrEnd = 0;
	else {
		ptr = vf.v.data; 
		ptrBeg = vf.v.data-1; 
		ptrEnd = vf.v.data+vf.v.tabLen; 
	}
	return *this;
}

/* Set a vector iterator with the last element in a vector. */
template <class T, class Resize> typename Vector<T, Resize>::Iter &
		Vector<T, Resize>::Iter::operator=( const IterLast &vl )    
{
	if ( vl.v.tabLen == 0 )
		ptr = ptrBeg = ptrEnd = 0;
	else {
		ptr = vl.v.data+vl.v.tabLen-1; 
		ptrBeg = vl.v.data-1; 
		ptrEnd = vl.v.data+vl.v.tabLen; 
	}
	return *this;
}

/* Set a vector iterator with the next of some other iterator. */
template <class T, class Resize> typename Vector<T, Resize>::Iter &
		Vector<T, Resize>::Iter::operator=( const IterNext &vn )    
{
	ptr = vn.i.ptr+1; 
	ptrBeg = vn.i.ptrBeg;
	ptrEnd = vn.i.ptrEnd;
	return *this;
}

/* Set a vector iterator with the prev of some other iterator. */
template <class T, class Resize> typename Vector<T, Resize>::Iter &
		Vector<T, Resize>::Iter::operator=( const IterPrev &vp )    
{
	ptr = vp.i.ptr-1; 
	ptrBeg = vp.i.ptrBeg;
	ptrEnd = vp.i.ptrEnd;
	return *this;
}

/**
 * \brief Forget all elements in the vector.
 *
 * The contents of the vector are reset to null without without the space
 * being freed.
 */
template<class T, class Resize> void Vector<T, Resize>::
		abandon()
{
	BaseTable::data = 0;
	BaseTable::tabLen = 0;
	BaseTable::allocLen = 0;
}

/**
 * \brief Transfer the contents of another vector into this vector.
 *
 * The dynamic array of the other vector is moved into this vector by
 * reference. If this vector is non-empty then its contents are first deleted.
 * Afterward the other vector will be empty.
 */
template<class T, class Resize> void Vector<T, Resize>::
		transfer( Vector &v )
{
	empty();

	BaseTable::data = v.data;
	BaseTable::tabLen = v.tabLen;
	BaseTable::allocLen = v.allocLen;

	v.abandon();
}

/**
 * \brief Deep copy another vector into this vector.
 *
 * Copies the entire contents of the other vector into this vector. Any
 * existing contents are first deleted. Equivalent to setAs.
 *
 * \returns A reference to this.
 */
template<class T, class Resize> Vector<T, Resize> &Vector<T, Resize>::
		operator=( const Vector &v )
{
	setAs(v.data, v.tabLen); 
	return *this;
}

/* Up resize the data for len elements using Resize::upResize to tell us the
 * new tabLen. Reads and writes allocLen. Does not read or write tabLen. */
template<class T, class Resize> void Vector<T, Resize>::
		upResize(long len)
{
	/* Ask the resizer what the new tabLen will be. */
	long newLen = Resize::upResize(BaseTable::allocLen, len);

	/* Did the data grow? */
	if ( newLen > BaseTable::allocLen ) {
		BaseTable::allocLen = newLen;
		if ( BaseTable::data != 0 ) {
			/* Table exists already, resize it up. */
			BaseTable::data = (T*) realloc( BaseTable::data, sizeof(T) * newLen );
			if ( BaseTable::data == 0 )
				throw std::bad_alloc();
		}
		else {
			/* Create the data. */
			BaseTable::data = (T*) malloc( sizeof(T) * newLen );
			if ( BaseTable::data == 0 )
				throw std::bad_alloc();
		}
	}
}

/* Down resize the data for len elements using Resize::downResize to determine
 * the new tabLen. Reads and writes allocLen. Does not read or write tabLen. */
template<class T, class Resize> void Vector<T, Resize>::
		downResize(long len)
{
	/* Ask the resizer what the new tabLen will be. */
	long newLen = Resize::downResize( BaseTable::allocLen, len );

	/* Did the data shrink? */
	if ( newLen < BaseTable::allocLen ) {
		BaseTable::allocLen = newLen;
		if ( newLen == 0 ) {
			/* Simply free the data. */
			free( BaseTable::data );
			BaseTable::data = 0;
		}
		else {
			/* Not shrinking to size zero, realloc it to the smaller size. */
			BaseTable::data = (T*) realloc( BaseTable::data, sizeof(T) * newLen );
			if ( BaseTable::data == 0 )
				throw std::bad_alloc();
		}
	}
}

/**
 * \brief Perform a deep copy of the vector.
 *
 * The contents of the other vector are copied into this vector. This vector
 * gets the same allocation size as the other vector. All items are copied
 * using the element's copy constructor.
 */
template<class T, class Resize> Vector<T, Resize>::
		Vector(const Vector<T, Resize> &v)
{
	BaseTable::tabLen = v.tabLen;
	BaseTable::allocLen = v.allocLen;

	if ( BaseTable::allocLen > 0 ) {
		/* Allocate needed space. */
		BaseTable::data = (T*) malloc(sizeof(T) * BaseTable::allocLen);
		if ( BaseTable::data == 0 )
			throw std::bad_alloc();

		/* If there are any items in the src data, copy them in. */
		T *dst = BaseTable::data, *src = v.data;
		for (long pos = 0; pos < BaseTable::tabLen; pos++, dst++, src++ )
			new(dst) T(*src);
	}
	else {
		/* Nothing allocated. */
		BaseTable::data = 0;
	}
}

/** \fn Vector::~Vector()
 * \brief Free all memory used by the vector. 
 *
 * The vector is reset to zero elements. Destructors are called on all
 * elements in the vector. The space allocated for the vector is freed.
 */


/**
 * \brief Free all memory used by the vector. 
 *
 * The vector is reset to zero elements. Destructors are called on all
 * elements in the vector. The space allocated for the vector is freed.
 */
template<class T, class Resize> void Vector<T, Resize>::
		empty()
{
	if ( BaseTable::data != 0 ) {
		/* Call All destructors. */
		T *pos = BaseTable::data;
		for ( long i = 0; i < BaseTable::tabLen; pos++, i++ )
			pos->~T();

		/* Free the data space. */
		free( BaseTable::data );
		BaseTable::data = 0;
		BaseTable::tabLen = BaseTable::allocLen = 0;
	}
}

/**
 * \brief Set the contents of the vector to be len elements exactly. 
 *
 * The vector becomes len elements in length. Destructors are called on any
 * existing elements in the vector. Copy constructors are used to place the
 * new elements in the vector. 
 */
template<class T, class Resize> void Vector<T, Resize>::
		setAs(const T *val, long len)
{
	/* Call All destructors. */
	long i;
	T *pos = BaseTable::data;
	for ( i = 0; i < BaseTable::tabLen; pos++, i++ )
		pos->~T();

	/* Adjust the allocated length. */
	if ( len < BaseTable::tabLen )
		downResize( len );
	else if ( len > BaseTable::tabLen )
		upResize( len );

	/* Set the new data length to exactly len. */
	BaseTable::tabLen = len;	
	
	/* Copy data in. */
	T *dst = BaseTable::data;
	const T *src = val;
	for ( i = 0; i < len; i++, dst++, src++ )
		new(dst) T(*src);
}

/**
 * \brief Set the vector to len copies of item.
 *
 * The vector becomes len elements in length. Destructors are called on any
 * existing elements in the vector. The element's copy constructor is used to
 * copy the item into the vector.
 */
template<class T, class Resize> void Vector<T, Resize>::
		setAsDup(const T &item, long len)
{
	/* Call All destructors. */
	T *pos = BaseTable::data;
	for ( long i = 0; i < BaseTable::tabLen; pos++, i++ )
		pos->~T();

	/* Adjust the allocated length. */
	if ( len < BaseTable::tabLen )
		downResize( len );
	else if ( len > BaseTable::tabLen )
		upResize( len );

	/* Set the new data length to exactly len. */
	BaseTable::tabLen = len;	
	
	/* Copy item in one spot at a time. */
	T *dst = BaseTable::data;
	for ( long i = 0; i < len; i++, dst++ )
		new(dst) T(item);
}

/**
 * \brief Set the vector to exactly len new items.
 *
 * The vector becomes len elements in length. Destructors are called on any
 * existing elements in the vector. Default constructors are used to init the
 * new items.
 */
template<class T, class Resize> void Vector<T, Resize>::
		setAsNew(long len)
{
	/* Call All destructors. */
	T *pos = BaseTable::data;
	for ( long i = 0; i < BaseTable::tabLen; pos++, i++ )
		pos->~T();

	/* Adjust the allocated length. */
	if ( len < BaseTable::tabLen )
		downResize( len );
	else if ( len > BaseTable::tabLen )
		upResize( len );

	/* Set the new data length to exactly len. */
	BaseTable::tabLen = len;	
	
	/* Create items using default constructor. */
	T *dst = BaseTable::data;
	for ( long i = 0; i < len; i++, dst++ )
		new(dst) T();
}


/**
 * \brief Replace len elements at position pos.
 *
 * If there are existing elements at the positions to be replaced, then
 * destructors are called before the space is used. Copy constructors are used
 * to place the elements into the vector. It is allowable for the pos and
 * length to specify a replacement that overwrites existing elements and
 * creates new ones.  If pos is greater than the length of the vector then
 * undefined behaviour results. If pos is negative, then it is treated as an
 * offset relative to the length of the vector.
 */
template<class T, class Resize> void Vector<T, Resize>::
		replace(long pos, const T *val, long len)
{
	long endPos, i;
	T *item;

	/* If we are given a negative position to replace at then
	 * treat it as a position relative to the length. */
	if ( pos < 0 )
		pos = BaseTable::tabLen + pos;

	/* The end is the one past the last item that we want
	 * to write to. */
	endPos = pos + len;

	/* Make sure we have enough space. */
	if ( endPos > BaseTable::tabLen ) {
		upResize( endPos );

		/* Delete any objects we need to delete. */
		item = BaseTable::data + pos;
		for ( i = pos; i < BaseTable::tabLen; i++, item++ )
			item->~T();
		
		/* We are extending the vector, set the new data length. */
		BaseTable::tabLen = endPos;
	}
	else {
		/* Delete any objects we need to delete. */
		item = BaseTable::data + pos;
		for ( i = pos; i < endPos; i++, item++ )
			item->~T();
	}

	/* Copy data in using copy constructor. */
	T *dst = BaseTable::data + pos;
	const T *src = val;
	for ( i = 0; i < len; i++, dst++, src++ )
		new(dst) T(*src);
}

/**
 * \brief Replace at position pos with len copies of an item.
 *
 * If there are existing elements at the positions to be replaced, then
 * destructors are called before the space is used. The copy constructor is
 * used to place the element into this vector. It is allowable for the pos and
 * length to specify a replacement that overwrites existing elements and
 * creates new ones. If pos is greater than the length of the vector then
 * undefined behaviour results.  If pos is negative, then it is treated as an
 * offset relative to the length of the vector.
 */
template<class T, class Resize> void Vector<T, Resize>::
		replaceDup(long pos, const T &val, long len)
{
	long endPos, i;
	T *item;

	/* If we are given a negative position to replace at then
	 * treat it as a position relative to the length. */
	if ( pos < 0 )
		pos = BaseTable::tabLen + pos;

	/* The end is the one past the last item that we want
	 * to write to. */
	endPos = pos + len;

	/* Make sure we have enough space. */
	if ( endPos > BaseTable::tabLen ) {
		upResize( endPos );

		/* Delete any objects we need to delete. */
		item = BaseTable::data + pos;
		for ( i = pos; i < BaseTable::tabLen; i++, item++ )
			item->~T();
		
		/* We are extending the vector, set the new data length. */
		BaseTable::tabLen = endPos;
	}
	else {
		/* Delete any objects we need to delete. */
		item = BaseTable::data + pos;
		for ( i = pos; i < endPos; i++, item++ )
			item->~T();
	}

	/* Copy data in using copy constructor. */
	T *dst = BaseTable::data + pos;
	for ( long i = 0; i < len; i++, dst++ )
		new(dst) T(val);
}

/**
 * \brief Replace at position pos with len new elements.
 *
 * If there are existing elements at the positions to be replaced, then
 * destructors are called before the space is used. The default constructor is
 * used to initialize the new elements. It is allowable for the pos and length
 * to specify a replacement that overwrites existing elements and creates new
 * ones. If pos is greater than the length of the vector then undefined
 * behaviour results. If pos is negative, then it is treated as an offset
 * relative to the length of the vector.
 */
template<class T, class Resize> void Vector<T, Resize>::
		replaceNew(long pos, long len)
{
	long endPos, i;
	T *item;

	/* If we are given a negative position to replace at then
	 * treat it as a position relative to the length. */
	if ( pos < 0 )
		pos = BaseTable::tabLen + pos;

	/* The end is the one past the last item that we want
	 * to write to. */
	endPos = pos + len;

	/* Make sure we have enough space. */
	if ( endPos > BaseTable::tabLen ) {
		upResize( endPos );

		/* Delete any objects we need to delete. */
		item = BaseTable::data + pos;
		for ( i = pos; i < BaseTable::tabLen; i++, item++ )
			item->~T();
		
		/* We are extending the vector, set the new data length. */
		BaseTable::tabLen = endPos;
	}
	else {
		/* Delete any objects we need to delete. */
		item = BaseTable::data + pos;
		for ( i = pos; i < endPos; i++, item++ )
			item->~T();
	}

	/* Copy data in using copy constructor. */
	T *dst = BaseTable::data + pos;
	for ( long i = 0; i < len; i++, dst++ )
		new(dst) T();
}

/**
 * \brief Remove len elements at position pos.
 *
 * Destructor is called on all elements removed. Elements to the right of pos
 * are shifted len spaces to the left to take up the free space. If pos is
 * greater than or equal to the length of the vector then undefined behavior
 * results. If pos is negative then it is treated as an offset relative to the
 * length of the vector.
 */
template<class T, class Resize> void Vector<T, Resize>::
		remove(long pos, long len)
{
	long newLen, lenToSlideOver, endPos;
	T *dst, *item;

	/* If we are given a negative position to remove at then
	 * treat it as a position relative to the length. */
	if ( pos < 0 )
		pos = BaseTable::tabLen + pos;

	/* The first position after the last item deleted. */
	endPos = pos + len;

	/* The new data length. */
	newLen = BaseTable::tabLen - len;

	/* The place in the data we are deleting at. */
	dst = BaseTable::data + pos;

	/* Call Destructors. */
	item = dst;
	for ( long i = 0; i < len; i += 1, item += 1 )
		item->~T();
	
	/* Shift data over if necessary. */
	lenToSlideOver = BaseTable::tabLen - endPos;	
	if ( len > 0 && lenToSlideOver > 0 )
		memmove(dst, dst + len, sizeof(T)*lenToSlideOver);

	/* Shrink the data if necessary. */
	downResize( newLen );

	/* Set the new data length. */
	BaseTable::tabLen = newLen;
}

/**
 * \brief Insert len elements at position pos.
 *
 * Elements in the vector from pos onward are shifted len spaces to the right.
 * The copy constructor is used to place the elements into this vector. If pos
 * is greater than the length of the vector then undefined behaviour results.
 * If pos is negative then it is treated as an offset relative to the length
 * of the vector.
 */
template<class T, class Resize> void Vector<T, Resize>::
		insert(long pos, const T *val, long len)
{
	/* If we are given a negative position to insert at then
	 * treat it as a position relative to the length. */
	if ( pos < 0 )
		pos = BaseTable::tabLen + pos;
	
	/* Calculate the new length. */
	long newLen = BaseTable::tabLen + len;

	/* Up resize, we are growing. */
	upResize( newLen );

	/* Shift over data at insert spot if needed. */
	if ( len > 0 && pos < BaseTable::tabLen ) {
		memmove(BaseTable::data + pos + len, BaseTable::data + pos,
				sizeof(T)*(BaseTable::tabLen-pos));
	}

	/* Copy data in element by element. */
	T *dst = BaseTable::data + pos;
	const T *src = val;
	for ( long i = 0; i < len; i++, dst++, src++ )
		new(dst) T(*src);

	/* Set the new length. */
	BaseTable::tabLen = newLen;
}

/**
 * \brief Insert len copies of item at position pos.
 *
 * Elements in the vector from pos onward are shifted len spaces to the right.
 * The copy constructor is used to place the element into this vector. If pos
 * is greater than the length of the vector then undefined behaviour results.
 * If pos is negative then it is treated as an offset relative to the length
 * of the vector.
 */
template<class T, class Resize> void Vector<T, Resize>::
		insertDup(long pos, const T &item, long len)
{
	/* If we are given a negative position to insert at then
	 * treat it as a position relative to the length. */
	if ( pos < 0 )
		pos = BaseTable::tabLen + pos;
	
	/* Calculate the new length. */
	long newLen = BaseTable::tabLen + len;

	/* Up resize, we are growing. */
	upResize( newLen );

	/* Shift over data at insert spot if needed. */
	if ( len > 0 && pos < BaseTable::tabLen ) {
		memmove(BaseTable::data + pos + len, BaseTable::data + pos,
				sizeof(T)*(BaseTable::tabLen-pos));
	}

	/* Copy the data item in one at a time. */
	T *dst = BaseTable::data + pos;
	for ( long i = 0; i < len; i++, dst++ )
		new(dst) T(item);

	/* Set the new length. */
	BaseTable::tabLen = newLen;
}

/**
 * \brief Insert len new elements using the default constructor.
 *
 * Elements in the vector from pos onward are shifted len spaces to the right.
 * Default constructors are used to init the new elements. If pos is off the
 * end of the vector then undefined behaviour results. If pos is negative then
 * it is treated as an offset relative to the length of the vector.
 */
template<class T, class Resize> void Vector<T, Resize>::
		insertNew(long pos, long len)
{
	/* If we are given a negative position to insert at then
	 * treat it as a position relative to the length. */
	if ( pos < 0 )
		pos = BaseTable::tabLen + pos;
	
	/* Calculate the new length. */
	long newLen = BaseTable::tabLen + len;

	/* Up resize, we are growing. */
	upResize( newLen );

	/* Shift over data at insert spot if needed. */
	if ( len > 0 && pos < BaseTable::tabLen ) {
		memmove(BaseTable::data + pos + len, BaseTable::data + pos,
				sizeof(T)*(BaseTable::tabLen-pos));
	}

	/* Init new data with default constructors. */
	T *dst = BaseTable::data + pos;
	for ( long i = 0; i < len; i++, dst++ )
		new(dst) T();

	/* Set the new length. */
	BaseTable::tabLen = newLen;
}

/* Makes space for len items, Does not init the items in any way.  If pos is
 * greater than the length of the vector then undefined behaviour results.
 * Updates the length of the vector. */
template<class T, class Resize> void Vector<T, Resize>::
		makeRawSpaceFor(long pos, long len)
{
	/* Calculate the new length. */
	long newLen = BaseTable::tabLen + len;

	/* Up resize, we are growing. */
	upResize( newLen );

	/* Shift over data at insert spot if needed. */
	if ( len > 0 && pos < BaseTable::tabLen ) {
		memmove(BaseTable::data + pos + len, BaseTable::data + pos,
			sizeof(T)*(BaseTable::tabLen-pos));
	}

	/* Save the new length. */
	BaseTable::tabLen = newLen;
}

#ifdef AAPL_NAMESPACE
}
#endif

#endif /* _AAPL_VECTOR_H */
