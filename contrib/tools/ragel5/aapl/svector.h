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

#ifndef _AAPL_SVECTOR_H
#define _AAPL_SVECTOR_H

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

/** \class SVector
 * \brief Copy-on-write dynamic array.
 *
 * SVector is a variant of Vector that employs copy-on-write behaviour. The
 * SVector copy constructor and = operator make shallow copies. If a vector
 * that references shared data is modified with insert, replace, append,
 * prepend, setAs or remove, a new copy is made so as not to interfere with
 * the shared data. However, shared individual elements may be modified by
 * bypassing the SVector interface.
 *
 * SVector is a dynamic array that can be used to contain complex data
 * structures that have constructors and destructors as well as simple types
 * such as integers and pointers.
 *
 * SVector supports inserting, overwriting, and removing single or multiple
 * elements at once. Constructors and destructors are called wherever
 * appropriate.  For example, before an element is overwritten, it's
 * destructor is called.
 *
 * SVector provides automatic resizing of allocated memory as needed and
 * offers different allocation schemes for controlling how the automatic
 * allocation is done.  Two senses of the the length of the data is
 * maintained: the amount of raw memory allocated to the vector and the number
 * of actual elements in the vector. The various allocation schemes control
 * how the allocated space is changed in relation to the number of elements in
 * the vector.
 */

/*@}*/

/* SVector */
template < class T, class Resize = ResizeExpn > class SVector :
	public STable<T>, public Resize
{
private:
	typedef STable<T> BaseTable;

public:
	/**
	 * \brief Initialize an empty vector with no space allocated.  
	 *
	 * If a linear resizer is used, the step defaults to 256 units of T. For a
	 * runtime vector both up and down allocation schemes default to
	 * Exponential.
	 */
	SVector() { }

	/**
	 * \brief Create a vector that contains an initial element.
	 *
	 * The vector becomes one element in length. The element's copy
	 * constructor is used to place the value in the vector.
	 */
	SVector(const T &val)             { setAs(&val, 1); }

	/**
	 * \brief Create a vector that contains an array of elements.
	 *
	 * The vector becomes len elements in length.  Copy constructors are used
	 * to place the new elements in the vector. 
	 */
	SVector(const T *val, long len)   { setAs(val, len); }

	/* Shallow copy. */
	SVector( const SVector &v );

	/**
	 * \brief Free all memory used by the vector. 
	 *
	 * The vector is reset to zero elements. Destructors are called on all
	 * elements in the vector. The space allocated for the vector is freed.
	 */
	~SVector() { empty(); }

	/* Delete all items. */
	void empty();

	/**
	 * \brief Deep copy another vector into this vector.
	 *
	 * Copies the entire contents of the other vector into this vector. Any
	 * existing contents are first deleted. Equivalent to setAs.
	 */
	void deepCopy( const SVector &v )     { setAs(v.data, v.length()); }

	/* Perform a shallow copy of another vector. */
	SVector &operator=( const SVector &v );


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
	void insert(long pos, const T &val)     { insert(pos, &val, 1); }

	/* Insert an array of values. */
	void insert(long pos, const T *val, long len);

	/**
	 * \brief Insert all the elements from another vector at position pos.
	 *
	 * Elements in this vector from pos onward are shifted v.length() spaces
	 * to the right. The element's copy constructor is used to copy the items
	 * into this vector. The other vector is left unchanged. If pos is off the
	 * end of the vector, then undefined behaviour results. If pos is negative
	 * then it is treated as an offset relative to the length of the vector.
	 * Equivalent to vector.insert(pos, other.data, other.length()).
	 */
	void insert(long pos, const SVector &v) { insert(pos, v.data, v.length()); }

	/* Insert len copies of val into the vector. */
	void insertDup(long pos, const T &val, long len);

	/**
	 * \brief Insert one new element using the default constrcutor.
	 *
	 * Elements in the vector from pos onward are shifted one space to the right.
	 * The default constructor is used to init the new element. If pos is greater
	 * than the length of the vector then undefined behaviour results. If pos is
	 * negative then it is treated as an offset relative to the length of the
	 * vector.
	 */
	void insertNew(long pos)                { insertNew(pos, 1); }

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
	void remove(long pos)                   { remove(pos, 1); }

	/* Delete a number of elements. */
	void remove(long pos, long len);
	/*@}*/

	/*@{*/
	/**
	 * \brief Replace one element at position pos.
	 *
	 * If there is an existing element at position pos (if pos is less than the
	 * length of the vector) then its destructor is called before the space is
	 * used. The copy constructor is used to place the element into the vector.
	 * If pos is greater than the length of the vector then undefined behaviour
	 * results.  If pos is negative then it is treated as an offset relative to
	 * the length of the vector.
	 */
	void replace(long pos, const T &val)     { replace(pos, &val, 1); }

	/* Replace with an array of values. */
	void replace(long pos, const T *val, long len);

	/**
	 * \brief Replace at position pos with all the elements of another vector.
	 *
	 * Replace at position pos with all the elements of another vector. The other
	 * vector is left unchanged. If there are existing elements at the positions
	 * to be replaced, then destructors are called before the space is used. Copy
	 * constructors are used to place the elements into this vector. It is
	 * allowable for the pos and length of the other vector to specify a
	 * replacement that overwrites existing elements and creates new ones.  If pos
	 * is greater than the length of the vector then undefined behaviour results.
	 * If pos is negative, then it is treated as an offset relative to the length
	 * of the vector.
	 */
	void replace(long pos, const SVector &v) { replace(pos, v.data, v.length()); }

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
	void replaceNew(long pos)                { replaceNew(pos, 1); }

	/* Replace len items at pos with newly constructed objects. */
	void replaceNew(long pos, long len);
	/*@}*/

	/*@{*/

	/**
	 * \brief Set the contents of the vector to be val exactly.
	 *
	 * The vector becomes one element in length. Destructors are called on any
	 * existing elements in the vector. The element's copy constructor is used to
	 * place the val in the vector.
	 */
	void setAs(const T &val)             { setAs(&val, 1); }

	/* Set to the contents of an array. */
	void setAs(const T *val, long len);

	/**
	 * \brief Set the vector to exactly the contents of another vector.
	 *
	 * The vector becomes v.length() elements in length. Destructors are called
	 * on any existing elements. Copy constructors are used to place the new
	 * elements in the vector.
	 */
	void setAs(const SVector &v)         { setAs(v.data, v.length()); }

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
	void append(const T &val)                { replace(BaseTable::length(), &val, 1); }

	/**
	 * \brief Append len elements to the end of the vector. 
	 *
	 * Copy constructors are used to place the elements in the vector. 
	 */
	void append(const T *val, long len)       { replace(BaseTable::length(), val, len); }

	/**
	 * \brief Append the contents of another vector.
	 *
	 * The other vector is left unchanged. Copy constructors are used to place
	 * the elements in the vector.
	 */
	void append(const SVector &v)            
			{ replace(BaseTable::length(), v.data, v.length()); }

	/**
	 * \brief Append len copies of item.
	 *
	 * The copy constructor is used to place the item in the vector.
	 */
	void appendDup(const T &item, long len)   { replaceDup(BaseTable::length(), item, len); }

	/**
	 * \brief Append a single newly created item. 
	 *
	 * The new element is initialized with the default constructor.
	 */
	void appendNew()                         { replaceNew(BaseTable::length(), 1); }

	/**
	 * \brief Append len newly created items.
	 *
	 * The new elements are initialized with the default constructor.
	 */
	void appendNew(long len)                  { replaceNew(BaseTable::length(), len); }
	/*@}*/


	/*@{*/
	/**
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
	 * The other vector is left unchanged. Copy constructors are used to place
	 * the elements in the vector.
	 */
	void prepend(const SVector &v)           { insert(0, v.data, v.length()); }

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
	long size() const           { return BaseTable::length(); }

	/* Various classes for setting the iterator */
	struct Iter;
	struct IterFirst { IterFirst( const SVector &v ) : v(v) { } const SVector &v; };
	struct IterLast { IterLast( const SVector &v ) : v(v) { } const SVector &v; };
	struct IterNext { IterNext( const Iter &i ) : i(i) { } const Iter &i; };
	struct IterPrev { IterPrev( const Iter &i ) : i(i) { } const Iter &i; };

	/** 
	 * \brief Shared Vector Iterator. 
	 * \ingroup iterators
	 */
	struct Iter
	{
		/* Construct, assign. */
		Iter() : ptr(0), ptrBeg(0), ptrEnd(0) { }

		/* Construct. */
		Iter( const SVector &v );
		Iter( const IterFirst &vf );
		Iter( const IterLast &vl );
		inline Iter( const IterNext &vn );
		inline Iter( const IterPrev &vp );

		/* Assign. */
		Iter &operator=( const SVector &v );
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

		/** \brief Move to previous item. */
		T *operator--()       { return --ptr; }

		/** \brief Move to previous item. */
		T *operator--(int)    { return ptr--; }

		/** \brief Move to previous item. */
		T *decrement()        { return --ptr; }

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

	void setAsCommon(long len);
	long replaceCommon(long pos, long len);
	long insertCommon(long pos, long len);

	void upResize(long len);
	void upResizeDup(long len);
	void upResizeFromEmpty(long len);
	void downResize(long len);
	void downResizeDup(long len);
};

/**
 * \brief Perform a shallow copy of the vector.
 *
 * Takes a reference to the contents of the other vector.
 */
template <class T, class Resize> SVector<T, Resize>::
		SVector(const SVector<T, Resize> &v)
{
	/* Take a reference to other, if any data is allocated. */
	if ( v.data == 0 )
		BaseTable::data = 0;
	else {
		/* Get the source header, up the refcount and ref it. */
		STabHead *srcHead = ((STabHead*) v.data) - 1;
		srcHead->refCount += 1;
		BaseTable::data = (T*) (srcHead + 1);
	}
}

/**
 * \brief Shallow copy another vector into this vector.
 *
 * Takes a reference to the other vector. The contents of this vector are
 * first emptied. 
 *
 * \returns A reference to this.
 */
template <class T, class Resize> SVector<T, Resize> &
		SVector<T, Resize>:: operator=( const SVector &v )
{
	/* First clean out the current contents. */
	empty();

	/* Take a reference to other, if any data is allocated. */
	if ( v.data == 0 )
		BaseTable::data = 0;
	else {
		/* Get the source header, up the refcount and ref it. */
		STabHead *srcHead = ((STabHead*) v.data) - 1;
		srcHead->refCount += 1;
		BaseTable::data = (T*) (srcHead + 1);
	}
	return *this;
}

/* Init a vector iterator with just a vector. */
template <class T, class Resize> SVector<T, Resize>::
		Iter::Iter( const SVector &v ) 
{
	long length;
	if ( v.data == 0 || (length=(((STabHead*)v.data)-1)->tabLen) == 0 )
		ptr = ptrBeg = ptrEnd = 0;
	else {
		ptr = v.data;
		ptrBeg = v.data-1;
		ptrEnd = v.data+length;
	}
}

/* Init a vector iterator with the first of a vector. */
template <class T, class Resize> SVector<T, Resize>::
		Iter::Iter( const IterFirst &vf ) 
{
	long length;
	if ( vf.v.data == 0 || (length=(((STabHead*)vf.v.data)-1)->tabLen) == 0 )
		ptr = ptrBeg = ptrEnd = 0;
	else {
		ptr = vf.v.data;
		ptrBeg = vf.v.data-1;
		ptrEnd = vf.v.data+length;
	}
}

/* Init a vector iterator with the last of a vector. */
template <class T, class Resize> SVector<T, Resize>::
		Iter::Iter( const IterLast &vl ) 
{
	long length;
	if ( vl.v.data == 0 || (length=(((STabHead*)vl.v.data)-1)->tabLen) == 0 )
		ptr = ptrBeg = ptrEnd = 0;
	else {
		ptr = vl.v.data+length-1;
		ptrBeg = vl.v.data-1;
		ptrEnd = vl.v.data+length;
	}
}

/* Init a vector iterator with the next of some other iterator. */
template <class T, class Resize> SVector<T, Resize>::
		Iter::Iter( const IterNext &vn ) 
:
	ptr(vn.i.ptr+1), 
	ptrBeg(vn.i.ptrBeg),
	ptrEnd(vn.i.ptrEnd)
{
}

/* Init a vector iterator with the prev of some other iterator. */
template <class T, class Resize> SVector<T, Resize>::
		Iter::Iter( const IterPrev &vp ) 
:
	ptr(vp.i.ptr-1),
	ptrBeg(vp.i.ptrBeg),
	ptrEnd(vp.i.ptrEnd)
{
}

/* Set a vector iterator with some vector. */
template <class T, class Resize> typename SVector<T, Resize>::Iter &
		SVector<T, Resize>::Iter::operator=( const SVector &v )    
{
	long length;
	if ( v.data == 0 || (length=(((STabHead*)v.data)-1)->tabLen) == 0 )
		ptr = ptrBeg = ptrEnd = 0;
	else {
		ptr = v.data; 
		ptrBeg = v.data-1; 
		ptrEnd = v.data+length; 
	}
	return *this;
}

/* Set a vector iterator with the first element in a vector. */
template <class T, class Resize> typename SVector<T, Resize>::Iter &
		SVector<T, Resize>::Iter::operator=( const IterFirst &vf )    
{
	long length;
	if ( vf.v.data == 0 || (length=(((STabHead*)vf.v.data)-1)->tabLen) == 0 )
		ptr = ptrBeg = ptrEnd = 0;
	else {
		ptr = vf.v.data; 
		ptrBeg = vf.v.data-1; 
		ptrEnd = vf.v.data+length; 
	}
	return *this;
}

/* Set a vector iterator with the last element in a vector. */
template <class T, class Resize> typename SVector<T, Resize>::Iter &
		SVector<T, Resize>::Iter::operator=( const IterLast &vl )    
{
	long length;
	if ( vl.v.data == 0 || (length=(((STabHead*)vl.v.data)-1)->tabLen) == 0 )
		ptr = ptrBeg = ptrEnd = 0;
	else {
		ptr = vl.v.data+length-1; 
		ptrBeg = vl.v.data-1; 
		ptrEnd = vl.v.data+length; 
	}
	return *this;
}

/* Set a vector iterator with the next of some other iterator. */
template <class T, class Resize> typename SVector<T, Resize>::Iter &
		SVector<T, Resize>::Iter::operator=( const IterNext &vn )    
{
	ptr = vn.i.ptr+1; 
	ptrBeg = vn.i.ptrBeg;
	ptrEnd = vn.i.ptrEnd;
	return *this;
}

/* Set a vector iterator with the prev of some other iterator. */
template <class T, class Resize> typename SVector<T, Resize>::Iter &
		SVector<T, Resize>::Iter::operator=( const IterPrev &vp )    
{
	ptr = vp.i.ptr-1; 
	ptrBeg = vp.i.ptrBeg;
	ptrEnd = vp.i.ptrEnd;
	return *this;
}

/* Up resize the data for len elements using Resize::upResize to tell us the
 * new length. Reads and writes allocLen. Does not read or write length.
 * Assumes that there is some data allocated already. */
template <class T, class Resize> void SVector<T, Resize>::
		upResize(long len)
{
	/* Get the current header. */
	STabHead *head = ((STabHead*)BaseTable::data) - 1;

	/* Ask the resizer what the new length will be. */
	long newLen = Resize::upResize(head->allocLen, len);

	/* Did the data grow? */
	if ( newLen > head->allocLen ) {
		head->allocLen = newLen;

		/* Table exists already, resize it up. */
		head = (STabHead*) realloc( head, sizeof(STabHead) + 
				sizeof(T) * newLen );
		if ( head == 0 )
			throw std::bad_alloc();

		/* Save the data pointer. */
		BaseTable::data = (T*) (head + 1);
	}
}

/* Allocates a new buffer for an up resize that requires a duplication of the
 * data. Uses Resize::upResize to get the allocation length.  Reads and writes
 * allocLen. This upResize does write the new length.  Assumes that there is
 * some data allocated already. */
template <class T, class Resize> void SVector<T, Resize>::
		upResizeDup(long len)
{
	/* Get the current header. */
	STabHead *head = ((STabHead*)BaseTable::data) - 1;

	/* Ask the resizer what the new length will be. */
	long newLen = Resize::upResize(head->allocLen, len);

	/* Dereferencing the existing data, decrement the refcount. */
	head->refCount -= 1;

	/* Table exists already, resize it up. */
	head = (STabHead*) malloc( sizeof(STabHead) + sizeof(T) * newLen );
	if ( head == 0 )
		throw std::bad_alloc();

	head->refCount = 1;
	head->allocLen = newLen;
	head->tabLen = len;

	/* Save the data pointer. */
	BaseTable::data = (T*) (head + 1);
}

/* Up resize the data for len elements using Resize::upResize to tell us the
 * new length. Reads and writes allocLen. This upresize DOES write length.
 * Assumes that no data is allocated. */
template <class T, class Resize> void SVector<T, Resize>::
		upResizeFromEmpty(long len)
{
	/* There is no table yet. If the len is zero, then there is no need to
	 * create a table. */
	if ( len > 0 ) {
		/* Ask the resizer what the new length will be. */
		long newLen = Resize::upResize(0, len);

		/* If len is greater than zero then we are always allocating the table. */
		STabHead *head = (STabHead*) malloc( sizeof(STabHead) + 
				sizeof(T) * newLen );
		if ( head == 0 )
			throw std::bad_alloc();

		/* Set up the header and save the data pointer. Note that we set the
		 * length here. This differs from the other upResizes. */
		head->refCount = 1;
		head->allocLen = newLen;
		head->tabLen = len;
		BaseTable::data = (T*) (head + 1);
	}
}

/* Down resize the data for len elements using Resize::downResize to determine
 * the new length. Reads and writes allocLen. Does not read or write length. */
template <class T, class Resize> void SVector<T, Resize>::
		downResize(long len)
{
	/* If there is already no length, then there is nothing we can do. */
	if ( BaseTable::data != 0 ) {
		/* Get the current header. */
		STabHead *head = ((STabHead*)BaseTable::data) - 1;

		/* Ask the resizer what the new length will be. */
		long newLen = Resize::downResize( head->allocLen, len );

		/* Did the data shrink? */
		if ( newLen < head->allocLen ) {
			if ( newLen == 0 ) {
				/* Simply free the data. */
				free( head );
				BaseTable::data = 0;
			}
			else {
				/* Save the new allocated length. */
				head->allocLen = newLen;

				/* Not shrinking to size zero, realloc it to the smaller size. */
				head = (STabHead*) realloc( head, sizeof(STabHead) + 
						sizeof(T) * newLen );
				if ( head == 0 )
					throw std::bad_alloc();
				
				/* Save the new data ptr. */
				BaseTable::data = (T*) (head + 1);
			}
		}
	}
}

/* Allocate a new buffer for a down resize and duplication of the array.  The
 * new array will be len long and allocation size will be determined using
 * Resize::downResize with the old array's allocLen. Does not actually copy
 * any data. Reads and writes allocLen and writes the new len. */
template <class T, class Resize> void SVector<T, Resize>::
		downResizeDup(long len)
{
	/* If there is already no length, then there is nothing we can do. */
	if ( BaseTable::data != 0 ) {
		/* Get the current header. */
		STabHead *head = ((STabHead*)BaseTable::data) - 1;

		/* Ask the resizer what the new length will be. */
		long newLen = Resize::downResize( head->allocLen, len );

		/* Detaching from the existing head, decrement the refcount. */
		head->refCount -= 1;

		/* Not shrinking to size zero, malloc it to the smaller size. */
		head = (STabHead*) malloc( sizeof(STabHead) + sizeof(T) * newLen );
		if ( head == 0 )
			throw std::bad_alloc();

		/* Save the new allocated length. */
		head->refCount = 1;
		head->allocLen = newLen;
		head->tabLen = len;

		/* Save the data pointer. */
		BaseTable::data = (T*) (head + 1);
	}
}

/**
 * \brief Free all memory used by the vector. 
 *
 * The vector is reset to zero elements. Destructors are called on all
 * elements in the vector. The space allocated for the vector is freed.
 */
template <class T, class Resize> void SVector<T, Resize>::
		empty()
{
	if ( BaseTable::data != 0 ) {
		/* Get the header and drop the refcount on the data. */
		STabHead *head = ((STabHead*) BaseTable::data) - 1;
		head->refCount -= 1;

		/* If the refcount just went down to zero nobody else is referencing
		 * the data. */
		if ( head->refCount == 0 ) {
			/* Call All destructors. */
			T *pos = BaseTable::data;
			for ( long i = 0; i < head->tabLen; pos++, i++ )
				pos->~T();

			/* Free the data space. */
			free( head );
		}

		/* Clear the pointer. */
		BaseTable::data = 0;
	}
}

/* Prepare for setting the contents of the vector to some array len long.
 * Handles reusing the existing space, detaching from a common space or
 * growing from zero length automatically. */
template <class T, class Resize> void SVector<T, Resize>::
		setAsCommon(long len)
{
	if ( BaseTable::data != 0 ) {
		/* Get the header. */
		STabHead *head = ((STabHead*)BaseTable::data) - 1;

		/* If the refCount is one, then we can reuse the space. Otherwise we
		 * must detach from the referenced data create new space. */
		if ( head->refCount == 1 ) {
			/* Call All destructors. */
			T *pos = BaseTable::data;
			for ( long i = 0; i < head->tabLen; pos++, i++ )
				pos->~T();

			/* Adjust the allocated length. */
			if ( len < head->tabLen )
				downResize( len );
			else if ( len > head->tabLen )
				upResize( len );

			if ( BaseTable::data != 0 ) {
				/* Get the header again and set the length. */
				head = ((STabHead*)BaseTable::data) - 1;
				head->tabLen = len;
			}
		}
		else {
			/* Just detach from the data. */
			head->refCount -= 1;
			BaseTable::data = 0;
			
			/* Make enough space. This will set the length. */
			upResizeFromEmpty( len );
		}
	}
	else {
		/* The table is currently empty. Make enough space. This will set the
		 * length. */
		upResizeFromEmpty( len );
	}
}

/**
 * \brief Set the contents of the vector to be len elements exactly. 
 *
 * The vector becomes len elements in length. Destructors are called on any
 * existing elements in the vector. Copy constructors are used to place the
 * new elements in the vector. 
 */
template <class T, class Resize> void SVector<T, Resize>::
		setAs(const T *val, long len)
{
	/* Common stuff for setting the array to len long. */
	setAsCommon( len );

	/* Copy data in. */
	T *dst = BaseTable::data;
	const T *src = val;
	for ( long i = 0; i < len; i++, dst++, src++ )
		new(dst) T(*src);
}


/**
 * \brief Set the vector to len copies of item.
 *
 * The vector becomes len elements in length. Destructors are called on any
 * existing elements in the vector. The element's copy constructor is used to
 * copy the item into the vector.
 */
template <class T, class Resize> void SVector<T, Resize>::
		setAsDup(const T &item, long len)
{
	/* Do the common stuff for setting the array to len long. */
	setAsCommon( len );

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
template <class T, class Resize> void SVector<T, Resize>::
		setAsNew(long len)
{
	/* Do the common stuff for setting the array to len long. */
	setAsCommon( len );

	/* Create items using default constructor. */
	T *dst = BaseTable::data;
	for ( long i = 0; i < len; i++, dst++ )
		new(dst) T();
}

/* Make space in vector for a replacement at pos of len items. Handles reusing
 * existing space, detaching or growing from zero space. */
template <class T, class Resize> long SVector<T, Resize>::
		replaceCommon(long pos, long len)
{
	if ( BaseTable::data != 0 ) {
		/* Get the header. */
		STabHead *head = ((STabHead*)BaseTable::data) - 1;

		/* If we are given a negative position to replace at then treat it as
		 * a position relative to the length. This doesn't have any meaning
		 * unless the length is at least one. */
		if ( pos < 0 )
			pos = head->tabLen + pos;

		/* The end is the one past the last item that we want to write to. */
		long i, endPos = pos + len;

		if ( head->refCount == 1 ) {
			/* We can reuse the space. Make sure we have enough space. */
			if ( endPos > head->tabLen ) {
				upResize( endPos );

				/* Get the header again, whose addr may have changed after
				 * resizing. */
				head = ((STabHead*)BaseTable::data) - 1;

				/* Delete any objects we need to delete. */
				T *item = BaseTable::data + pos;
				for ( i = pos; i < head->tabLen; i++, item++ )
					item->~T();
		
				/* We are extending the vector, set the new data length. */
				head->tabLen = endPos;
			}
			else {
				/* Delete any objects we need to delete. */
				T *item = BaseTable::data + pos;
				for ( i = pos; i < endPos; i++, item++ )
					item->~T();
			}
		}
		else {
			/* Use endPos to calc the end of the vector. */
			long newLen = endPos;
			if ( newLen < head->tabLen )
				newLen = head->tabLen;

			/* Duplicate and grow up to endPos. This will set the length. */
			upResizeDup( newLen );

			/* Copy from src up to pos. */
			const T *src = (T*) (head + 1);
			T *dst = BaseTable::data;
			for ( i = 0; i < pos; i++, dst++, src++)
				new(dst) T(*src);

			/* Copy any items after the replace range. */
			for ( i += len, src += len, dst += len; 
					i < head->tabLen; i++, dst++, src++ )
				new(dst) T(*src);
		}
	}
	else {
		/* There is no data initially, must grow from zero. This will set the
		 * new length. */
		upResizeFromEmpty( len );
	}

	return pos;
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
template <class T, class Resize> void SVector<T, Resize>::
		replace(long pos, const T *val, long len)
{
	/* Common work for replacing in the vector. */
	pos = replaceCommon( pos, len );

	/* Copy data in using copy constructor. */
	T *dst = BaseTable::data + pos;
	const T *src = val;
	for ( long i = 0; i < len; i++, dst++, src++ )
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
template <class T, class Resize> void SVector<T, Resize>::
		replaceDup(long pos, const T &val, long len)
{
	/* Common replacement stuff. */
	pos = replaceCommon( pos, len );

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
template <class T, class Resize> void SVector<T, Resize>::
		replaceNew(long pos, long len)
{
	/* Do the common replacement stuff. */
	pos = replaceCommon( pos, len );

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
template <class T, class Resize> void SVector<T, Resize>::
		remove(long pos, long len)
{
	/* If there is no data, we can't delete anything anyways. */
	if ( BaseTable::data != 0 ) {
		/* Get the header. */
		STabHead *head = ((STabHead*)BaseTable::data) - 1;

		/* If we are given a negative position to remove at then
		 * treat it as a position relative to the length. */
		if ( pos < 0 )
			pos = head->tabLen + pos;

		/* The first position after the last item deleted. */
		long endPos = pos + len;

		/* The New data length. */
		long i, newLen = head->tabLen - len;

		if ( head->refCount == 1 ) {
			/* We are the only ones using the data. We can reuse 
			 * the existing space. */

			/* The place in the data we are deleting at. */
			T *dst = BaseTable::data + pos;

			/* Call Destructors. */
			T *item = BaseTable::data + pos;
			for ( i = 0; i < len; i += 1, item += 1 )
				item->~T();

			/* Shift data over if necessary. */
			long lenToSlideOver = head->tabLen - endPos;	
			if ( len > 0 && lenToSlideOver > 0 )
				memmove(BaseTable::data + pos, dst + len, sizeof(T)*lenToSlideOver);

			/* Shrink the data if necessary. */
			downResize( newLen );

			if ( BaseTable::data != 0 ) {
				/* Get the header again (because of the resize) and set the
				 * new data length. */
				head = ((STabHead*)BaseTable::data) - 1;
				head->tabLen = newLen;
			}
		}
		else {
			/* Must detach from the common data. Just copy the non-deleted
			 * items from the common data. */

			/* Duplicate and grow down to newLen. This will set the length. */
			downResizeDup( newLen );

			/* Copy over just the non-deleted parts. */
			const T *src = (T*) (head + 1);
			T *dst = BaseTable::data;
			for ( i = 0; i < pos; i++, dst++, src++ )
				new(dst) T(*src);

			/* ... and the second half. */
			for ( i += len, src += len; i < head->tabLen; i++, src++, dst++ )
				new(dst) T(*src);
		}
	}
}

/* Shift over existing data. Handles reusing existing space, detaching or
 * growing from zero space. */
template <class T, class Resize> long SVector<T, Resize>::
		insertCommon(long pos, long len)
{
	if ( BaseTable::data != 0 ) {
		/* Get the header. */
		STabHead *head = ((STabHead*)BaseTable::data) - 1;

		/* If we are given a negative position to insert at then treat it as a
		 * position relative to the length. This only has meaning if there is
		 * existing data. */
		if ( pos < 0 )
			pos = head->tabLen + pos;

		/* Calculate the new length. */
		long i, newLen = head->tabLen + len;

		if ( head->refCount == 1 ) {
			/* Up resize, we are growing. */
			upResize( newLen );

			/* Get the header again, (the addr may have changed after
			 * resizing). */
			head = ((STabHead*)BaseTable::data) - 1;

			/* Shift over data at insert spot if needed. */
			if ( len > 0 && pos < head->tabLen ) {
				memmove( BaseTable::data + pos + len, BaseTable::data + pos,
						sizeof(T)*(head->tabLen - pos) );
			}

			/* Grow the length by the len inserted. */
			head->tabLen += len;
		}
		else {
			/* Need to detach from the existing array. Copy over the other
			 * parts. This will set the length. */
			upResizeDup( newLen );

			/* Copy over the parts around the insert. */
			const T *src = (T*) (head + 1);
			T *dst = BaseTable::data;
			for ( i = 0; i < pos; i++, dst++, src++ )
				new(dst) T(*src);

			/* ... and the second half. */
			for ( dst += len; i < head->tabLen; i++, src++, dst++ )
				new(dst) T(*src);
		}
	}
	else {
		/* There is no existing data. Start from zero. This will set the
		 * length. */
		upResizeFromEmpty( len );
	}

	return pos;
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
template <class T, class Resize> void SVector<T, Resize>::
		insert(long pos, const T *val, long len)
{
	/* Do the common insertion stuff. */
	pos = insertCommon( pos, len );

	/* Copy data in element by element. */
	T *dst = BaseTable::data + pos;
	const T *src = val;
	for ( long i = 0; i < len; i++, dst++, src++ )
		new(dst) T(*src);
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
template <class T, class Resize> void SVector<T, Resize>::
		insertDup(long pos, const T &item, long len)
{
	/* Do the common insertion stuff. */
	pos = insertCommon( pos, len );

	/* Copy the data item in one at a time. */
	T *dst = BaseTable::data + pos;
	for ( long i = 0; i < len; i++, dst++ )
		new(dst) T(item);
}


/**
 * \brief Insert len new elements using the default constructor.
 *
 * Elements in the vector from pos onward are shifted len spaces to the right.
 * Default constructors are used to init the new elements. If pos is off the
 * end of the vector then undefined behaviour results. If pos is negative then
 * it is treated as an offset relative to the length of the vector.
 */
template <class T, class Resize> void SVector<T, Resize>::
		insertNew(long pos, long len)
{
	/* Do the common insertion stuff. */
	pos = insertCommon( pos, len );

	/* Init new data with default constructors. */
	T *dst = BaseTable::data + pos;
	for ( long i = 0; i < len; i++, dst++ )
		new(dst) T();
}

/* Makes space for len items, Does not init the items in any way.  If pos is
 * greater than the length of the vector then undefined behaviour results.
 * Updates the length of the vector. */
template <class T, class Resize> void SVector<T, Resize>::
		makeRawSpaceFor(long pos, long len)
{
	if ( BaseTable::data != 0 ) {
		/* Get the header. */
		STabHead *head = ((STabHead*)BaseTable::data) - 1;

		/* Calculate the new length. */
		long i, newLen = head->tabLen + len;

		if ( head->refCount == 1 ) {
			/* Up resize, we are growing. */
			upResize( newLen );

			/* Get the header again, (the addr may have changed after
			 * resizing). */
			head = ((STabHead*)BaseTable::data) - 1;

			/* Shift over data at insert spot if needed. */
			if ( len > 0 && pos < head->tabLen ) {
				memmove( BaseTable::data + pos + len, BaseTable::data + pos,
						sizeof(T)*(head->tabLen - pos) );
			}

			/* Grow the length by the len inserted. */
			head->tabLen += len;
		}
		else {
			/* Need to detach from the existing array. Copy over the other
			 * parts. This will set the length. */
			upResizeDup( newLen );

			/* Copy over the parts around the insert. */
			const T *src = (T*) (head + 1);
			T *dst = BaseTable::data;
			for ( i = 0; i < pos; i++, dst++, src++ )
				new(dst) T(*src);

			/* ... and the second half. */
			for ( dst += len; i < head->tabLen; i++, src++, dst++ )
				new(dst) T(*src);
		}
	}
	else {
		/* There is no existing data. Start from zero. This will set the
		 * length. */
		upResizeFromEmpty( len );
	}
}


#ifdef AAPL_NAMESPACE
}
#endif


#endif /* _AAPL_SVECTOR_H */
