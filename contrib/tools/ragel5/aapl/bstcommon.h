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

/* This header is not wrapped in ifndefs because it is
 * not intended to be included by users directly. */

#ifdef AAPL_NAMESPACE
namespace Aapl {
#endif

/* Binary Search Table */
template < BST_TEMPL_DECLARE > class BstTable :
		public Compare,
		public Vector< Element, Resize >
{
	typedef Vector<Element, Resize> BaseVector;
	typedef Table<Element> BaseTable;

public:
	/**
	 * \brief Default constructor.
	 *
	 * Create an empty binary search table.
	 */
	BstTable() { }

	/**
	 * \brief Construct with initial value.
	 *
	 * Constructs a binary search table with an initial item. Uses the default
	 * constructor for initializing Value.
	 */
	BstTable(const Key &key)
		{ insert(key); }

#if defined( BSTMAP )
	/**
	 * \brief Construct with initial value.
	 *
	 * Constructs a binary search table with an initial key/value pair.
	 */
	BstTable(const Key &key, const Value &val)
		{ insert(key, val); }
#endif

#if ! defined( BSTSET )
	/**
	 * \brief Construct with initial value.
	 *
	 * Constructs a binary search table with an initial Element.
	 */
	BstTable(const Element &el)
		{ insert(el); }
#endif

	Element *insert(const Key &key, Element **lastFound = 0);
	Element *insertMulti(const Key &key);

	bool insert(const BstTable &other);
	void insertMulti(const BstTable &other);

#if defined( BSTMAP )
	Element *insert(const Key &key, const Value &val,
			Element **lastFound = 0);
	Element *insertMulti(const Key &key, const Value &val );
#endif

#if ! defined( BSTSET )
	Element *insert(const Element &el, Element **lastFound = 0);
	Element *insertMulti(const Element &el);
#endif

	Element *find(const Key &key, Element **lastFound = 0) const;
	bool findMulti( const Key &key, Element *&lower,
			Element *&upper ) const;

	bool remove(const Key &key);
	bool remove(Element *item);
	long removeMulti(const Key &key);
	long removeMulti(Element *lower, Element *upper);

	/* The following provide access to the underlying insert and remove
	 * functions that my be hidden by the BST insert and remove. The insertDup
	 * and insertNew functions will never be hidden. They are provided for
	 * consistency. The difference between the non-shared and the shared
	 * tables is the documentation reference to the invoked function. */

#if !defined( SHARED_BST )
	/*@{*/

	/** \brief Call the insert of the underlying vector.
	 *
	 * Provides to access to the vector insert, which may become hidden. Care
	 * should be taken to ensure that after the insert the ordering of
	 * elements is preserved.
	 * Invokes Vector::insert( long pos, const T &val ).
	 */
	void vinsert(long pos, const Element &val)
		{ Vector< Element, Resize >::insert( pos, &val, 1 ); }

	/** \brief Call the insert of the underlying vector.
	 *
	 * Provides to access to the vector insert, which may become hidden. Care
	 * should be taken to ensure that after the insert the ordering of
	 * elements is preserved.
	 * Invokes Vector::insert( long pos, const T *val, long len ).
	 */
	void vinsert(long pos, const Element *val, long len)
		{ Vector< Element, Resize >::insert( pos, val, len ); }

	/** \brief Call the insert of the underlying vector.
	 *
	 * Provides to access to the vector insert, which may become hidden. Care
	 * should be taken to ensure that after the insert the ordering of
	 * elements is preserved.
	 * Invokes Vector::insert( long pos, const Vector &v ).
	 */
	void vinsert(long pos, const BstTable &v)
		{ Vector< Element, Resize >::insert( pos, v.data, v.tabLen ); }

	/*@}*/

	/*@{*/

	/** \brief Call the remove of the underlying vector.
	 *
	 * 	Provides access to the vector remove, which may become hidden.
	 * 	Invokes Vector::remove( long pos ).
	 */
	void vremove(long pos)
		{ Vector< Element, Resize >::remove( pos, 1 ); }

	/** \brief Call the remove of the underlying vector.
	 *
	 * Proves access to the vector remove, which may become hidden.
	 * Invokes Vector::remove( long pos, long len ).
	 */
	void vremove(long pos, long len)
		{ Vector< Element, Resize >::remove( pos, len ); }

	/*@}*/
#else /* SHARED_BST */
	/*@{*/

	/** \brief Call the insert of the underlying vector.
	 *
	 * Provides to access to the vector insert, which may become hidden. Care
	 * should be taken to ensure that after the insert the ordering of
	 * elements is preserved.
	 * Invokes SVector::insert( long pos, const T &val ).
	 */
	void vinsert(long pos, const Element &val)
		{ Vector< Element, Resize >::insert( pos, &val, 1 ); }

	/** \brief Call the insert of the underlying vector.
	 *
	 * Provides to access to the vector insert, which may become hidden. Care
	 * should be taken to ensure that after the insert the ordering of
	 * elements is preserved.
	 * Invokes SVector::insert( long pos, const T *val, long len ).
	 */
	void vinsert(long pos, const Element *val, long len)
		{ Vector< Element, Resize >::insert( pos, val, len ); }

	/** \brief Call the insert of the underlying vector.
	 *
	 * Provides to access to the vector insert, which may become hidden. Care
	 * should be taken to ensure that after the insert the ordering of
	 * elements is preserved.
	 * Invokes SVector::insert( long pos, const SVector &v ).
	 */
	void vinsert(long pos, const BstTable &v)
		{ Vector< Element, Resize >::insert( pos, v.data, v.length() ); }

	/*@}*/

	/*@{*/

	/** \brief Call the remove of the underlying vector.
	 *
	 * 	Provides access to the vector remove, which may become hidden.
	 * 	Invokes SVector::remove( long pos ).
	 */
	void vremove(long pos)
		{ Vector< Element, Resize >::remove( pos, 1 ); }

	/** \brief Call the remove of the underlying vector.
	 *
	 * Proves access to the vector remove, which may become hidden.
	 * Invokes SVector::remove( long pos, long len ).
	 */
	void vremove(long pos, long len)
		{ Vector< Element, Resize >::remove( pos, len ); }

	/*@}*/

#endif /* SHARED_BST */
};


#if 0
#if defined( SHARED_BST )
/**
 * \brief Construct a binary search table with an initial amount of
 * allocation.
 *
 * The table is initialized to have room for allocLength elements. The
 * table starts empty.
 */
template <BST_TEMPL_DEF> BstTable<BST_TEMPL_USE>::
		BstTable( long allocLen )
{
	/* Allocate the space if we are given a positive allocLen. */
	if ( allocLen > 0 ) {
		/* Allocate the data needed. */
		STabHead *head = (STabHead*)
				malloc( sizeof(STabHead) + sizeof(Element) * allocLen );
		if ( head == 0 )
			throw std::bad_alloc();

		/* Set up the header and save the data pointer. */
		head->refCount = 1;
		head->allocLen = allocLen;
		head->tabLen = 0;
		BaseTable::data = (Element*) (head + 1);
	}
}
#else
/**
 * \brief Construct a binary search table with an initial amount of
 * allocation.
 *
 * The table is initialized to have room for allocLength elements. The
 * table starts empty.
 */
template <BST_TEMPL_DEF> BstTable<BST_TEMPL_USE>::
		BstTable( long allocLen )
{
	/* Allocate the space if we are given a positive allocLen. */
	BaseTable::allocLen = allocLen;
	if ( BaseTable::allocLen > 0 ) {
		BaseTable::data = (Element*) malloc(sizeof(Element) * BaseTable::allocLen);
		if ( BaseTable::data == NULL )
			throw std::bad_alloc();
	}
}

#endif
#endif

/**
 * \brief Find the element with the given key and remove it.
 *
 * If multiple elements with the given key exist, then it is unspecified which
 * element will be removed.
 *
 * \returns True if an element is found and consequently removed, false
 * otherwise.
 */
template <BST_TEMPL_DEF> bool BstTable<BST_TEMPL_USE>::
		remove(const Key &key)
{
	Element *el = find(key);
	if ( el != 0 ) {
		Vector< Element >::remove(el - BaseTable::data);
		return true;
	}
	return false;
}

/**
 * \brief Remove the element pointed to by item.
 *
 * If item does not point to an element in the tree, then undefined behaviour
 * results. If item is null, then remove has no effect.
 *
 * \returns True if item is not null, false otherwise.
 */
template <BST_TEMPL_DEF> bool BstTable<BST_TEMPL_USE>::
		remove( Element *item )
{
	if ( item != 0 ) {
		Vector< Element >::remove(item - BaseTable::data);
		return true;
	}
	return false;
}

/**
 * \brief Find and remove the entire range of elements with the given key.
 *
 * \returns The number of elements removed.
 */
template <BST_TEMPL_DEF> long BstTable<BST_TEMPL_USE>::
		removeMulti(const Key &key)
{
	Element *low, *high;
	if ( findMulti(key, low, high) ) {
		/* Get the length of the range. */
		long num = high - low + 1;
		Vector< Element >::remove(low - BaseTable::data, num);
		return num;
	}

	return 0;
}

template <BST_TEMPL_DEF> long BstTable<BST_TEMPL_USE>::
		removeMulti(Element *lower, Element *upper)
{
	/* Get the length of the range. */
	long num = upper - lower + 1;
	Vector< Element >::remove(lower - BaseTable::data, num);
	return num;
}


/**
 * \brief Find a range of elements with the given key.
 *
 * If any elements with the given key exist then lower and upper are set to
 * the low and high ends of the continous range of elements with the key.
 * Lower and upper will point to the first and last elements with the key.
 *
 * \returns True if any elements are found, false otherwise.
 */
template <BST_TEMPL_DEF> bool BstTable<BST_TEMPL_USE>::
		findMulti(const Key &key, Element *&low, Element *&high ) const
{
	const Element *lower, *mid, *upper;
	long keyRelation;
	const long tblLen = BaseTable::length();

	if ( BaseTable::data == 0 )
		return false;

	lower = BaseTable::data;
	upper = BaseTable::data + tblLen - 1;
	while ( true ) {
		if ( upper < lower ) {
			/* Did not find the fd in the array. */
			return false;
		}

		mid = lower + ((upper-lower)>>1);
		keyRelation = this->compare(key, GET_KEY(*mid));

		if ( keyRelation < 0 )
			upper = mid - 1;
		else if ( keyRelation > 0 )
			lower = mid + 1;
		else {
			Element *lowEnd = BaseTable::data - 1;
			Element *highEnd = BaseTable::data + tblLen;

			lower = mid - 1;
			while ( lower != lowEnd &&
					this->compare(key, GET_KEY(*lower)) == 0 )
				lower--;

			upper = mid + 1;
			while ( upper != highEnd &&
					this->compare(key, GET_KEY(*upper)) == 0 )
				upper++;

			low = (Element*)lower + 1;
			high = (Element*)upper - 1;
			return true;
		}
	}
}

/**
 * \brief Find an element with the given key.
 *
 * If the find succeeds then lastFound is set to the element found. If the
 * find fails then lastFound is set the location where the key would be
 * inserted. If there is more than one element in the tree with the given key,
 * then it is unspecified which element is returned as the match.
 *
 * \returns The element found on success, null on failure.
 */
template <BST_TEMPL_DEF> Element *BstTable<BST_TEMPL_USE>::
		find( const Key &key, Element **lastFound ) const
{
	const Element *lower, *mid, *upper;
	long keyRelation;
	const long tblLen = BaseTable::length();

	if ( BaseTable::data == 0 )
		return 0;

	lower = BaseTable::data;
	upper = BaseTable::data + tblLen - 1;
	while ( true ) {
		if ( upper < lower ) {
			/* Did not find the key. Last found gets the insert location. */
			if ( lastFound != 0 )
				*lastFound = (Element*)lower;
			return 0;
		}

		mid = lower + ((upper-lower)>>1);
		keyRelation = this->compare(key, GET_KEY(*mid));

		if ( keyRelation < 0 )
			upper = mid - 1;
		else if ( keyRelation > 0 )
			lower = mid + 1;
		else {
			/* Key is found. Last found gets the found record. */
			if ( lastFound != 0 )
				*lastFound = (Element*)mid;
			return (Element*)mid;
		}
	}
}

template <BST_TEMPL_DEF> Element *BstTable<BST_TEMPL_USE>::
		insert(const Key &key, Element **lastFound)
{
	const Element *lower, *mid, *upper;
	long keyRelation, insertPos;
	const long tblLen = BaseTable::length();

	if ( tblLen == 0 ) {
		/* If the table is empty then go straight to insert. */
		lower = BaseTable::data;
		goto insert;
	}

	lower = BaseTable::data;
	upper = BaseTable::data + tblLen - 1;
	while ( true ) {
		if ( upper < lower ) {
			/* Did not find the key in the array.
			 * Place to insert at is lower. */
			goto insert;
		}

		mid = lower + ((upper-lower)>>1);
		keyRelation = this->compare(key, GET_KEY(*mid));

		if ( keyRelation < 0 )
			upper = mid - 1;
		else if ( keyRelation > 0 )
			lower = mid + 1;
		else {
			if ( lastFound != 0 )
				*lastFound = (Element*)mid;
			return 0;
		}
	}

insert:
	/* Get the insert pos. */
	insertPos = lower - BaseTable::data;

	/* Do the insert. After makeRawSpaceFor, lower pointer is no good. */
	BaseVector::makeRawSpaceFor(insertPos, 1);
	new(BaseTable::data + insertPos) Element(key);

	/* Set lastFound */
	if ( lastFound != 0 )
		*lastFound = BaseTable::data + insertPos;
	return BaseTable::data + insertPos;
}


template <BST_TEMPL_DEF> Element *BstTable<BST_TEMPL_USE>::
		insertMulti(const Key &key)
{
	const Element *lower, *mid, *upper;
	long keyRelation, insertPos;
	const long tblLen = BaseTable::length();

	if ( tblLen == 0 ) {
		/* If the table is empty then go straight to insert. */
		lower = BaseTable::data;
		goto insert;
	}

	lower = BaseTable::data;
	upper = BaseTable::data + tblLen - 1;
	while ( true ) {
		if ( upper < lower ) {
			/* Did not find the key in the array.
			 * Place to insert at is lower. */
			goto insert;
		}

		mid = lower + ((upper-lower)>>1);
		keyRelation = compare(key, GET_KEY(*mid));

		if ( keyRelation < 0 )
			upper = mid - 1;
		else if ( keyRelation > 0 )
			lower = mid + 1;
		else {
			lower = mid;
			goto insert;
		}
	}

insert:
	/* Get the insert pos. */
	insertPos = lower - BaseTable::data;

	/* Do the insert. */
	BaseVector::makeRawSpaceFor(insertPos, 1);
	new(BaseTable::data + insertPos) Element(key);

	/* Return the element inserted. */
	return BaseTable::data + insertPos;
}

/**
 * \brief Insert each element from other.
 *
 * Always attempts to insert all elements even if the insert of some item from
 * other fails.
 *
 * \returns True if all items inserted successfully, false if any insert
 * failed.
 */
template <BST_TEMPL_DEF> bool BstTable<BST_TEMPL_USE>::
		insert(const BstTable &other)
{
	bool allSuccess = true;
	long otherLen = other.length();
	for ( long i = 0; i < otherLen; i++ ) {
		Element *el = insert( other.data[i] );
		if ( el == 0 )
			allSuccess = false;
	}
	return allSuccess;
}

/**
 * \brief Insert each element from other even if the elements exist already.
 *
 * No individual insertMulti can fail.
 */
template <BST_TEMPL_DEF> void BstTable<BST_TEMPL_USE>::
		insertMulti(const BstTable &other)
{
	long otherLen = other.length();
	for ( long i = 0; i < otherLen; i++ )
		insertMulti( other.data[i] );
}

#if ! defined( BSTSET )

/**
 * \brief Insert the given element.
 *
 * If the key in the given element does not already exist in the table then a
 * new element is inserted. They element copy constructor is used to place the
 * element into the table. If lastFound is given, it is set to the new element
 * created. If the insert fails then lastFound is set to the existing element
 * of the same key.
 *
 * \returns The new element created upon success, null upon failure.
 */
template <BST_TEMPL_DEF> Element *BstTable<BST_TEMPL_USE>::
		insert(const Element &el, Element **lastFound )
{
	const Element *lower, *mid, *upper;
	long keyRelation, insertPos;
	const long tblLen = BaseTable::length();

	if ( tblLen == 0 ) {
		/* If the table is empty then go straight to insert. */
		lower = BaseTable::data;
		goto insert;
	}

	lower = BaseTable::data;
	upper = BaseTable::data + tblLen - 1;
	while ( true ) {
		if ( upper < lower ) {
			/* Did not find the key in the array.
			 * Place to insert at is lower. */
			goto insert;
		}

		mid = lower + ((upper-lower)>>1);
		keyRelation = compare(GET_KEY(el), GET_KEY(*mid));

		if ( keyRelation < 0 )
			upper = mid - 1;
		else if ( keyRelation > 0 )
			lower = mid + 1;
		else {
			if ( lastFound != 0 )
				*lastFound = (Element*)mid;
			return 0;
		}
	}

insert:
	/* Get the insert pos. */
	insertPos = lower - BaseTable::data;

	/* Do the insert. After makeRawSpaceFor, lower pointer is no good. */
	BaseVector::makeRawSpaceFor(insertPos, 1);
	new(BaseTable::data + insertPos) Element(el);

	/* Set lastFound */
	if ( lastFound != 0 )
		*lastFound = BaseTable::data + insertPos;
	return BaseTable::data + insertPos;
}

/**
 * \brief Insert the given element even if it exists already.
 *
 * If the key in the given element exists already then the new element is
 * placed next to some other element of the same key. InsertMulti cannot fail.
 * The element copy constructor is used to place the element in the table.
 *
 * \returns The new element created.
 */
template <BST_TEMPL_DEF> Element *BstTable<BST_TEMPL_USE>::
		insertMulti(const Element &el)
{
	const Element *lower, *mid, *upper;
	long keyRelation, insertPos;
	const long tblLen = BaseTable::length();

	if ( tblLen == 0 ) {
		/* If the table is empty then go straight to insert. */
		lower = BaseTable::data;
		goto insert;
	}

	lower = BaseTable::data;
	upper = BaseTable::data + tblLen - 1;
	while ( true ) {
		if ( upper < lower ) {
			/* Did not find the fd in the array.
			 * Place to insert at is lower. */
			goto insert;
		}

		mid = lower + ((upper-lower)>>1);
		keyRelation = this->compare(GET_KEY(el), GET_KEY(*mid));

		if ( keyRelation < 0 )
			upper = mid - 1;
		else if ( keyRelation > 0 )
			lower = mid + 1;
		else {
			lower = mid;
			goto insert;
		}
	}

insert:
	/* Get the insert pos. */
	insertPos = lower - BaseTable::data;

	/* Do the insert. */
	BaseVector::makeRawSpaceFor(insertPos, 1);
	new(BaseTable::data + insertPos) Element(el);

	/* Return the element inserted. */
	return BaseTable::data + insertPos;
}
#endif


#if defined( BSTMAP )

/**
 * \brief Insert the given key-value pair.
 *
 * If the given key does not already exist in the table then the key-value
 * pair is inserted. Copy constructors are used to place the pair in the
 * table. If lastFound is given, it is set to the new entry created. If the
 * insert fails then lastFound is set to the existing pair of the same key.
 *
 * \returns The new element created upon success, null upon failure.
 */
template <BST_TEMPL_DEF> Element *BstTable<BST_TEMPL_USE>::
		insert(const Key &key, const Value &val, Element **lastFound)
{
	const Element *lower, *mid, *upper;
	long keyRelation, insertPos;
	const long tblLen = BaseTable::length();

	if ( tblLen == 0 ) {
		/* If the table is empty then go straight to insert. */
		lower = BaseTable::data;
		goto insert;
	}

	lower = BaseTable::data;
	upper = BaseTable::data + tblLen - 1;
	while ( true ) {
		if ( upper < lower ) {
			/* Did not find the fd in the array.
			 * Place to insert at is lower. */
			goto insert;
		}

		mid = lower + ((upper-lower)>>1);
		keyRelation = Compare::compare(key, mid->key);

		if ( keyRelation < 0 )
			upper = mid - 1;
		else if ( keyRelation > 0 )
			lower = mid + 1;
		else {
			if ( lastFound != NULL )
				*lastFound = (Element*)mid;
			return 0;
		}
	}

insert:
	/* Get the insert pos. */
	insertPos = lower - BaseTable::data;

	/* Do the insert. */
	BaseVector::makeRawSpaceFor(insertPos, 1);
	new(BaseTable::data + insertPos) Element(key, val);

	/* Set lastFound */
	if ( lastFound != NULL )
		*lastFound = BaseTable::data + insertPos;
	return BaseTable::data + insertPos;
}


/**
 * \brief Insert the given key-value pair even if the key exists already.
 *
 * If the key exists already then the key-value pair is placed next to some
 * other pair of the same key. InsertMulti cannot fail. Copy constructors are
 * used to place the pair in the table.
 *
 * \returns The new element created.
 */
template <BST_TEMPL_DEF> Element *BstTable<BST_TEMPL_USE>::
		insertMulti(const Key &key, const Value &val)
{
	const Element *lower, *mid, *upper;
	long keyRelation, insertPos;
	const long tblLen = BaseTable::length();

	if ( tblLen == 0 ) {
		/* If the table is empty then go straight to insert. */
		lower = BaseTable::data;
		goto insert;
	}

	lower = BaseTable::data;
	upper = BaseTable::data + tblLen - 1;
	while ( true ) {
		if ( upper < lower ) {
			/* Did not find the key in the array.
			 * Place to insert at is lower. */
			goto insert;
		}

		mid = lower + ((upper-lower)>>1);
		keyRelation = Compare::compare(key, mid->key);

		if ( keyRelation < 0 )
			upper = mid - 1;
		else if ( keyRelation > 0 )
			lower = mid + 1;
		else {
			lower = mid;
			goto insert;
		}
	}

insert:
	/* Get the insert pos. */
	insertPos = lower - BaseTable::data;

	/* Do the insert. */
	BaseVector::makeRawSpaceFor(insertPos, 1);
	new(BaseTable::data + insertPos) Element(key, val);

	/* Return the element inserted. */
	return BaseTable::data + insertPos;
}

#endif

#ifdef AAPL_NAMESPACE
}
#endif
