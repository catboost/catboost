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

/* This header is not wrapped in ifndef becuase it is not intended to
 * be included by the user. */

#ifdef AAPL_NAMESPACE
namespace Aapl {
#endif

#if defined( DOUBLELIST_VALUE )
/**
 * \brief Double list element for DListVal.
 *
 * DListValEl stores the type T of DListVal by value. 
 */
template <class T> struct DListValEl
{
	/**
	 * \brief Construct a DListValEl with a given value.
	 *
	 * The only constructor available initializes the value element. This
	 * enforces that DListVal elements are never created without having their
	 * value intialzed by the user. T's copy constructor is used to copy the
	 * value in.
	 */
	DListValEl( const T &val ) : value(val) { }

	/**
	 * \brief Value stored by the list element.
	 *
	 * Value is always copied into new list elements using the copy
	 * constructor.
	 */
	T value;

	/**
	 * \brief List previous pointer.
	 *
	 * Points to the previous item in the list. If this is the first item in
	 * the list, then prev is NULL. If this element is not in a list then
	 * prev is undefined.
	 */
	DListValEl<T> *prev;

	/**
	 * \brief List next pointer.
	 *
	 * Points to the next item in the list. If this is the list item in the
	 * list, then next is NULL. If this element is not in a list then next is
	 * undefined.
	 */
	DListValEl<T> *next;
};
#else

#ifndef __AAPL_DOUBLE_LIST_EL
#define __AAPL_DOUBLE_LIST_EL
/**
 * \brief Double list element properties.
 *
 * This class can be inherited to make a class suitable to be a double list
 * element. It simply provides the next and previous pointers. An alternative
 * is to put the next and previous pointers in the class directly.
 */
template <class Element> struct DListEl
{
	/**
	 * \brief List previous pointer.
	 *
	 * Points to the previous item in the list. If this is the first item in
	 * the list, then prev is NULL. If this element is not in a list then
	 * prev is undefined.
	 */
	Element *prev;

	/**
	 * \brief List next pointer.
	 *
	 * Points to the next item in the list. If this is the list item in the
	 * list, then next is NULL. If this element is not in a list then next is
	 * undefined.
	 */
	Element *next;
};
#endif /* __AAPL_DOUBLE_LIST_EL */

#endif

/* Doubly Linked List */
template <DLMEL_TEMPDEF> class DList
{
public:
	/** \brief Initialize an empty list. */
	DList() : head(0), tail(0), listLen(0) {}

	/** 
	 * \brief Perform a deep copy of the list.
	 * 
	 * The elements of the other list are duplicated and put into this list.
	 * Elements are copied using the copy constructor.
	 */
	DList(const DList &other);

#ifdef DOUBLELIST_VALUE
	/**
	 * \brief Clear the double list contents.
	 *
	 * All elements are deleted.
	 */
	~DList() { empty(); }

	/**
	 * \brief Assign another list into this list using a deep copy.
	 *
	 * The elements of the other list are duplicated and put into this list.
	 * Each list item is created using the copy constructor. If this list
	 * contains any elements before the copy, they are deleted first.
	 *
	 * \returns A reference to this.
	 */
	DList &operator=(const DList &other);

	/**
	 * \brief Transfer the contents of another list into this list.
	 *
	 * The elements of the other list moved in. The other list will be empty
	 * afterwards.  If this list contains any elements before the copy, then
	 * they are deleted. 
	 */
	void transfer(DList &other);
#else
	/**
	 * \brief Abandon all elements in the list. 
	 *
	 * List elements are not deleted.
	 */
	~DList() {}

	/**
	 * \brief Perform a deep copy of the list.
	 *
	 * The elements of the other list are duplicated and put into this list.
	 * Each list item is created using the copy constructor. If this list
	 * contains any elements before the copy, they are abandoned.
	 *
	 * \returns A reference to this.
	 */
	DList &operator=(const DList &other);

	/**
	 * \brief Transfer the contents of another list into this list.
	 *
	 * The elements of the other list moved in. The other list will be empty
	 * afterwards.  If this list contains any elements before the copy, they
	 * are abandoned. 
	 */
	void transfer(DList &other);
#endif


#ifdef DOUBLELIST_VALUE
	/**
	 * \brief Make a new element and prepend it to the front of the list.
	 *
	 * The item is copied into the new element using the copy constructor.
	 * Equivalent to list.addBefore(list.head, item).
	 */
	void prepend(const T &item);

	/**
	 * \brief Make a new element and append it to the end of the list.
	 *
	 * The item is copied into the new element using the copy constructor.
	 * Equivalent to list.addAfter(list.tail, item).
	 */
	void append(const T &item);

	/**
	 * \brief Make a new element and insert it immediately after an element in
	 * the list.
	 *
	 * The item is copied into the new element using the copy constructor. If
	 * prev_el is NULL then the new element is prepended to the front of the
	 * list. If prev_el is not already in the list then undefined behaviour
	 * results.  Equivalent to list.addAfter(prev_el, new DListValEl(item)).
	 */
	void addAfter(Element *prev_el, const T &item);

	/**
	 * \brief Make a new element and insert it immediately before an element
	 * in the list. 
	 *
	 * The item is copied into the new element using the copy construcotor. If
	 * next_el is NULL then the new element is appended to the end of the
	 * list.  If next_el is not already in the list then undefined behaviour
	 * results.  Equivalent to list.addBefore(next_el, new DListValEl(item)).
	 */
	void addBefore(Element *next_el, const T &item);
#endif

	/**
	 * \brief Prepend a single element to the front of the list.
	 *
	 * If new_el is already an element of some list, then undefined behaviour
	 * results. Equivalent to list.addBefore(list.head, new_el).
	 */
	void prepend(Element *new_el) { addBefore(head, new_el); }

	/**
	 * \brief Append a single element to the end of the list.
	 *
	 * If new_el is alreay an element of some list, then undefined behaviour
	 * results.  Equivalent to list.addAfter(list.tail, new_el).
	 */
	void append(Element *new_el)  { addAfter(tail, new_el); }

	/**
	 * \brief Prepend an entire list to the beginning of this list.
	 *
	 * All items are moved, not copied. Afterwards, the other list is emtpy.
	 * All items are prepended at once, so this is an O(1) operation.
	 * Equivalent to list.addBefore(list.head, dl).
	 */
	void prepend(DList &dl)       { addBefore(head, dl); }

	/**
	 * \brief Append an entire list to the end of the list.
	 *
	 * All items are moved, not copied. Afterwards, the other list is empty.
	 * All items are appened at once, so this is an O(1) operation.
	 * Equivalent to list.addAfter(list.tail, dl).
	 */
	void append(DList &dl)        { addAfter(tail, dl); }

	void addAfter(Element *prev_el, Element *new_el);
	void addBefore(Element *next_el, Element *new_el);

	void addAfter(Element *prev_el, DList &dl);
	void addBefore(Element *next_el, DList &dl);

	/**
	 * \brief Detach the head of the list
	 *
	 * The element detached is not deleted. If there is no head of the list
	 * (the list is empty) then undefined behaviour results.  Equivalent to
	 * list.detach(list.head).
	 *
	 * \returns The element detached.
	 */
	Element *detachFirst()        { return detach(head); }

	/**
	 * \brief Detach the tail of the list
	 *
	 * The element detached is not deleted. If there is no tail of the list
	 * (the list is empty) then undefined behaviour results.  Equivalent to
	 * list.detach(list.tail).
	 *
	 * \returns The element detached.
	 */
	Element *detachLast()         { return detach(tail); }

 	/* Detaches an element from the list. Does not free any memory. */
	Element *detach(Element *el);

	/**
	 * \brief Detach and delete the first element in the list.
	 *
	 * If there is no first element (the list is empty) then undefined
	 * behaviour results.  Equivalent to delete list.detach(list.head);
	 */
	void removeFirst()         { delete detach( head ); }

	/**
	 * \brief Detach and delete the last element in the list.
	 *
	 * If there is no last element (the list is emtpy) then undefined
	 * behaviour results.  Equivalent to delete list.detach(list.tail);
	 */
	void removeLast()          { delete detach( tail ); }

	/**
	 * \brief Detach and delete an element from the list.
	 *
	 * If the element is not in the list, then undefined behaviour results.
	 * Equivalent to delete list.detach(el);
	 */
	void remove(Element *el)   { delete detach( el ); }
	
	void empty();
	void abandon();

	/** \brief The number of elements in the list. */
	long length() const { return listLen; }

	/** \brief Head and tail of the linked list. */
	Element *head, *tail;

	/** \brief The number of element in the list. */
	long listLen;

	/* Convenience access. */
	long size() const           { return listLen; }

	/* Forward this so a ref can be used. */
	struct Iter;

	/* Class for setting the iterator. */
	struct IterFirst { IterFirst( const DList &l ) : l(l) { } const DList &l; };
	struct IterLast { IterLast( const DList &l ) : l(l) { } const DList &l; };
	struct IterNext { IterNext( const Iter &i ) : i(i) { } const Iter &i; };
	struct IterPrev { IterPrev( const Iter &i ) : i(i) { } const Iter &i; };

	/**
	 * \brief Double List Iterator. 
	 * \ingroup iterators
	 */
	struct Iter
	{
		/* Default construct. */
		Iter() : ptr(0) { }

		/* Construct from a double list. */
		Iter( const DList &dl )      : ptr(dl.head) { }
		Iter( Element *el )          : ptr(el) { }
		Iter( const IterFirst &dlf ) : ptr(dlf.l.head) { }
		Iter( const IterLast &dll )  : ptr(dll.l.tail) { }
		Iter( const IterNext &dln )  : ptr(dln.i.ptr->BASE_EL(next)) { }
		Iter( const IterPrev &dlp )  : ptr(dlp.i.ptr->BASE_EL(prev)) { }

		/* Assign from a double list. */
		Iter &operator=( const DList &dl )     { ptr = dl.head; return *this; }
		Iter &operator=( Element *el )         { ptr = el; return *this; }
		Iter &operator=( const IterFirst &af ) { ptr = af.l.head; return *this; }
		Iter &operator=( const IterLast &al )  { ptr = al.l.tail; return *this; }
		Iter &operator=( const IterNext &an )  { ptr = an.i.ptr->BASE_EL(next); return *this; }
		Iter &operator=( const IterPrev &ap )  { ptr = ap.i.ptr->BASE_EL(prev); return *this; }

		/** \brief Less than end? */
		bool lte() const    { return ptr != 0; }

		/** \brief At end? */
		bool end() const    { return ptr == 0; }

		/** \brief Greater than beginning? */
		bool gtb() const { return ptr != 0; }

		/** \brief At beginning? */
		bool beg() const { return ptr == 0; }

		/** \brief At first element? */
		bool first() const { return ptr && ptr->BASE_EL(prev) == 0; }

		/** \brief At last element? */
		bool last() const  { return ptr && ptr->BASE_EL(next) == 0; }

		/** \brief Implicit cast to Element*. */
		operator Element*() const   { return ptr; }

		/** \brief Dereference operator returns Element&. */
		Element &operator *() const { return *ptr; }

		/** \brief Arrow operator returns Element*. */
		Element *operator->() const { return ptr; }

		/** \brief Move to next item. */
		inline Element *operator++()      { return ptr = ptr->BASE_EL(next); }

		/** \brief Move to next item. */
		inline Element *increment()       { return ptr = ptr->BASE_EL(next); }

		/** \brief Move to next item. */
		inline Element *operator++(int);

		/** \brief Move to previous item. */
		inline Element *operator--()      { return ptr = ptr->BASE_EL(prev); }

		/** \brief Move to previous item. */
		inline Element *decrement()       { return ptr = ptr->BASE_EL(prev); }

		/** \brief Move to previous item. */
		inline Element *operator--(int);

		/** \brief Return the next item. Does not modify this. */
		inline IterNext next() const { return IterNext(*this); }

		/** \brief Return the prev item. Does not modify this. */
		inline IterPrev prev() const { return IterPrev(*this); }

		/** \brief The iterator is simply a pointer. */
		Element *ptr;
	};

	/** \brief Return first element. */
	IterFirst first()  { return IterFirst(*this); }

	/** \brief Return last element. */
	IterLast last()    { return IterLast(*this); }
};

/* Copy constructor, does a deep copy of other. */
template <DLMEL_TEMPDEF> DList<DLMEL_TEMPUSE>::
		DList(const DList<DLMEL_TEMPUSE> &other) :
			head(0), tail(0), listLen(0)
{
	Element *el = other.head;
	while( el != 0 ) {
		append( new Element(*el) );
		el = el->BASE_EL(next);
	}
}

#ifdef DOUBLELIST_VALUE

/* Assignement operator does deep copy. */
template <DLMEL_TEMPDEF> DList<DLMEL_TEMPUSE> &DList<DLMEL_TEMPUSE>::
		operator=(const DList &other)
{
	/* Free the old list. The value list assumes items were allocated on the
	 * heap by itself. */
	empty();

	Element *el = other.head;
	while( el != 0 ) {
		append( new Element(*el) );
		el = el->BASE_EL(next);
	}
	return *this;
}

template <DLMEL_TEMPDEF> void DList<DLMEL_TEMPUSE>::
		transfer(DList &other)
{
	/* Free the old list. The value list assumes items were allocated on the
	 * heap by itself. */
	empty();

	head = other.head;
	tail = other.tail;
	listLen = other.listLen;

	other.abandon();
}

#else 

/* Assignement operator does deep copy. */
template <DLMEL_TEMPDEF> DList<DLMEL_TEMPUSE> &DList<DLMEL_TEMPUSE>::
		operator=(const DList &other)
{
	Element *el = other.head;
	while( el != 0 ) {
		append( new Element(*el) );
		el = el->BASE_EL(next);
	}
	return *this;
}

template <DLMEL_TEMPDEF> void DList<DLMEL_TEMPUSE>::
		transfer(DList &other)
{
	head = other.head;
	tail = other.tail;
	listLen = other.listLen;

	other.abandon();
}

#endif

#ifdef DOUBLELIST_VALUE

/* Prepend a new item. Inlining this bloats the caller with new overhead. */
template <DLMEL_TEMPDEF> void DList<DLMEL_TEMPUSE>::
		prepend(const T &item)
{
	addBefore(head, new Element(item)); 
}

/* Append a new item. Inlining this bloats the caller with the new overhead. */
template <DLMEL_TEMPDEF> void DList<DLMEL_TEMPUSE>::
		append(const T &item)
{
	addAfter(tail, new Element(item));
}

/* Add a new item after a prev element. Inlining this bloats the caller with
 * the new overhead. */
template <DLMEL_TEMPDEF> void DList<DLMEL_TEMPUSE>::
		addAfter(Element *prev_el, const T &item)
{
	addAfter(prev_el, new Element(item));
}

/* Add a new item before a next element. Inlining this bloats the caller with
 * the new overhead. */
template <DLMEL_TEMPDEF> void DList<DLMEL_TEMPUSE>::
		addBefore(Element *next_el, const T &item)
{
	addBefore(next_el, new Element(item));
}

#endif

/*
 * The larger iterator operators.
 */

/* Postfix ++ */
template <DLMEL_TEMPDEF> Element *DList<DLMEL_TEMPUSE>::Iter::
		operator++(int)       
{
	Element *rtn = ptr; 
	ptr = ptr->BASE_EL(next);
	return rtn;
}

/* Postfix -- */
template <DLMEL_TEMPDEF> Element *DList<DLMEL_TEMPUSE>::Iter::
		operator--(int)       
{
	Element *rtn = ptr;
	ptr = ptr->BASE_EL(prev);
	return rtn;
}

/**
 * \brief Insert an element immediately after an element in the list.
 *
 * If prev_el is NULL then new_el is prepended to the front of the list. If
 * prev_el is not in the list or if new_el is already in a list, then
 * undefined behaviour results.
 */
template <DLMEL_TEMPDEF> void DList<DLMEL_TEMPUSE>::
		addAfter(Element *prev_el, Element *new_el)
{
	/* Set the previous pointer of new_el to prev_el. We do
	 * this regardless of the state of the list. */
	new_el->BASE_EL(prev) = prev_el; 

	/* Set forward pointers. */
	if (prev_el == 0) {
		/* There was no prev_el, we are inserting at the head. */
		new_el->BASE_EL(next) = head;
		head = new_el;
	} 
	else {
		/* There was a prev_el, we can access previous next. */
		new_el->BASE_EL(next) = prev_el->BASE_EL(next);
		prev_el->BASE_EL(next) = new_el;
	} 

	/* Set reverse pointers. */
	if (new_el->BASE_EL(next) == 0) {
		/* There is no next element. Set the tail pointer. */
		tail = new_el;
	}
	else {
		/* There is a next element. Set it's prev pointer. */
		new_el->BASE_EL(next)->BASE_EL(prev) = new_el;
	}

	/* Update list length. */
	listLen++;
}

/**
 * \brief Insert an element immediatly before an element in the list.
 *
 * If next_el is NULL then new_el is appended to the end of the list. If
 * next_el is not in the list or if new_el is already in a list, then
 * undefined behaviour results.
 */
template <DLMEL_TEMPDEF> void DList<DLMEL_TEMPUSE>::
		addBefore(Element *next_el, Element *new_el)
{
	/* Set the next pointer of the new element to next_el. We do
	 * this regardless of the state of the list. */
	new_el->BASE_EL(next) = next_el; 

	/* Set reverse pointers. */
	if (next_el == 0) {
		/* There is no next elememnt. We are inserting at the tail. */
		new_el->BASE_EL(prev) = tail;
		tail = new_el;
	} 
	else {
		/* There is a next element and we can access next's previous. */
		new_el->BASE_EL(prev) = next_el->BASE_EL(prev);
		next_el->BASE_EL(prev) = new_el;
	} 

	/* Set forward pointers. */
	if (new_el->BASE_EL(prev) == 0) {
		/* There is no previous element. Set the head pointer.*/
		head = new_el;
	}
	else {
		/* There is a previous element, set it's next pointer to new_el. */
		new_el->BASE_EL(prev)->BASE_EL(next) = new_el;
	}

	/* Update list length. */
	listLen++;
}

/**
 * \brief Insert an entire list immediatly after an element in this list.
 *
 * Elements are moved, not copied. Afterwards, the other list is empty. If
 * prev_el is NULL then the elements are prepended to the front of the list.
 * If prev_el is not in the list then undefined behaviour results. All
 * elements are inserted into the list at once, so this is an O(1) operation.
 */
template <DLMEL_TEMPDEF> void DList<DLMEL_TEMPUSE>::
		addAfter( Element *prev_el, DList<DLMEL_TEMPUSE> &dl )
{
	/* Do not bother if dl has no elements. */
	if ( dl.listLen == 0 )
		return;

	/* Set the previous pointer of dl.head to prev_el. We do
	 * this regardless of the state of the list. */
	dl.head->BASE_EL(prev) = prev_el; 

	/* Set forward pointers. */
	if (prev_el == 0) {
		/* There was no prev_el, we are inserting at the head. */
		dl.tail->BASE_EL(next) = head;
		head = dl.head;
	} 
	else {
		/* There was a prev_el, we can access previous next. */
		dl.tail->BASE_EL(next) = prev_el->BASE_EL(next);
		prev_el->BASE_EL(next) = dl.head;
	} 

	/* Set reverse pointers. */
	if (dl.tail->BASE_EL(next) == 0) {
		/* There is no next element. Set the tail pointer. */
		tail = dl.tail;
	}
	else {
		/* There is a next element. Set it's prev pointer. */
		dl.tail->BASE_EL(next)->BASE_EL(prev) = dl.tail;
	}

	/* Update the list length. */
	listLen += dl.listLen;

	/* Empty out dl. */
	dl.head = dl.tail = 0;
	dl.listLen = 0;
}

/**
 * \brief Insert an entire list immediately before an element in this list.
 *
 * Elements are moved, not copied. Afterwards, the other list is empty. If
 * next_el is NULL then the elements are appended to the end of the list. If
 * next_el is not in the list then undefined behaviour results. All elements
 * are inserted at once, so this is an O(1) operation.
 */
template <DLMEL_TEMPDEF> void DList<DLMEL_TEMPUSE>::
		addBefore( Element *next_el, DList<DLMEL_TEMPUSE> &dl )
{
	/* Do not bother if dl has no elements. */
	if ( dl.listLen == 0 )
		return;

	/* Set the next pointer of dl.tail to next_el. We do
	 * this regardless of the state of the list. */
	dl.tail->BASE_EL(next) = next_el; 

	/* Set reverse pointers. */
	if (next_el == 0) {
		/* There is no next elememnt. We are inserting at the tail. */
		dl.head->BASE_EL(prev) = tail;
		tail = dl.tail;
	} 
	else {
		/* There is a next element and we can access next's previous. */
		dl.head->BASE_EL(prev) = next_el->BASE_EL(prev);
		next_el->BASE_EL(prev) = dl.tail;
	} 

	/* Set forward pointers. */
	if (dl.head->BASE_EL(prev) == 0) {
		/* There is no previous element. Set the head pointer.*/
		head = dl.head;
	}
	else {
		/* There is a previous element, set it's next pointer to new_el. */
		dl.head->BASE_EL(prev)->BASE_EL(next) = dl.head;
	}

	/* Update list length. */
	listLen += dl.listLen;

	/* Empty out dl. */
	dl.head = dl.tail = 0;
	dl.listLen = 0;
}


/**
 * \brief Detach an element from the list.
 *
 * The element is not deleted. If the element is not in the list, then
 * undefined behaviour results.
 *
 * \returns The element detached.
 */
template <DLMEL_TEMPDEF> Element *DList<DLMEL_TEMPUSE>::
		detach(Element *el)
{
	/* Set forward pointers to skip over el. */
	if (el->BASE_EL(prev) == 0) 
		head = el->BASE_EL(next); 
	else {
		el->BASE_EL(prev)->BASE_EL(next) =
				el->BASE_EL(next); 
	}

	/* Set reverse pointers to skip over el. */
	if (el->BASE_EL(next) == 0) 
		tail = el->BASE_EL(prev); 
	else {
		el->BASE_EL(next)->BASE_EL(prev) =
				el->BASE_EL(prev); 
	}

	/* Update List length and return element we detached. */
	listLen--;
	return el;
}

/**
 * \brief Clear the list by deleting all elements.
 *
 * Each item in the list is deleted. The list is reset to its initial state.
 */
template <DLMEL_TEMPDEF> void DList<DLMEL_TEMPUSE>::empty()
{
	Element *nextToGo = 0, *cur = head;
	
	while (cur != 0)
	{
		nextToGo = cur->BASE_EL(next);
		delete cur;
		cur = nextToGo;
	}
	head = tail = 0;
	listLen = 0;
}

/**
 * \brief Clear the list by forgetting all elements.
 *
 * All elements are abandoned, not deleted. The list is reset to it's initial
 * state.
 */
template <DLMEL_TEMPDEF> void DList<DLMEL_TEMPUSE>::abandon()
{
	head = tail = 0;
	listLen = 0;
}

#ifdef AAPL_NAMESPACE
}
#endif
