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

#include <assert.h>

#ifdef AAPL_NAMESPACE
namespace Aapl {
#endif

#ifdef WALKABLE
/* This is used by AvlTree, AvlMel and AvlMelKey so it
 * must be protected by global ifdefs. */
#ifndef __AAPL_AVLI_EL__
#define __AAPL_AVLI_EL__

/**
 * \brief Tree element properties for linked AVL trees.
 *
 * AvliTreeEl needs to be inherited by classes that intend to be element in an
 * AvliTree.
 */
template<class SubClassEl> struct AvliTreeEl
{
	/**
	 * \brief Tree pointers connecting element in a tree.
	 */
	SubClassEl *left, *right, *parent;

	/**
	 * \brief Linked list pointers.
	 */
	SubClassEl *prev, *next;

	/**
	 * \brief Height of the tree rooted at this element.
	 *
	 * Height is required by the AVL balancing algorithm.
	 */
	long height;
};
#endif /* __AAPL_AVLI_EL__ */

#else /* not WALKABLE */

/* This is used by All the non walkable trees so it must be
 * protected by a global ifdef. */
#ifndef __AAPL_AVL_EL__
#define __AAPL_AVL_EL__
/**
 * \brief Tree element properties for linked AVL trees.
 *
 * AvlTreeEl needs to be inherited by classes that intend to be element in an
 * AvlTree.
 */
template<class SubClassEl> struct AvlTreeEl
{
	/**
	 * \brief Tree pointers connecting element in a tree.
	 */
	SubClassEl *left, *right, *parent;

	/**
	 * \brief Height of the tree rooted at this element.
	 *
	 * Height is required by the AVL balancing algorithm.
	 */
	long height;
};
#endif /* __AAPL_AVL_EL__ */
#endif /* def WALKABLE */


#if defined( AVLTREE_MAP )

#ifdef WALKABLE

/**
 * \brief Tree element for AvliMap
 *
 * Stores the key and value pair.
 */
template <class Key, class Value> struct AvliMapEl :
		public AvliTreeEl< AvliMapEl<Key, Value> >
{
	AvliMapEl(const Key &key)
		: key(key) { }
	AvliMapEl(const Key &key, const Value &value)
		: key(key), value(value) { }

	const Key &getKey() const { return key; }

	/** \brief The key. */
	Key key;

	/** \brief The value. */
	Value value;
};
#else /* not WALKABLE */

/**
 * \brief Tree element for AvlMap
 *
 * Stores the key and value pair.
 */
template <class Key, class Value> struct AvlMapEl :
		public AvlTreeEl< AvlMapEl<Key, Value> >
{
	AvlMapEl(const Key &key)
		: key(key) { }
	AvlMapEl(const Key &key, const Value &value)
		: key(key), value(value) { }

	const Key &getKey() const { return key; }

	/** \brief The key. */
	Key key;

	/** \brief The value. */
	Value value;
};
#endif /* def WALKABLE */

#elif defined( AVLTREE_SET )

#ifdef WALKABLE
/**
 * \brief Tree element for AvliSet
 *
 * Stores the key.
 */
template <class Key> struct AvliSetEl :
		public AvliTreeEl< AvliSetEl<Key> >
{
	AvliSetEl(const Key &key) : key(key) { }

	const Key &getKey() const { return key; }

	/** \brief The key. */
	Key key;
};
#else /* not WALKABLE */
/**
 * \brief Tree element for AvlSet
 *
 * Stores the key.
 */
template <class Key> struct AvlSetEl :
		public AvlTreeEl< AvlSetEl<Key> >
{
	AvlSetEl(const Key &key) : key(key) { }

	const Key &getKey() const { return key; }

	/** \brief The key. */
	Key key;
};
#endif /* def WALKABLE */

#endif /* AVLTREE_SET */

/* Common AvlTree Class */
template < AVLMEL_CLASSDEF > class AvlTree
#if !defined( AVL_KEYLESS ) && defined ( WALKABLE )
		: public Compare, public BASELIST
#elif !defined( AVL_KEYLESS )
		: public Compare
#elif defined( WALKABLE )
		: public BASELIST
#endif
{
public:
	/**
	 * \brief Create an empty tree.
	 */
#ifdef WALKABLE
	AvlTree() : root(0), treeSize(0) { }
#else
	AvlTree() : root(0), head(0), tail(0), treeSize(0) { }
#endif

	/**
	 * \brief Perform a deep copy of the tree.
	 *
	 * Each element is duplicated for the new tree. Copy constructors are used
	 * to create the new elements.
	 */
	AvlTree(const AvlTree &other);

#if defined( AVLTREE_MAP ) || defined( AVLTREE_SET )
	/**
	 * \brief Clear the contents of the tree.
	 *
	 * All element are deleted.
	 */
	~AvlTree() { empty(); }

	/**
	 * \brief Perform a deep copy of the tree.
	 *
	 * Each element is duplicated for the new tree. Copy constructors are used
	 * to create the new element. If this tree contains items, they are first
	 * deleted.
	 *
	 * \returns A reference to this.
	 */
	AvlTree &operator=( const AvlTree &tree );

	/**
	 * \brief Transfer the elements of another tree into this.
	 *
	 * First deletes all elements in this tree.
	 */
	void transfer( AvlTree &tree );
#else
	/**
	 * \brief Abandon all elements in the tree.
	 *
	 * Tree elements are not deleted.
	 */
	~AvlTree() {}

	/**
	 * \brief Perform a deep copy of the tree.
	 *
	 * Each element is duplicated for the new tree. Copy constructors are used
	 * to create the new element. If this tree contains items, they are
	 * abandoned.
	 *
	 * \returns A reference to this.
	 */
	AvlTree &operator=( const AvlTree &tree );

	/**
	 * \brief Transfer the elements of another tree into this.
	 *
	 * All elements in this tree are abandoned first.
	 */
	void transfer( AvlTree &tree );
#endif

#ifndef AVL_KEYLESS
	/* Insert a element into the tree. */
	Element *insert( Element *element, Element **lastFound = 0 );

#ifdef AVL_BASIC
	/* Find a element in the tree. Returns the element if
	 * element exists, false otherwise. */
	Element *find( const Element *element ) const;

#else
	Element *insert( const Key &key, Element **lastFound = 0 );

#ifdef AVLTREE_MAP
	Element *insert( const Key &key, const Value &val,
			Element **lastFound = 0 );
#endif

	/* Find a element in the tree. Returns the element if
	 * key exists, false otherwise. */
	Element *find( const Key &key ) const;

	/* Detach a element from the tree. */
	Element *detach( const Key &key );

	/* Detach and delete a element from the tree. */
	bool remove( const Key &key );
#endif /* AVL_BASIC */
#endif /* AVL_KEYLESS */

	/* Detach a element from the tree. */
	Element *detach( Element *element );

	/* Detach and delete a element from the tree. */
	void remove( Element *element );

	/* Free all memory used by tree. */
	void empty();

	/* Abandon all element in the tree. Does not delete element. */
	void abandon();

	/** Root element of the tree. */
	Element *root;

#ifndef WALKABLE
	Element *head, *tail;
#endif

	/** The number of element in the tree. */
	long treeSize;

	/** \brief Return the number of elements in the tree. */
	long length() const         { return treeSize; }

	/** \brief Return the number of elements in the tree. */
	long size() const           { return treeSize; }

	/* Various classes for setting the iterator */
	struct Iter;
	struct IterFirst { IterFirst( const AvlTree &t ) : t(t) { } const AvlTree &t; };
	struct IterLast { IterLast( const AvlTree &t ) : t(t) { } const AvlTree &t; };
	struct IterNext { IterNext( const Iter &i ) : i(i) { } const Iter &i; };
	struct IterPrev { IterPrev( const Iter &i ) : i(i) { } const Iter &i; };

#ifdef WALKABLE
	/**
	 * \brief Avl Tree Iterator.
	 * \ingroup iterators
	 */
	struct Iter
	{
		/* Default construct. */
		Iter() : ptr(0) { }

		/* Construct from an avl tree and iterator-setting classes. */
		Iter( const AvlTree &t ) : ptr(t.head) { }
		Iter( const IterFirst &af ) : ptr(af.t.head) { }
		Iter( const IterLast &al ) : ptr(al.t.tail) { }
		Iter( const IterNext &an ) : ptr(findNext(an.i.ptr)) { }
		Iter( const IterPrev &ap ) : ptr(findPrev(ap.i.ptr)) { }

		/* Assign from a tree and iterator-setting classes. */
		Iter &operator=( const AvlTree &tree ) { ptr = tree.head; return *this; }
		Iter &operator=( const IterFirst &af ) { ptr = af.t.head; return *this; }
		Iter &operator=( const IterLast &al )  { ptr = al.t.tail; return *this; }
		Iter &operator=( const IterNext &an )  { ptr = findNext(an.i.ptr); return *this; }
		Iter &operator=( const IterPrev &ap )  { ptr = findPrev(ap.i.ptr); return *this; }

		/** \brief Less than end? */
		bool lte() const { return ptr != 0; }

		/** \brief At end? */
		bool end() const { return ptr == 0; }

		/** \brief Greater than beginning? */
		bool gtb() const { return ptr != 0; }

		/** \brief At beginning? */
		bool beg() const { return ptr == 0; }

		/** \brief At first element? */
		bool first() const { return ptr && ptr->BASE_EL(prev) == 0; }

		/** \brief At last element? */
		bool last() const { return ptr && ptr->BASE_EL(next) == 0; }

		/** \brief Implicit cast to Element*. */
		operator Element*() const      { return ptr; }

		/** \brief Dereference operator returns Element&. */
		Element &operator *() const    { return *ptr; }

		/** \brief Arrow operator returns Element*. */
		Element *operator->() const    { return ptr; }

		/** \brief Move to next item. */
		inline Element *operator++();

		/** \brief Move to next item. */
		inline Element *operator++(int);

		/** \brief Move to next item. */
		inline Element *increment();

		/** \brief Move to previous item. */
		inline Element *operator--();

		/** \brief Move to previous item. */
		inline Element *operator--(int);

		/** \brief Move to previous item. */
		inline Element *decrement();

		/** \brief Return the next item. Does not modify this. */
		IterNext next() const { return IterNext( *this ); }

		/** \brief Return the previous item. Does not modify this. */
		IterPrev prev() const { return IterPrev( *this ); }

	private:
		static Element *findPrev( Element *element ) { return element->BASE_EL(prev); }
		static Element *findNext( Element *element ) { return element->BASE_EL(next); }

	public:

		/** \brief The iterator is simply a pointer. */
		Element *ptr;
	};

#else

	/**
	 * \brief Avl Tree Iterator.
	 * \ingroup iterators
	 */
	struct Iter
	{
		/* Default construct. */
		Iter() : ptr(0), tree(0) { }

		/* Construct from a tree and iterator-setting classes. */
		Iter( const AvlTree &t ) : ptr(t.head), tree(&t) { }
		Iter( const IterFirst &af ) : ptr(af.t.head), tree(&af.t) { }
		Iter( const IterLast &al ) : ptr(al.t.tail), tree(&al.t) { }
		Iter( const IterNext &an ) : ptr(findNext(an.i.ptr)), tree(an.i.tree) { }
		Iter( const IterPrev &ap ) : ptr(findPrev(ap.i.ptr)), tree(ap.i.tree) { }

		/* Assign from a tree and iterator-setting classes. */
		Iter &operator=( const AvlTree &t )
				{ ptr = t.head; tree = &t; return *this; }
		Iter &operator=( const IterFirst &af )
				{ ptr = af.t.head; tree = &af.t; return *this; }
		Iter &operator=( const IterLast &al )
				{ ptr = al.t.tail; tree = &al.t; return *this; }
		Iter &operator=( const IterNext &an )
				{ ptr = findNext(an.i.ptr); tree = an.i.tree; return *this; }
		Iter &operator=( const IterPrev &ap )
				{ ptr = findPrev(ap.i.ptr); tree = ap.i.tree; return *this; }

		/** \brief Less than end? */
		bool lte() const { return ptr != 0; }

		/** \brief At end? */
		bool end() const { return ptr == 0; }

		/** \brief Greater than beginning? */
		bool gtb() const { return ptr != 0; }

		/** \brief At beginning? */
		bool beg() const { return ptr == 0; }

		/** \brief At first element? */
		bool first() const { return ptr && ptr == tree->head; }

		/** \brief At last element? */
		bool last() const { return ptr && ptr == tree->tail; }

		/** \brief Implicit cast to Element*. */
		operator Element*() const      { return ptr; }

		/** \brief Dereference operator returns Element&. */
		Element &operator *() const    { return *ptr; }

		/** \brief Arrow operator returns Element*. */
		Element *operator->() const    { return ptr; }

		/** \brief Move to next item. */
		inline Element *operator++();

		/** \brief Move to next item. */
		inline Element *operator++(int);

		/** \brief Move to next item. */
		inline Element *increment();

		/** \brief Move to previous item. */
		inline Element *operator--();

		/** \brief Move to previous item. */
		inline Element *operator--(int);

		/** \brief Move to previous item. */
		inline Element *decrement();

		/** \brief Return the next item. Does not modify this. */
		IterNext next() const { return IterNext( *this ); }

		/** \brief Return the previous item. Does not modify this. */
		IterPrev prev() const { return IterPrev( *this ); }

	private:
		static Element *findPrev( Element *element );
		static Element *findNext( Element *element );

	public:
		/** \brief The iterator is simply a pointer. */
		Element *ptr;

		/* The list is not walkable so we need to keep a pointerto the tree
		 * so we can test against head and tail in O(1) time. */
		const AvlTree *tree;
	};
#endif

	/** \brief Return first element. */
	IterFirst first()  { return IterFirst( *this ); }

	/** \brief Return last element. */
	IterLast last()    { return IterLast( *this ); }

protected:
	/* Recursive worker for the copy constructor. */
	Element *copyBranch( Element *element );

	/* Recursively delete element in the tree. */
	void deleteChildrenOf(Element *n);

	/* rebalance the tree beginning at the leaf whose
	 * grandparent is unbalanced. */
	Element *rebalance(Element *start);

	/* Move up the tree from a given element, recalculating the heights. */
	void recalcHeights(Element *start);

	/* Move up the tree and find the first element whose
	 * grand-parent is unbalanced. */
	Element *findFirstUnbalGP(Element *start);

	/* Move up the tree and find the first element which is unbalanced. */
	Element *findFirstUnbalEl(Element *start);

	/* Replace a element in the tree with another element not in the tree. */
	void replaceEl(Element *element, Element *replacement);

	/* Remove a element from the tree and put another (normally a child of element)
	 * in its place. */
	void removeEl(Element *element, Element *filler);

	/* Once an insertion point is found at a leaf then do the insert. */
	void attachRebal( Element *element, Element *parentEl, Element *lastLess );
};

/* Copy constructor. New up each item. */
template <AVLMEL_TEMPDEF> AvlTree<AVLMEL_TEMPUSE>::
		AvlTree(const AvlTree<AVLMEL_TEMPUSE> &other)
#ifdef WALKABLE
:
	/* Make an empty list, copyBranch will fill in the details for us. */
	BASELIST()
#endif
{
	treeSize = other.treeSize;
	root = other.root;

#ifndef WALKABLE
	head = 0;
	tail = 0;
#endif

	/* If there is a root, copy the tree. */
	if ( other.root != 0 )
		root = copyBranch( other.root );
}

#if defined( AVLTREE_MAP ) || defined( AVLTREE_SET )

/* Assignment does deep copy. */
template <AVLMEL_TEMPDEF> AvlTree<AVLMEL_TEMPUSE> &AvlTree<AVLMEL_TEMPUSE>::
	operator=( const AvlTree &other )
{
	/* Clear the tree first. */
	empty();

	/* Reset the list pointers, the tree copy will fill in the list for us. */
#ifdef WALKABLE
	BASELIST::abandon();
#else
	head = 0;
	tail = 0;
#endif

	/* Copy the entire tree. */
	treeSize = other.treeSize;
	root = other.root;
	if ( other.root != 0 )
		root = copyBranch( other.root );
	return *this;
}

template <AVLMEL_TEMPDEF> void AvlTree<AVLMEL_TEMPUSE>::
		transfer(AvlTree<AVLMEL_TEMPUSE> &other)
{
	/* Clear the tree first. */
	empty();

	treeSize = other.treeSize;
	root = other.root;

#ifdef WALKABLE
	BASELIST::head = other.BASELIST::head;
	BASELIST::tail = other.BASELIST::tail;
	BASELIST::listLen = other.BASELIST::listLen;
#else
	head = other.head;
	tail = other.tail;
#endif

	other.abandon();
}

#else /* ! AVLTREE_MAP && ! AVLTREE_SET */

/* Assignment does deep copy. This version does not clear the tree first. */
template <AVLMEL_TEMPDEF> AvlTree<AVLMEL_TEMPUSE> &AvlTree<AVLMEL_TEMPUSE>::
	operator=( const AvlTree &other )
{
	/* Reset the list pointers, the tree copy will fill in the list for us. */
#ifdef WALKABLE
	BASELIST::abandon();
#else
	head = 0;
	tail = 0;
#endif

	/* Copy the entire tree. */
	treeSize = other.treeSize;
	root = other.root;
	if ( other.root != 0 )
		root = copyBranch( other.root );
	return *this;
}

template <AVLMEL_TEMPDEF> void AvlTree<AVLMEL_TEMPUSE>::
		transfer(AvlTree<AVLMEL_TEMPUSE> &other)
{
	treeSize = other.treeSize;
	root = other.root;

#ifdef WALKABLE
	BASELIST::head = other.BASELIST::head;
	BASELIST::tail = other.BASELIST::tail;
	BASELIST::listLen = other.BASELIST::listLen;
#else
	head = other.head;
	tail = other.tail;
#endif

	other.abandon();
}

#endif

/*
 * Iterator operators.
 */

/* Prefix ++ */
template <AVLMEL_TEMPDEF> Element *AvlTree<AVLMEL_TEMPUSE>::Iter::
		operator++()
{
	return ptr = findNext( ptr );
}

/* Postfix ++ */
template <AVLMEL_TEMPDEF> Element *AvlTree<AVLMEL_TEMPUSE>::Iter::
		operator++(int)
{
	Element *rtn = ptr;
	ptr = findNext( ptr );
	return rtn;
}

/* increment */
template <AVLMEL_TEMPDEF> Element *AvlTree<AVLMEL_TEMPUSE>::Iter::
		increment()
{
	return ptr = findNext( ptr );
}

/* Prefix -- */
template <AVLMEL_TEMPDEF> Element *AvlTree<AVLMEL_TEMPUSE>::Iter::
		operator--()
{
	return ptr = findPrev( ptr );
}

/* Postfix -- */
template <AVLMEL_TEMPDEF> Element *AvlTree<AVLMEL_TEMPUSE>::Iter::
		operator--(int)
{
	Element *rtn = ptr;
	ptr = findPrev( ptr );
	return rtn;
}

/* decrement */
template <AVLMEL_TEMPDEF> Element *AvlTree<AVLMEL_TEMPUSE>::Iter::
		decrement()
{
	return ptr = findPrev( ptr );
}

#ifndef WALKABLE

/* Move ahead one. */
template <AVLMEL_TEMPDEF> Element *AvlTree<AVLMEL_TEMPUSE>::Iter::
		findNext( Element *element )
{
	/* Try to go right once then infinite left. */
	if ( element->BASE_EL(right) != 0 ) {
		element = element->BASE_EL(right);
		while ( element->BASE_EL(left) != 0 )
			element = element->BASE_EL(left);
	}
	else {
		/* Go up to parent until we were just a left child. */
		while ( true ) {
			Element *last = element;
			element = element->BASE_EL(parent);
			if ( element == 0 || element->BASE_EL(left) == last )
				break;
		}
	}
	return element;
}

/* Move back one. */
template <AVLMEL_TEMPDEF> Element *AvlTree<AVLMEL_TEMPUSE>::Iter::
		findPrev( Element *element )
{
	/* Try to go left once then infinite right. */
	if ( element->BASE_EL(left) != 0 ) {
		element = element->BASE_EL(left);
		while ( element->BASE_EL(right) != 0 )
			element = element->BASE_EL(right);
	}
	else {
		/* Go up to parent until we were just a left child. */
		while ( true ) {
			Element *last = element;
			element = element->BASE_EL(parent);
			if ( element == 0 || element->BASE_EL(right) == last )
				break;
		}
	}
	return element;
}

#endif


/* Recursive worker for tree copying. */
template <AVLMEL_TEMPDEF> Element *AvlTree<AVLMEL_TEMPUSE>::
		copyBranch( Element *element )
{
	/* Duplicate element. Either the base element's copy constructor or defaul
	 * constructor will get called. Both will suffice for initting the
	 * pointers to null when they need to be. */
	Element *retVal = new Element(*element);

	/* If the left tree is there, copy it. */
	if ( retVal->BASE_EL(left) ) {
		retVal->BASE_EL(left) = copyBranch(retVal->BASE_EL(left));
		retVal->BASE_EL(left)->BASE_EL(parent) = retVal;
	}

#ifdef WALKABLE
	BASELIST::addAfter( BASELIST::tail, retVal );
#else
	if ( head == 0 )
		head = retVal;
	tail = retVal;
#endif

	/* If the right tree is there, copy it. */
	if ( retVal->BASE_EL(right) ) {
		retVal->BASE_EL(right) = copyBranch(retVal->BASE_EL(right));
		retVal->BASE_EL(right)->BASE_EL(parent) = retVal;
	}
	return retVal;
}

/* Once an insertion position is found, attach a element to the tree. */
template <AVLMEL_TEMPDEF> void AvlTree<AVLMEL_TEMPUSE>::
		attachRebal( Element *element, Element *parentEl, Element *lastLess )
{
	/* Increment the number of element in the tree. */
	treeSize += 1;

	/* Set element's parent. */
	element->BASE_EL(parent) = parentEl;

	/* New element always starts as a leaf with height 1. */
	element->BASE_EL(left) = 0;
	element->BASE_EL(right) = 0;
	element->BASE_EL(height) = 1;

	/* Are we inserting in the tree somewhere? */
	if ( parentEl != 0 ) {
		/* We have a parent so we are somewhere in the tree. If the parent
		 * equals lastLess, then the last traversal in the insertion went
		 * left, otherwise it went right. */
		if ( lastLess == parentEl ) {
			parentEl->BASE_EL(left) = element;
#ifdef WALKABLE
			BASELIST::addBefore( parentEl, element );
#endif
		}
		else {
			parentEl->BASE_EL(right) = element;
#ifdef WALKABLE
			BASELIST::addAfter( parentEl, element );
#endif
		}

#ifndef WALKABLE
		/* Maintain the first and last pointers. */
		if ( head->BASE_EL(left) == element )
			head = element;

		/* Maintain the first and last pointers. */
		if ( tail->BASE_EL(right) == element )
			tail = element;
#endif
	}
	else {
		/* No parent element so we are inserting the root. */
		root = element;
#ifdef WALKABLE
		BASELIST::addAfter( BASELIST::tail, element );
#else
		head = tail = element;
#endif
	}


	/* Recalculate the heights. */
	recalcHeights(parentEl);

	/* Find the first unbalance. */
	Element *ub = findFirstUnbalGP(element);

	/* rebalance. */
	if ( ub != 0 )
	{
		/* We assert that after this single rotation the
		 * tree is now properly balanced. */
		rebalance(ub);
	}
}

#ifndef AVL_KEYLESS

/**
 * \brief Insert an existing element into the tree.
 *
 * If the insert succeeds and lastFound is given then it is set to the element
 * inserted. If the insert fails then lastFound is set to the existing element in
 * the tree that has the same key as element. If the element's avl pointers are
 * already in use then undefined behaviour results.
 *
 * \returns The element inserted upon success, null upon failure.
 */
template <AVLMEL_TEMPDEF> Element *AvlTree<AVLMEL_TEMPUSE>::
		insert( Element *element, Element **lastFound )
{
	long keyRelation;
	Element *curEl = root, *parentEl = 0;
	Element *lastLess = 0;

	while (true) {
		if ( curEl == 0 ) {
			/* We are at an external element and did not find the key we were
			 * looking for. Attach underneath the leaf and rebalance. */
			attachRebal( element, parentEl, lastLess );

			if ( lastFound != 0 )
				*lastFound = element;
			return element;
		}

#ifdef AVL_BASIC
		keyRelation = this->compare( *element, *curEl );
#else
		keyRelation = this->compare( element->BASEKEY(getKey()),
				curEl->BASEKEY(getKey()) );
#endif

		/* Do we go left? */
		if ( keyRelation < 0 ) {
			parentEl = lastLess = curEl;
			curEl = curEl->BASE_EL(left);
		}
		/* Do we go right? */
		else if ( keyRelation > 0 ) {
			parentEl = curEl;
			curEl = curEl->BASE_EL(right);
		}
		/* We have hit the target. */
		else {
			if ( lastFound != 0 )
				*lastFound = curEl;
			return 0;
		}
	}
}

#ifdef AVL_BASIC

/**
 * \brief Find a element in the tree with the given key.
 *
 * \returns The element if key exists, null if the key does not exist.
 */
template <AVLMEL_TEMPDEF> Element *AvlTree<AVLMEL_TEMPUSE>::
		find( const Element *element ) const
{
	Element *curEl = root;
	long keyRelation;

	while (curEl) {
		keyRelation = this->compare( *element, *curEl );

		/* Do we go left? */
		if ( keyRelation < 0 )
			curEl = curEl->BASE_EL(left);
		/* Do we go right? */
		else if ( keyRelation > 0 )
			curEl = curEl->BASE_EL(right);
		/* We have hit the target. */
		else {
			return curEl;
		}
	}
	return 0;
}

#else

/**
 * \brief Insert a new element into the tree with given key.
 *
 * If the key is not already in the tree then a new element is made using the
 * Element(const Key &key) constructor and the insert succeeds. If lastFound is
 * given then it is set to the element inserted. If the insert fails then
 * lastFound is set to the existing element in the tree that has the same key as
 * element.
 *
 * \returns The new element upon success, null upon failure.
 */
template <AVLMEL_TEMPDEF> Element *AvlTree<AVLMEL_TEMPUSE>::
		insert( const Key &key, Element **lastFound )
{
	long keyRelation;
	Element *curEl = root, *parentEl = 0;
	Element *lastLess = 0;

	while (true) {
		if ( curEl == 0 ) {
			/* We are at an external element and did not find the key we were
			 * looking for. Create the new element, attach it underneath the leaf
			 * and rebalance. */
			Element *element = new Element( key );
			attachRebal( element, parentEl, lastLess );

			if ( lastFound != 0 )
				*lastFound = element;
			return element;
		}

		keyRelation = this->compare( key, curEl->BASEKEY(getKey()) );

		/* Do we go left? */
		if ( keyRelation < 0 ) {
			parentEl = lastLess = curEl;
			curEl = curEl->BASE_EL(left);
		}
		/* Do we go right? */
		else if ( keyRelation > 0 ) {
			parentEl = curEl;
			curEl = curEl->BASE_EL(right);
		}
		/* We have hit the target. */
		else {
			if ( lastFound != 0 )
				*lastFound = curEl;
			return 0;
		}
	}
}

#ifdef AVLTREE_MAP
/**
 * \brief Insert a new element into the tree with key and value.
 *
 * If the key is not already in the tree then a new element is constructed and
 * the insert succeeds. If lastFound is given then it is set to the element
 * inserted. If the insert fails then lastFound is set to the existing element in
 * the tree that has the same key as element. This insert routine is only
 * available in AvlMap because it is the only class that knows about a Value
 * type.
 *
 * \returns The new element upon success, null upon failure.
 */
template <AVLMEL_TEMPDEF> Element *AvlTree<AVLMEL_TEMPUSE>::
		insert( const Key &key, const Value &val, Element **lastFound )
{
	long keyRelation;
	Element *curEl = root, *parentEl = 0;
	Element *lastLess = 0;

	while (true) {
		if ( curEl == 0 ) {
			/* We are at an external element and did not find the key we were
			 * looking for. Create the new element, attach it underneath the leaf
			 * and rebalance. */
			Element *element = new Element( key, val );
			attachRebal( element, parentEl, lastLess );

			if ( lastFound != 0 )
				*lastFound = element;
			return element;
		}

		keyRelation = this->compare(key, curEl->getKey());

		/* Do we go left? */
		if ( keyRelation < 0 ) {
			parentEl = lastLess = curEl;
			curEl = curEl->BASE_EL(left);
		}
		/* Do we go right? */
		else if ( keyRelation > 0 ) {
			parentEl = curEl;
			curEl = curEl->BASE_EL(right);
		}
		/* We have hit the target. */
		else {
			if ( lastFound != 0 )
				*lastFound = curEl;
			return 0;
		}
	}
}
#endif /* AVLTREE_MAP */


/**
 * \brief Find a element in the tree with the given key.
 *
 * \returns The element if key exists, null if the key does not exist.
 */
template <AVLMEL_TEMPDEF> Element *AvlTree<AVLMEL_TEMPUSE>::
		find( const Key &key ) const
{
	Element *curEl = root;
	long keyRelation;

	while (curEl) {
		keyRelation = this->compare( key, curEl->BASEKEY(getKey()) );

		/* Do we go left? */
		if ( keyRelation < 0 )
			curEl = curEl->BASE_EL(left);
		/* Do we go right? */
		else if ( keyRelation > 0 )
			curEl = curEl->BASE_EL(right);
		/* We have hit the target. */
		else {
			return curEl;
		}
	}
	return 0;
}


/**
 * \brief Find a element, then detach it from the tree.
 *
 * The element is not deleted.
 *
 * \returns The element detached if the key is found, othewise returns null.
 */
template <AVLMEL_TEMPDEF> Element *AvlTree<AVLMEL_TEMPUSE>::
		detach(const Key &key)
{
	Element *element = find( key );
	if ( element ) {
		detach(element);
	}

	return element;
}

/**
 * \brief Find, detach and delete a element from the tree.
 *
 * \returns True if the element was found and deleted, false otherwise.
 */
template <AVLMEL_TEMPDEF> bool AvlTree<AVLMEL_TEMPUSE>::
		remove(const Key &key)
{
	/* Assume not found. */
	bool retVal = false;

	/* Look for the key. */
	Element *element = find( key );
	if ( element != 0 ) {
		/* If found, detach the element and delete. */
		detach( element );
		delete element;
		retVal = true;
	}

	return retVal;
}

#endif /* AVL_BASIC */
#endif /* AVL_KEYLESS */


/**
 * \brief Detach and delete a element from the tree.
 *
 * If the element is not in the tree then undefined behaviour results.
 */
template <AVLMEL_TEMPDEF> void AvlTree<AVLMEL_TEMPUSE>::
		remove(Element *element)
{
	/* Detach and delete. */
	detach(element);
	delete element;
}

/**
 * \brief Detach a element from the tree.
 *
 * If the element is not in the tree then undefined behaviour results.
 *
 * \returns The element given.
 */
template <AVLMEL_TEMPDEF> Element *AvlTree<AVLMEL_TEMPUSE>::
		detach(Element *element)
{
	Element *replacement, *fixfrom;
	long lheight, rheight;

#ifdef WALKABLE
	/* Remove the element from the ordered list. */
	BASELIST::detach( element );
#endif

	/* Update treeSize. */
	treeSize--;

	/* Find a replacement element. */
	if (element->BASE_EL(right))
	{
		/* Find the leftmost element of the right subtree. */
		replacement = element->BASE_EL(right);
		while (replacement->BASE_EL(left))
			replacement = replacement->BASE_EL(left);

		/* If replacing the element the with its child then we need to start
		 * fixing at the replacement, otherwise we start fixing at the
		 * parent of the replacement. */
		if (replacement->BASE_EL(parent) == element)
			fixfrom = replacement;
		else
			fixfrom = replacement->BASE_EL(parent);

#ifndef WALKABLE
		if ( element == head )
			head = replacement;
#endif

		removeEl(replacement, replacement->BASE_EL(right));
		replaceEl(element, replacement);
	}
	else if (element->BASE_EL(left))
	{
		/* Find the rightmost element of the left subtree. */
		replacement = element->BASE_EL(left);
		while (replacement->BASE_EL(right))
			replacement = replacement->BASE_EL(right);

		/* If replacing the element the with its child then we need to start
		 * fixing at the replacement, otherwise we start fixing at the
		 * parent of the replacement. */
		if (replacement->BASE_EL(parent) == element)
			fixfrom = replacement;
		else
			fixfrom = replacement->BASE_EL(parent);

#ifndef WALKABLE
		if ( element == tail )
			tail = replacement;
#endif

		removeEl(replacement, replacement->BASE_EL(left));
		replaceEl(element, replacement);
	}
	else
	{
		/* We need to start fixing at the parent of the element. */
		fixfrom = element->BASE_EL(parent);

#ifndef WALKABLE
		if ( element == head )
			head = element->BASE_EL(parent);
		if ( element == tail )
			tail = element->BASE_EL(parent);
#endif

		/* The element we are deleting is a leaf element. */
		removeEl(element, 0);
	}

	/* If fixfrom is null it means we just deleted
	 * the root of the tree. */
	if ( fixfrom == 0 )
		return element;

	/* Fix the heights after the deletion. */
	recalcHeights(fixfrom);

	/* Fix every unbalanced element going up in the tree. */
	Element *ub = findFirstUnbalEl(fixfrom);
	while ( ub )
	{
		/* Find the element to rebalance by moving down from the first unbalanced
		 * element 2 levels in the direction of the greatest heights. On the
		 * second move down, the heights may be equal ( but not on the first ).
		 * In which case go in the direction of the first move. */
		lheight = ub->BASE_EL(left) ? ub->BASE_EL(left)->BASE_EL(height) : 0;
		rheight = ub->BASE_EL(right) ? ub->BASE_EL(right)->BASE_EL(height) : 0;
		assert( lheight != rheight );
		if (rheight > lheight)
		{
			ub = ub->BASE_EL(right);
			lheight = ub->BASE_EL(left) ?
					ub->BASE_EL(left)->BASE_EL(height) : 0;
			rheight = ub->BASE_EL(right) ?
					ub->BASE_EL(right)->BASE_EL(height) : 0;
			if (rheight > lheight)
				ub = ub->BASE_EL(right);
			else if (rheight < lheight)
				ub = ub->BASE_EL(left);
			else
				ub = ub->BASE_EL(right);
		}
		else
		{
			ub = ub->BASE_EL(left);
			lheight = ub->BASE_EL(left) ?
					ub->BASE_EL(left)->BASE_EL(height) : 0;
			rheight = ub->BASE_EL(right) ?
					ub->BASE_EL(right)->BASE_EL(height) : 0;
			if (rheight > lheight)
				ub = ub->BASE_EL(right);
			else if (rheight < lheight)
				ub = ub->BASE_EL(left);
			else
				ub = ub->BASE_EL(left);
		}


		/* rebalance returns the grandparant of the subtree formed
		 * by the element that were rebalanced.
		 * We must continue upward from there rebalancing. */
		fixfrom = rebalance(ub);

		/* Find the next unbalaced element. */
		ub = findFirstUnbalEl(fixfrom);
	}

	return element;
}


/**
 * \brief Empty the tree and delete all the element.
 *
 * Resets the tree to its initial state.
 */
template <AVLMEL_TEMPDEF> void AvlTree<AVLMEL_TEMPUSE>::empty()
{
	if ( root ) {
		/* Recursively delete from the tree structure. */
		deleteChildrenOf(root);
		delete root;
		root = 0;
		treeSize = 0;

#ifdef WALKABLE
		BASELIST::abandon();
#endif
	}
}

/**
 * \brief Forget all element in the tree.
 *
 * Does not delete element. Resets the the tree to it's initial state.
 */
template <AVLMEL_TEMPDEF> void AvlTree<AVLMEL_TEMPUSE>::abandon()
{
	root = 0;
	treeSize = 0;

#ifdef WALKABLE
	BASELIST::abandon();
#endif
}

/* Recursively delete all the children of a element. */
template <AVLMEL_TEMPDEF> void AvlTree<AVLMEL_TEMPUSE>::
		deleteChildrenOf( Element *element )
{
	/* Recurse left. */
	if (element->BASE_EL(left)) {
		deleteChildrenOf(element->BASE_EL(left));

		/* Delete left element. */
		delete element->BASE_EL(left);
		element->BASE_EL(left) = 0;
	}

	/* Recurse right. */
	if (element->BASE_EL(right)) {
		deleteChildrenOf(element->BASE_EL(right));

		/* Delete right element. */
		delete element->BASE_EL(right);
		element->BASE_EL(right) = 0;
	}
}

/* rebalance from a element whose gradparent is unbalanced. Only
 * call on a element that has a grandparent. */
template <AVLMEL_TEMPDEF> Element *AvlTree<AVLMEL_TEMPUSE>::
		rebalance(Element *n)
{
	long lheight, rheight;
	Element *a, *b, *c;
	Element *t1, *t2, *t3, *t4;

	Element *p = n->BASE_EL(parent);      /* parent (Non-NUL). L*/
	Element *gp = p->BASE_EL(parent);     /* Grand-parent (Non-NULL). */
	Element *ggp = gp->BASE_EL(parent);   /* Great grand-parent (may be NULL). */

	if (gp->BASE_EL(right) == p)
	{
		/*  gp
		 *   \
		 *    p
		 */
		if (p->BASE_EL(right) == n)
		{
			/*  gp
			 *   \
			 *    p
			 *     \
			 *      n
			 */
			a = gp;
			b = p;
			c = n;
			t1 = gp->BASE_EL(left);
			t2 = p->BASE_EL(left);
			t3 = n->BASE_EL(left);
			t4 = n->BASE_EL(right);
		}
		else
		{
			/*  gp
			 *     \
			 *       p
			 *      /
			 *     n
			 */
			a = gp;
			b = n;
			c = p;
			t1 = gp->BASE_EL(left);
			t2 = n->BASE_EL(left);
			t3 = n->BASE_EL(right);
			t4 = p->BASE_EL(right);
		}
	}
	else
	{
		/*    gp
		 *   /
		 *  p
		 */
		if (p->BASE_EL(right) == n)
		{
			/*      gp
			 *    /
			 *  p
			 *   \
			 *    n
			 */
			a = p;
			b = n;
			c = gp;
			t1 = p->BASE_EL(left);
			t2 = n->BASE_EL(left);
			t3 = n->BASE_EL(right);
			t4 = gp->BASE_EL(right);
		}
		else
		{
			/*      gp
			 *     /
			 *    p
			 *   /
			 *  n
			 */
			a = n;
			b = p;
			c = gp;
			t1 = n->BASE_EL(left);
			t2 = n->BASE_EL(right);
			t3 = p->BASE_EL(right);
			t4 = gp->BASE_EL(right);
		}
	}

	/* Perform rotation.
	 */

	/* Tie b to the great grandparent. */
	if ( ggp == 0 )
		root = b;
	else if ( ggp->BASE_EL(left) == gp )
		ggp->BASE_EL(left) = b;
	else
		ggp->BASE_EL(right) = b;
	b->BASE_EL(parent) = ggp;

	/* Tie a as a leftchild of b. */
	b->BASE_EL(left) = a;
	a->BASE_EL(parent) = b;

	/* Tie c as a rightchild of b. */
	b->BASE_EL(right) = c;
	c->BASE_EL(parent) = b;

	/* Tie t1 as a leftchild of a. */
	a->BASE_EL(left) = t1;
	if ( t1 != 0 ) t1->BASE_EL(parent) = a;

	/* Tie t2 as a rightchild of a. */
	a->BASE_EL(right) = t2;
	if ( t2 != 0 ) t2->BASE_EL(parent) = a;

	/* Tie t3 as a leftchild of c. */
	c->BASE_EL(left) = t3;
	if ( t3 != 0 ) t3->BASE_EL(parent) = c;

	/* Tie t4 as a rightchild of c. */
	c->BASE_EL(right) = t4;
	if ( t4 != 0 ) t4->BASE_EL(parent) = c;

	/* The heights are all recalculated manualy and the great
	 * grand-parent is passed to recalcHeights() to ensure
	 * the heights are correct up the tree.
	 *
	 * Note that recalcHeights() cuts out when it comes across
	 * a height that hasn't changed.
	 */

	/* Fix height of a. */
	lheight = a->BASE_EL(left) ? a->BASE_EL(left)->BASE_EL(height) : 0;
	rheight = a->BASE_EL(right) ? a->BASE_EL(right)->BASE_EL(height) : 0;
	a->BASE_EL(height) = (lheight > rheight ? lheight : rheight) + 1;

	/* Fix height of c. */
	lheight = c->BASE_EL(left) ? c->BASE_EL(left)->BASE_EL(height) : 0;
	rheight = c->BASE_EL(right) ? c->BASE_EL(right)->BASE_EL(height) : 0;
	c->BASE_EL(height) = (lheight > rheight ? lheight : rheight) + 1;

	/* Fix height of b. */
	lheight = a->BASE_EL(height);
	rheight = c->BASE_EL(height);
	b->BASE_EL(height) = (lheight > rheight ? lheight : rheight) + 1;

	/* Fix height of b's parents. */
	recalcHeights(ggp);
	return ggp;
}

/* Recalculates the heights of all the ancestors of element. */
template <AVLMEL_TEMPDEF> void AvlTree<AVLMEL_TEMPUSE>::
		recalcHeights(Element *element)
{
	long lheight, rheight, new_height;
	while ( element != 0 )
	{
		lheight = element->BASE_EL(left) ? element->BASE_EL(left)->BASE_EL(height) : 0;
		rheight = element->BASE_EL(right) ? element->BASE_EL(right)->BASE_EL(height) : 0;

		new_height = (lheight > rheight ? lheight : rheight) + 1;

		/* If there is no chage in the height, then there will be no
		 * change in any of the ancestor's height. We can stop going up.
		 * If there was a change, continue upward. */
		if (new_height == element->BASE_EL(height))
			return;
		else
			element->BASE_EL(height) = new_height;

		element = element->BASE_EL(parent);
	}
}

/* Finds the first element whose grandparent is unbalanced. */
template <AVLMEL_TEMPDEF> Element *AvlTree<AVLMEL_TEMPUSE>::
		findFirstUnbalGP(Element *element)
{
	long lheight, rheight, balanceProp;
	Element *gp;

	if ( element == 0 || element->BASE_EL(parent) == 0 ||
			element->BASE_EL(parent)->BASE_EL(parent) == 0 )
		return 0;

	/* Don't do anything if we we have no grandparent. */
	gp = element->BASE_EL(parent)->BASE_EL(parent);
	while ( gp != 0 )
	{
		lheight = gp->BASE_EL(left) ? gp->BASE_EL(left)->BASE_EL(height) : 0;
		rheight = gp->BASE_EL(right) ? gp->BASE_EL(right)->BASE_EL(height) : 0;
		balanceProp = lheight - rheight;

		if ( balanceProp < -1 || balanceProp > 1 )
			return element;

		element = element->BASE_EL(parent);
		gp = gp->BASE_EL(parent);
	}
	return 0;
}


/* Finds the first element that is unbalanced. */
template <AVLMEL_TEMPDEF> Element *AvlTree<AVLMEL_TEMPUSE>::
		findFirstUnbalEl(Element *element)
{
	if ( element == 0 )
		return 0;

	while ( element != 0 )
	{
		long lheight = element->BASE_EL(left) ?
				element->BASE_EL(left)->BASE_EL(height) : 0;
		long rheight = element->BASE_EL(right) ?
				element->BASE_EL(right)->BASE_EL(height) : 0;
		long balanceProp = lheight - rheight;

		if ( balanceProp < -1 || balanceProp > 1 )
			return element;

		element = element->BASE_EL(parent);
	}
	return 0;
}

/* Replace a element in the tree with another element not in the tree. */
template <AVLMEL_TEMPDEF> void AvlTree<AVLMEL_TEMPUSE>::
		replaceEl(Element *element, Element *replacement)
{
	Element *parent = element->BASE_EL(parent),
		*left = element->BASE_EL(left),
		*right = element->BASE_EL(right);

	replacement->BASE_EL(left) = left;
	if (left)
		left->BASE_EL(parent) = replacement;
	replacement->BASE_EL(right) = right;
	if (right)
		right->BASE_EL(parent) = replacement;

	replacement->BASE_EL(parent) = parent;
	if (parent)
	{
		if (parent->BASE_EL(left) == element)
			parent->BASE_EL(left) = replacement;
		else
			parent->BASE_EL(right) = replacement;
	}
	else
		root = replacement;

	replacement->BASE_EL(height) = element->BASE_EL(height);
}

/* Removes a element from a tree and puts filler in it's place.
 * Filler should be null or a child of element. */
template <AVLMEL_TEMPDEF> void AvlTree<AVLMEL_TEMPUSE>::
		removeEl(Element *element, Element *filler)
{
	Element *parent = element->BASE_EL(parent);

	if (parent)
	{
		if (parent->BASE_EL(left) == element)
			parent->BASE_EL(left) = filler;
		else
			parent->BASE_EL(right) = filler;
	}
	else
		root = filler;

	if (filler)
		filler->BASE_EL(parent) = parent;

	return;
}

#ifdef AAPL_NAMESPACE
}
#endif
