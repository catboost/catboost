/*
 *  Copyright 2001-2007 Adrian Thurston <thurston@complang.org>
 */

/*  This file is part of Ragel.
 *
 *  Ragel is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 * 
 *  Ragel is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 * 
 *  You should have received a copy of the GNU General Public License
 *  along with Ragel; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA 
 */

#ifndef _FSMGRAPH_H
#define _FSMGRAPH_H

#include "config.h"
#include <assert.h>
#include <iostream>
#include <string>
#include "common.h"
#include "vector.h"
#include "bstset.h"
#include "compare.h"
#include "avltree.h"
#include "dlist.h"
#include "bstmap.h"
#include "sbstmap.h"
#include "sbstset.h"
#include "sbsttable.h"
#include "avlset.h"
#include "avlmap.h"
#include "ragel.h"

//#define LOG_CONDS

/* Flags that control merging. */
#define STB_GRAPH1     0x01
#define STB_GRAPH2     0x02
#define STB_BOTH       0x03
#define STB_ISFINAL    0x04
#define STB_ISMARKED   0x08
#define STB_ONLIST     0x10

using std::ostream;

struct TransAp;
struct StateAp;
struct FsmAp;
struct Action;
struct LongestMatchPart;
struct LengthDef;

/* State list element for unambiguous access to list element. */
struct FsmListEl 
{
	StateAp *prev, *next;
};

/* This is the marked index for a state pair. Used in minimization. It keeps
 * track of whether or not the state pair is marked. */
struct MarkIndex
{
	MarkIndex(int states);
	~MarkIndex();

	void markPair(int state1, int state2);
	bool isPairMarked(int state1, int state2);

private:
	int numStates;
	bool *array;
};

extern KeyOps *keyOps;

/* Transistion Action Element. */
typedef SBstMapEl< int, Action* > ActionTableEl;

/* Nodes in the tree that use this action. */
struct NameInst;
struct InlineList;
typedef Vector<NameInst*> ActionRefs;

/* Element in list of actions. Contains the string for the code to exectute. */
struct Action 
:
	public DListEl<Action>,
	public AvlTreeEl<Action>
{
public:

	Action( const InputLoc &loc, const char *name, InlineList *inlineList, int condId )
	:
		loc(loc),
		name(name),
		inlineList(inlineList), 
		actionId(-1),
		numTransRefs(0),
		numToStateRefs(0),
		numFromStateRefs(0),
		numEofRefs(0),
		numCondRefs(0),
		anyCall(false),
		isLmAction(false),
		condId(condId)
	{
	}

	/* Key for action dictionary. */
	const char *getKey() const { return name; }

	/* Data collected during parse. */
	InputLoc loc;
	const char *name;
	InlineList *inlineList;
	int actionId;

	void actionName( ostream &out )
	{
		if ( name != 0 )
			out << name;
		else
			out << loc.line << ":" << loc.col;
	}

	/* Places in the input text that reference the action. */
	ActionRefs actionRefs;

	/* Number of references in the final machine. */
	int numRefs() 
		{ return numTransRefs + numToStateRefs + numFromStateRefs + numEofRefs; }
	int numTransRefs;
	int numToStateRefs;
	int numFromStateRefs;
	int numEofRefs;
	int numCondRefs;
	bool anyCall;

	bool isLmAction;
	int condId;
};

struct CmpCondId
{
	static inline int compare( const Action *cond1, const Action *cond2 )
	{
		if ( cond1->condId < cond2->condId )
			return -1;
		else if ( cond1->condId > cond2->condId )
			return 1;
		return 0;
	}
};

/* A list of actions. */
typedef DList<Action> ActionList;
typedef AvlTree<Action, char *, CmpStr> ActionDict;

/* Structure for reverse action mapping. */
struct RevActionMapEl
{
	char *name;
	InputLoc location;
};


/* Transition Action Table.  */
struct ActionTable 
	: public SBstMap< int, Action*, CmpOrd<int> >
{
	void setAction( int ordering, Action *action );
	void setActions( int *orderings, Action **actions, int nActs );
	void setActions( const ActionTable &other );

	bool hasAction( Action *action );
};

typedef SBstSet< Action*, CmpOrd<Action*> > ActionSet;
typedef CmpSTable< Action*, CmpOrd<Action*> > CmpActionSet;

/* Transistion Action Element. */
typedef SBstMapEl< int, LongestMatchPart* > LmActionTableEl;

/* Transition Action Table.  */
struct LmActionTable 
	: public SBstMap< int, LongestMatchPart*, CmpOrd<int> >
{
	void setAction( int ordering, LongestMatchPart *action );
	void setActions( const LmActionTable &other );
};

/* Compare of a whole action table element (key & value). */
struct CmpActionTableEl
{
	static int compare( const ActionTableEl &action1, 
			const ActionTableEl &action2 )
	{
		if ( action1.key < action2.key )
			return -1;
		else if ( action1.key > action2.key )
			return 1;
		else if ( action1.value < action2.value )
			return -1;
		else if ( action1.value > action2.value )
			return 1;
		return 0;
	}
};

/* Compare for ActionTable. */
typedef CmpSTable< ActionTableEl, CmpActionTableEl > CmpActionTable;

/* Compare of a whole lm action table element (key & value). */
struct CmpLmActionTableEl
{
	static int compare( const LmActionTableEl &lmAction1, 
			const LmActionTableEl &lmAction2 )
	{
		if ( lmAction1.key < lmAction2.key )
			return -1;
		else if ( lmAction1.key > lmAction2.key )
			return 1;
		else if ( lmAction1.value < lmAction2.value )
			return -1;
		else if ( lmAction1.value > lmAction2.value )
			return 1;
		return 0;
	}
};

/* Compare for ActionTable. */
typedef CmpSTable< LmActionTableEl, CmpLmActionTableEl > CmpLmActionTable;

/* Action table element for error action tables. Adds the encoding of transfer
 * point. */
struct ErrActionTableEl
{
	ErrActionTableEl( Action *action, int ordering, int transferPoint )
		: ordering(ordering), action(action), transferPoint(transferPoint) { }

	/* Ordering and id of the action embedding. */
	int ordering;
	Action *action;

	/* Id of point of transfere from Error action table to transtions and
	 * eofActionTable. */
	int transferPoint;

	int getKey() const { return ordering; }
};

struct ErrActionTable
	: public SBstTable< ErrActionTableEl, int, CmpOrd<int> >
{
	void setAction( int ordering, Action *action, int transferPoint );
	void setActions( const ErrActionTable &other );
};

/* Compare of an error action table element (key & value). */
struct CmpErrActionTableEl
{
	static int compare( const ErrActionTableEl &action1, 
			const ErrActionTableEl &action2 )
	{
		if ( action1.ordering < action2.ordering )
			return -1;
		else if ( action1.ordering > action2.ordering )
			return 1;
		else if ( action1.action < action2.action )
			return -1;
		else if ( action1.action > action2.action )
			return 1;
		else if ( action1.transferPoint < action2.transferPoint )
			return -1;
		else if ( action1.transferPoint > action2.transferPoint )
			return 1;
		return 0;
	}
};

/* Compare for ErrActionTable. */
typedef CmpSTable< ErrActionTableEl, CmpErrActionTableEl > CmpErrActionTable;


/* Descibe a priority, shared among PriorEls. 
 * Has key and whether or not used. */
struct PriorDesc
{
	int key;
	int priority;
};

/* Element in the arrays of priorities for transitions and arrays. Ordering is
 * unique among instantiations of machines, desc is shared. */
struct PriorEl
{
	PriorEl( int ordering, PriorDesc *desc ) 
		: ordering(ordering), desc(desc) { }

	int ordering;
	PriorDesc *desc;
};

/* Compare priority elements, which are ordered by the priority descriptor
 * key. */
struct PriorElCmp
{
	static inline int compare( const PriorEl &pel1, const PriorEl &pel2 ) 
	{
		if ( pel1.desc->key < pel2.desc->key )
			return -1;
		else if ( pel1.desc->key > pel2.desc->key )
			return 1;
		else
			return 0;
	}
};


/* Priority Table. */
struct PriorTable 
	: public SBstSet< PriorEl, PriorElCmp >
{
	void setPrior( int ordering, PriorDesc *desc );
	void setPriors( const PriorTable &other );
};

/* Compare of prior table elements for distinguising state data. */
struct CmpPriorEl
{
	static inline int compare( const PriorEl &pel1, const PriorEl &pel2 )
	{
		if ( pel1.desc < pel2.desc )
			return -1;
		else if ( pel1.desc > pel2.desc )
			return 1;
		else if ( pel1.ordering < pel2.ordering )
			return -1;
		else if ( pel1.ordering > pel2.ordering )
			return 1;
		return 0;
	}
};

/* Compare of PriorTable distinguising state data. Using a compare of the
 * pointers is a little more strict than it needs be. It requires that
 * prioritiy tables have the exact same set of priority assignment operators
 * (from the input lang) to be considered equal. 
 *
 * Really only key-value pairs need be tested and ordering be merged. However
 * this would require that in the fuseing of states, priority descriptors be
 * chosen for the new fused state based on priority. Since the out transition
 * lists and ranges aren't necessarily going to line up, this is more work for
 * little gain. Final compression resets all priorities first, so this would
 * only be useful for compression at every operator, which is only an
 * undocumented test feature.
 */
typedef CmpSTable<PriorEl, CmpPriorEl> CmpPriorTable;

/* Plain action list that imposes no ordering. */
typedef Vector<int> TransFuncList;

/* Comparison for TransFuncList. */
typedef CmpTable< int, CmpOrd<int> > TransFuncListCompare;

/* Transition class that implements actions and priorities. */
struct TransAp 
{
	TransAp() : fromState(0), toState(0) {}
	TransAp( const TransAp &other ) :
		lowKey(other.lowKey),
		highKey(other.highKey),
		fromState(0), toState(0),
		actionTable(other.actionTable),
		priorTable(other.priorTable),
		lmActionTable(other.lmActionTable) {}

	Key lowKey, highKey;
	StateAp *fromState;
	StateAp *toState;

	/* Pointers for outlist. */
	TransAp *prev, *next;

	/* Pointers for in-list. */
	TransAp *ilprev, *ilnext;

	/* The function table and priority for the transition. */
	ActionTable actionTable;
	PriorTable priorTable;

	LmActionTable lmActionTable;
};

/* In transition list. Like DList except only has head pointers, which is all
 * that is required. Insertion and deletion is handled by the graph. This
 * class provides the iterator of a single list. */
struct TransInList
{
	TransInList() : head(0) { }

	TransAp *head;

	struct Iter
	{
		/* Default construct. */
		Iter() : ptr(0) { }

		/* Construct, assign from a list. */
		Iter( const TransInList &il )  : ptr(il.head) { }
		Iter &operator=( const TransInList &dl ) { ptr = dl.head; return *this; }

		/* At the end */
		bool lte() const    { return ptr != 0; }
		bool end() const    { return ptr == 0; }

		/* At the first, last element. */
		bool first() const { return ptr && ptr->ilprev == 0; }
		bool last() const  { return ptr && ptr->ilnext == 0; }

		/* Cast, dereference, arrow ops. */
		operator TransAp*() const   { return ptr; }
		TransAp &operator *() const { return *ptr; }
		TransAp *operator->() const { return ptr; }

		/* Increment, decrement. */
		inline void operator++(int)   { ptr = ptr->ilnext; }
		inline void operator--(int)   { ptr = ptr->ilprev; }

		/* The iterator is simply a pointer. */
		TransAp *ptr;
	};
};

typedef DList<TransAp> TransList;

/* Set of states, list of states. */
typedef BstSet<StateAp*> StateSet;
typedef DList<StateAp> StateList;

/* A element in a state dict. */
struct StateDictEl 
:
	public AvlTreeEl<StateDictEl>
{
	StateDictEl(const StateSet &stateSet) 
		: stateSet(stateSet) { }

	const StateSet &getKey() { return stateSet; }
	StateSet stateSet;
	StateAp *targState;
};

/* Dictionary mapping a set of states to a target state. */
typedef AvlTree< StateDictEl, StateSet, CmpTable<StateAp*> > StateDict;

/* Data needed for a merge operation. */
struct MergeData
{
	MergeData() 
		: stfillHead(0), stfillTail(0) { }

	StateDict stateDict;

	StateAp *stfillHead;
	StateAp *stfillTail;

	void fillListAppend( StateAp *state );
};

struct TransEl
{
	/* Constructors. */
	TransEl() { }
	TransEl( Key lowKey, Key highKey ) 
		: lowKey(lowKey), highKey(highKey) { }
	TransEl( Key lowKey, Key highKey, TransAp *value ) 
		: lowKey(lowKey), highKey(highKey), value(value) { }

	Key lowKey, highKey;
	TransAp *value;
};

struct CmpKey
{
	static int compare( const Key key1, const Key key2 )
	{
		if ( key1 < key2 )
			return -1;
		else if ( key1 > key2 )
			return 1;
		else
			return 0;
	}
};

/* Vector based set of key items. */
typedef BstSet<Key, CmpKey> KeySet;

struct MinPartition 
{
	MinPartition() : active(false) { }

	StateList list;
	bool active;

	MinPartition *prev, *next;
};

/* Epsilon transition stored in a state. Specifies the target */
typedef Vector<int> EpsilonTrans;

/* List of states that are to be drawn into this. */
struct EptVectEl
{
	EptVectEl( StateAp *targ, bool leaving ) 
		: targ(targ), leaving(leaving) { }

	StateAp *targ;
	bool leaving;
};
typedef Vector<EptVectEl> EptVect;

/* Set of entry ids that go into this state. */
typedef BstSet<int> EntryIdSet;

/* Set of longest match items that may be active in a given state. */
typedef BstSet<LongestMatchPart*> LmItemSet;

/* A Conditions which is to be 
 * transfered on pending out transitions. */
struct OutCond
{
	OutCond( Action *action, bool sense )
		: action(action), sense(sense) {}

	Action *action;
	bool sense;
};

struct CmpOutCond
{
	static int compare( const OutCond &outCond1, const OutCond &outCond2 )
	{
		if ( outCond1.action < outCond2.action )
			return -1;
		else if ( outCond1.action > outCond2.action )
			return 1;
		else if ( outCond1.sense < outCond2.sense )
			return -1;
		else if ( outCond1.sense > outCond2.sense )
			return 1;
		return 0;
	}
};

/* Set of conditions to be transfered to on pending out transitions. */
typedef SBstSet< OutCond, CmpOutCond > OutCondSet;
typedef CmpSTable< OutCond, CmpOutCond > CmpOutCondSet;

/* Conditions. */
typedef BstSet< Action*, CmpCondId > CondSet;
typedef CmpTable< Action*, CmpCondId > CmpCondSet;

struct CondSpace
	: public AvlTreeEl<CondSpace>
{
	CondSpace( const CondSet &condSet )
		: condSet(condSet) {}
	
	const CondSet &getKey() { return condSet; }

	CondSet condSet;
	Key baseKey;
	long condSpaceId;
};

typedef Vector<CondSpace*> CondSpaceVect;

typedef AvlTree<CondSpace, CondSet, CmpCondSet> CondSpaceMap;

struct StateCond
{
	StateCond( Key lowKey, Key highKey ) :
		lowKey(lowKey), highKey(highKey) {}

	Key lowKey;
	Key highKey;
	CondSpace *condSpace;

	StateCond *prev, *next;
};

typedef DList<StateCond> StateCondList;
typedef Vector<long> LongVect;

struct Expansion
{
	Expansion( Key lowKey, Key highKey ) :
		lowKey(lowKey), highKey(highKey),
		fromTrans(0), fromCondSpace(0), 
		toCondSpace(0) {}
	
	~Expansion()
	{
		if ( fromTrans != 0 )
			delete fromTrans;
	}

	Key lowKey;
	Key highKey;

	TransAp *fromTrans;
	CondSpace *fromCondSpace;
	long fromVals;

	CondSpace *toCondSpace;
	LongVect toValsList;

	Expansion *prev, *next;
};

typedef DList<Expansion> ExpansionList;

struct Removal
{
	Key lowKey;
	Key highKey;

	Removal *next;
};

struct CondData
{
	CondData() : lastCondKey(0) {}

	/* Condition info. */
	Key lastCondKey;

	CondSpaceMap condSpaceMap;
};

extern CondData *condData;

struct FsmConstructFail
{
	enum Reason
	{
		CondNoKeySpace
	};

	FsmConstructFail( Reason reason ) 
		: reason(reason) {}
	Reason reason;
};

/* State class that implements actions and priorities. */
struct StateAp 
{
	StateAp();
	StateAp(const StateAp &other);
	~StateAp();

	/* Is the state final? */
	bool isFinState() { return stateBits & STB_ISFINAL; }

	/* Out transition list and the pointer for the default out trans. */
	TransList outList;

	/* In transition Lists. */
	TransInList inList;

	/* Set only during scanner construction when actions are added. NFA to DFA
	 * code can ignore this. */
	StateAp *eofTarget;

	/* Entry points into the state. */
	EntryIdSet entryIds;

	/* Epsilon transitions. */
	EpsilonTrans epsilonTrans;

	/* Condition info. */
	StateCondList stateCondList;

	/* Number of in transitions from states other than ourselves. */
	int foreignInTrans;

	/* Temporary data for various algorithms. */
	union {
		/* When duplicating the fsm we need to map each 
		 * state to the new state representing it. */
		StateAp *stateMap;

		/* When minimizing machines by partitioning, this maps to the group
		 * the state is in. */
		MinPartition *partition;

		/* When merging states (state machine operations) this next pointer is
		 * used for the list of states that need to be filled in. */
		StateAp *next;

		/* Identification for printing and stable minimization. */
		int stateNum;

	} alg;

	/* Data used in epsilon operation, maybe fit into alg? */
	StateAp *isolatedShadow;
	int owningGraph;

	/* A pointer to a dict element that contains the set of states this state
	 * represents. This cannot go into alg, because alg.next is used during
	 * the merging process. */
	StateDictEl *stateDictEl;

	/* When drawing epsilon transitions, holds the list of states to merge
	 * with. */
	EptVect *eptVect;

	/* Bits controlling the behaviour of the state during collapsing to dfa. */
	int stateBits;

	/* State list elements. */
	StateAp *next, *prev;

	/* 
	 * Priority and Action data.
	 */

	/* Out priorities transfered to out transitions. */
	PriorTable outPriorTable;

	/* The following two action tables are distinguished by the fact that when
	 * toState actions are executed immediatly after transition actions of
	 * incoming transitions and the current character will be the same as the
	 * one available then. The fromState actions are executed immediately
	 * before the transition actions of outgoing transitions and the current
	 * character is same as the one available then. */

	/* Actions to execute upon entering into a state. */
	ActionTable toStateActionTable;

	/* Actions to execute when going from the state to the transition. */
	ActionTable fromStateActionTable;

	/* Actions to add to any future transitions that leave via this state. */
	ActionTable outActionTable;

	/* Conditions to add to any future transiions that leave via this sttate. */
	OutCondSet outCondSet;

	/* Error action tables. */
	ErrActionTable errActionTable;

	/* Actions to execute on eof. */
	ActionTable eofActionTable;

	/* Set of longest match items that may be active in this state. */
	LmItemSet lmItemSet;
};

template <class ListItem> struct NextTrans
{
	Key lowKey, highKey;
	ListItem *trans;
	ListItem *next;

	void load() {
		if ( trans == 0 )
			next = 0;
		else {
			next = trans->next;
			lowKey = trans->lowKey;
			highKey = trans->highKey;
		}
	}

	void set( ListItem *t ) {
		trans = t;
		load();
	}

	void increment() {
		trans = next;
		load();
	}
};


/* Encodes the different states that are meaningful to the of the iterator. */
enum PairIterUserState
{
	RangeInS1, RangeInS2,
	RangeOverlap,
	BreakS1, BreakS2
};

template <class ListItem1, class ListItem2 = ListItem1> struct PairIter
{
	/* Encodes the different states that an fsm iterator can be in. */
	enum IterState {
		Begin,
		ConsumeS1Range, ConsumeS2Range,
		OnlyInS1Range,  OnlyInS2Range,
		S1SticksOut,    S1SticksOutBreak,
		S2SticksOut,    S2SticksOutBreak,
		S1DragsBehind,  S1DragsBehindBreak,
		S2DragsBehind,  S2DragsBehindBreak,
		ExactOverlap,   End
	};

	PairIter( ListItem1 *list1, ListItem2 *list2 );
	
	/* Query iterator. */
	bool lte() { return itState != End; }
	bool end() { return itState == End; }
	void operator++(int) { findNext(); }
	void operator++()    { findNext(); }

	/* Iterator state. */
	ListItem1 *list1;
	ListItem2 *list2;
	IterState itState;
	PairIterUserState userState;

	NextTrans<ListItem1> s1Tel;
	NextTrans<ListItem2> s2Tel;
	Key bottomLow, bottomHigh;
	ListItem1 *bottomTrans1;
	ListItem2 *bottomTrans2;

private:
	void findNext();
};

/* Init the iterator by advancing to the first item. */
template <class ListItem1, class ListItem2> PairIter<ListItem1, ListItem2>::PairIter( 
		ListItem1 *list1, ListItem2 *list2 )
:
	list1(list1),
	list2(list2),
	itState(Begin)
{
	findNext();
}

/* Return and re-entry for the co-routine iterators. This should ALWAYS be
 * used inside of a block. */
#define CO_RETURN(label) \
	itState = label; \
	return; \
	entry##label: {}

/* Return and re-entry for the co-routine iterators. This should ALWAYS be
 * used inside of a block. */
#define CO_RETURN2(label, uState) \
	itState = label; \
	userState = uState; \
	return; \
	entry##label: {}

/* Advance to the next transition. When returns, trans points to the next
 * transition, unless there are no more, in which case end() returns true. */
template <class ListItem1, class ListItem2> void PairIter<ListItem1, ListItem2>::findNext()
{
	/* Jump into the iterator routine base on the iterator state. */
	switch ( itState ) {
		case Begin:              goto entryBegin;
		case ConsumeS1Range:     goto entryConsumeS1Range;
		case ConsumeS2Range:     goto entryConsumeS2Range;
		case OnlyInS1Range:      goto entryOnlyInS1Range;
		case OnlyInS2Range:      goto entryOnlyInS2Range;
		case S1SticksOut:        goto entryS1SticksOut;
		case S1SticksOutBreak:   goto entryS1SticksOutBreak;
		case S2SticksOut:        goto entryS2SticksOut;
		case S2SticksOutBreak:   goto entryS2SticksOutBreak;
		case S1DragsBehind:      goto entryS1DragsBehind;
		case S1DragsBehindBreak: goto entryS1DragsBehindBreak;
		case S2DragsBehind:      goto entryS2DragsBehind;
		case S2DragsBehindBreak: goto entryS2DragsBehindBreak;
		case ExactOverlap:       goto entryExactOverlap;
		case End:                goto entryEnd;
	}

entryBegin:
	/* Set up the next structs at the head of the transition lists. */
	s1Tel.set( list1 );
	s2Tel.set( list2 );

	/* Concurrently scan both out ranges. */
	while ( true ) {
		if ( s1Tel.trans == 0 ) {
			/* We are at the end of state1's ranges. Process the rest of
			 * state2's ranges. */
			while ( s2Tel.trans != 0 ) {
				/* Range is only in s2. */
				CO_RETURN2( ConsumeS2Range, RangeInS2 );
				s2Tel.increment();
			}
			break;
		}
		else if ( s2Tel.trans == 0 ) {
			/* We are at the end of state2's ranges. Process the rest of
			 * state1's ranges. */
			while ( s1Tel.trans != 0 ) {
				/* Range is only in s1. */
				CO_RETURN2( ConsumeS1Range, RangeInS1 );
				s1Tel.increment();
			}
			break;
		}
		/* Both state1's and state2's transition elements are good.
		 * The signiture of no overlap is a back key being in front of a
		 * front key. */
		else if ( s1Tel.highKey < s2Tel.lowKey ) {
			/* A range exists in state1 that does not overlap with state2. */
			CO_RETURN2( OnlyInS1Range, RangeInS1 );
			s1Tel.increment();
		}
		else if ( s2Tel.highKey < s1Tel.lowKey ) {
			/* A range exists in state2 that does not overlap with state1. */
			CO_RETURN2( OnlyInS2Range, RangeInS2 );
			s2Tel.increment();
		}
		/* There is overlap, must mix the ranges in some way. */
		else if ( s1Tel.lowKey < s2Tel.lowKey ) {
			/* Range from state1 sticks out front. Must break it into
			 * non-overlaping and overlaping segments. */
			bottomLow = s2Tel.lowKey;
			bottomHigh = s1Tel.highKey;
			s1Tel.highKey = s2Tel.lowKey;
			s1Tel.highKey.decrement();
			bottomTrans1 = s1Tel.trans;

			/* Notify the caller that we are breaking s1. This gives them a
			 * chance to duplicate s1Tel[0,1].value. */
			CO_RETURN2( S1SticksOutBreak, BreakS1 );

			/* Broken off range is only in s1. */
			CO_RETURN2( S1SticksOut, RangeInS1 );

			/* Advance over the part sticking out front. */
			s1Tel.lowKey = bottomLow;
			s1Tel.highKey = bottomHigh;
			s1Tel.trans = bottomTrans1;
		}
		else if ( s2Tel.lowKey < s1Tel.lowKey ) {
			/* Range from state2 sticks out front. Must break it into
			 * non-overlaping and overlaping segments. */
			bottomLow = s1Tel.lowKey;
			bottomHigh = s2Tel.highKey;
			s2Tel.highKey = s1Tel.lowKey;
			s2Tel.highKey.decrement();
			bottomTrans2 = s2Tel.trans;

			/* Notify the caller that we are breaking s2. This gives them a
			 * chance to duplicate s2Tel[0,1].value. */
			CO_RETURN2( S2SticksOutBreak, BreakS2 );

			/* Broken off range is only in s2. */
			CO_RETURN2( S2SticksOut, RangeInS2 );

			/* Advance over the part sticking out front. */
			s2Tel.lowKey = bottomLow;
			s2Tel.highKey = bottomHigh;
			s2Tel.trans = bottomTrans2;
		}
		/* Low ends are even. Are the high ends even? */
		else if ( s1Tel.highKey < s2Tel.highKey ) {
			/* Range from state2 goes longer than the range from state1. We
			 * must break the range from state2 into an evenly overlaping
			 * segment. */
			bottomLow = s1Tel.highKey;
			bottomLow.increment();
			bottomHigh = s2Tel.highKey;
			s2Tel.highKey = s1Tel.highKey;
			bottomTrans2 = s2Tel.trans;

			/* Notify the caller that we are breaking s2. This gives them a
			 * chance to duplicate s2Tel[0,1].value. */
			CO_RETURN2( S2DragsBehindBreak, BreakS2 );

			/* Breaking s2 produces exact overlap. */
			CO_RETURN2( S2DragsBehind, RangeOverlap );

			/* Advance over the front we just broke off of range 2. */
			s2Tel.lowKey = bottomLow;
			s2Tel.highKey = bottomHigh;
			s2Tel.trans = bottomTrans2;

			/* Advance over the entire s1Tel. We have consumed it. */
			s1Tel.increment();
		}
		else if ( s2Tel.highKey < s1Tel.highKey ) {
			/* Range from state1 goes longer than the range from state2. We
			 * must break the range from state1 into an evenly overlaping
			 * segment. */
			bottomLow = s2Tel.highKey;
			bottomLow.increment();
			bottomHigh = s1Tel.highKey;
			s1Tel.highKey = s2Tel.highKey;
			bottomTrans1 = s1Tel.trans;

			/* Notify the caller that we are breaking s1. This gives them a
			 * chance to duplicate s2Tel[0,1].value. */
			CO_RETURN2( S1DragsBehindBreak, BreakS1 );

			/* Breaking s1 produces exact overlap. */
			CO_RETURN2( S1DragsBehind, RangeOverlap );

			/* Advance over the front we just broke off of range 1. */
			s1Tel.lowKey = bottomLow;
			s1Tel.highKey = bottomHigh;
			s1Tel.trans = bottomTrans1;

			/* Advance over the entire s2Tel. We have consumed it. */
			s2Tel.increment();
		}
		else {
			/* There is an exact overlap. */
			CO_RETURN2( ExactOverlap, RangeOverlap );

			s1Tel.increment();
			s2Tel.increment();
		}
	}

	/* Done, go into end state. */
	CO_RETURN( End );
}


/* Compare lists of epsilon transitions. Entries are name ids of targets. */
typedef CmpTable< int, CmpOrd<int> > CmpEpsilonTrans;

/* Compare class for the Approximate minimization. */
class ApproxCompare
{
public:
	ApproxCompare() { }
	int compare( const StateAp *pState1, const StateAp *pState2 );
};

/* Compare class for the initial partitioning of a partition minimization. */
class InitPartitionCompare
{
public:
	InitPartitionCompare() { }
	int compare( const StateAp *pState1, const StateAp *pState2 );
};

/* Compare class for the regular partitioning of a partition minimization. */
class PartitionCompare
{
public:
	PartitionCompare() { }
	int compare( const StateAp *pState1, const StateAp *pState2 );
};

/* Compare class for a minimization that marks pairs. Provides the shouldMark
 * routine. */
class MarkCompare
{
public:
	MarkCompare() { }
	bool shouldMark( MarkIndex &markIndex, const StateAp *pState1, 
			const StateAp *pState2 );
};

/* List of partitions. */
typedef DList< MinPartition > PartitionList;

/* List of transtions out of a state. */
typedef Vector<TransEl> TransListVect;

/* Entry point map used for keeping track of entry points in a machine. */
typedef BstSet< int > EntryIdSet;
typedef BstMapEl< int, StateAp* > EntryMapEl;
typedef BstMap< int, StateAp* > EntryMap;
typedef Vector<EntryMapEl> EntryMapBase;

/* Graph class that implements actions and priorities. */
struct FsmAp 
{
	/* Constructors/Destructors. */
	FsmAp( );
	FsmAp( const FsmAp &graph );
	~FsmAp();

	/* The list of states. */
	StateList stateList;
	StateList misfitList;

	/* The map of entry points. */
	EntryMap entryPoints;

	/* The start state. */
	StateAp *startState;

	/* Error state, possibly created only when the final machine has been
	 * created and the XML machine is about to be written. No transitions
	 * point to this state. */
	StateAp *errState;

	/* The set of final states. */
	StateSet finStateSet;

	/* Misfit Accounting. Are misfits put on a separate list. */
	bool misfitAccounting;

	/*
	 * Transition actions and priorities.
	 */

	/* Set priorities on transtions. */
	void startFsmPrior( int ordering, PriorDesc *prior );
	void allTransPrior( int ordering, PriorDesc *prior );
	void finishFsmPrior( int ordering, PriorDesc *prior );
	void leaveFsmPrior( int ordering, PriorDesc *prior );

	/* Action setting support. */
	void transferOutActions( StateAp *state );
	void transferErrorActions( StateAp *state, int transferPoint );
	void setErrorActions( StateAp *state, const ActionTable &other );
	void setErrorAction( StateAp *state, int ordering, Action *action );

	/* Fill all spaces in a transition list with an error transition. */
	void fillGaps( StateAp *state );

	/* Similar to setErrorAction, instead gives a state to go to on error. */
	void setErrorTarget( StateAp *state, StateAp *target, int *orderings, 
			Action **actions, int nActs );

	/* Set actions to execute. */
	void startFsmAction( int ordering, Action *action );
	void allTransAction( int ordering, Action *action );
	void finishFsmAction( int ordering, Action *action );
	void leaveFsmAction( int ordering, Action *action );
	void longMatchAction( int ordering, LongestMatchPart *lmPart );

	/* Set conditions. */
	CondSpace *addCondSpace( const CondSet &condSet );

	void findEmbedExpansions( ExpansionList &expansionList, 
		StateAp *destState, Action *condAction, bool sense );
	void embedCondition( MergeData &md, StateAp *state, Action *condAction, bool sense );
	void embedCondition( StateAp *state, Action *condAction, bool sense );

	void startFsmCondition( Action *condAction, bool sense );
	void allTransCondition( Action *condAction, bool sense );
	void leaveFsmCondition( Action *condAction, bool sense );

	/* Set error actions to execute. */
	void startErrorAction( int ordering, Action *action, int transferPoint );
	void allErrorAction( int ordering, Action *action, int transferPoint );
	void finalErrorAction( int ordering, Action *action, int transferPoint );
	void notStartErrorAction( int ordering, Action *action, int transferPoint );
	void notFinalErrorAction( int ordering, Action *action, int transferPoint );
	void middleErrorAction( int ordering, Action *action, int transferPoint );

	/* Set EOF actions. */
	void startEOFAction( int ordering, Action *action );
	void allEOFAction( int ordering, Action *action );
	void finalEOFAction( int ordering, Action *action );
	void notStartEOFAction( int ordering, Action *action );
	void notFinalEOFAction( int ordering, Action *action );
	void middleEOFAction( int ordering, Action *action );

	/* Set To State actions. */
	void startToStateAction( int ordering, Action *action );
	void allToStateAction( int ordering, Action *action );
	void finalToStateAction( int ordering, Action *action );
	void notStartToStateAction( int ordering, Action *action );
	void notFinalToStateAction( int ordering, Action *action );
	void middleToStateAction( int ordering, Action *action );

	/* Set From State actions. */
	void startFromStateAction( int ordering, Action *action );
	void allFromStateAction( int ordering, Action *action );
	void finalFromStateAction( int ordering, Action *action );
	void notStartFromStateAction( int ordering, Action *action );
	void notFinalFromStateAction( int ordering, Action *action );
	void middleFromStateAction( int ordering, Action *action );

	/* Shift the action ordering of the start transitions to start at
	 * fromOrder and increase in units of 1. Useful before kleene star
	 * operation.  */
	int shiftStartActionOrder( int fromOrder );

	/* Clear all priorities from the fsm to so they won't affcet minimization
	 * of the final fsm. */
	void clearAllPriorities();

	/* Zero out all the function keys. */
	void nullActionKeys();

	/* Walk the list of states and verify state properties. */
	void verifyStates();

	/* Misfit Accounting. Are misfits put on a separate list. */
	void setMisfitAccounting( bool val ) 
		{ misfitAccounting = val; }

	/* Set and Unset a state as final. */
	void setFinState( StateAp *state );
	void unsetFinState( StateAp *state );

	void setStartState( StateAp *state );
	void unsetStartState( );
	
	/* Set and unset a state as an entry point. */
	void setEntry( int id, StateAp *state );
	void changeEntry( int id, StateAp *to, StateAp *from );
	void unsetEntry( int id, StateAp *state );
	void unsetEntry( int id );
	void unsetAllEntryPoints();

	/* Epsilon transitions. */
	void epsilonTrans( int id );
	void shadowReadWriteStates( MergeData &md );

	/*
	 * Basic attaching and detaching.
	 */

	/* Common to attaching/detaching list and default. */
	void attachToInList( StateAp *from, StateAp *to, TransAp *&head, TransAp *trans );
	void detachFromInList( StateAp *from, StateAp *to, TransAp *&head, TransAp *trans );

	/* Attach with a new transition. */
	TransAp *attachNewTrans( StateAp *from, StateAp *to,
			Key onChar1, Key onChar2 );

	/* Attach with an existing transition that already in an out list. */
	void attachTrans( StateAp *from, StateAp *to, TransAp *trans );
	
	/* Redirect a transition away from error and towards some state. */
	void redirectErrorTrans( StateAp *from, StateAp *to, TransAp *trans );

	/* Detach a transition from a target state. */
	void detachTrans( StateAp *from, StateAp *to, TransAp *trans );

	/* Detach a state from the graph. */
	void detachState( StateAp *state );

	/*
	 * NFA to DFA conversion routines.
	 */

	/* Duplicate a transition that will dropin to a free spot. */
	TransAp *dupTrans( StateAp *from, TransAp *srcTrans );

	/* In crossing, two transitions both go to real states. */
	TransAp *fsmAttachStates( MergeData &md, StateAp *from,
			TransAp *destTrans, TransAp *srcTrans );

	/* Two transitions are to be crossed, handle the possibility of either
	 * going to the error state. */
	TransAp *mergeTrans( MergeData &md, StateAp *from,
			TransAp *destTrans, TransAp *srcTrans );

	/* Compare deterimne relative priorities of two transition tables. */
	int comparePrior( const PriorTable &priorTable1, const PriorTable &priorTable2 );

	/* Cross a src transition with one that is already occupying a spot. */
	TransAp *crossTransitions( MergeData &md, StateAp *from,
			TransAp *destTrans, TransAp *srcTrans );

	void outTransCopy( MergeData &md, StateAp *dest, TransAp *srcList );

	void doRemove( MergeData &md, StateAp *destState, ExpansionList &expList1 );
	void doExpand( MergeData &md, StateAp *destState, ExpansionList &expList1 );
	void findCondExpInTrans( ExpansionList &expansionList, StateAp *state, 
			Key lowKey, Key highKey, CondSpace *fromCondSpace, CondSpace *toCondSpace,
			long destVals, LongVect &toValsList );
	void findTransExpansions( ExpansionList &expansionList, 
			StateAp *destState, StateAp *srcState );
	void findCondExpansions( ExpansionList &expansionList, 
			StateAp *destState, StateAp *srcState );
	void mergeStateConds( StateAp *destState, StateAp *srcState );

	/* Merge a set of states into newState. */
	void mergeStates( MergeData &md, StateAp *destState, 
			StateAp **srcStates, int numSrc );
	void mergeStatesLeaving( MergeData &md, StateAp *destState, StateAp *srcState );
	void mergeStates( MergeData &md, StateAp *destState, StateAp *srcState );

	/* Make all states that are combinations of other states and that
	 * have not yet had their out transitions filled in. This will 
	 * empty out stateDict and stFil. */
	void fillInStates( MergeData &md );

	/*
	 * Transition Comparison.
	 */

	/* Compare transition data. Either of the pointers may be null. */
	static inline int compareDataPtr( TransAp *trans1, TransAp *trans2 );

	/* Compare target state and transition data. Either pointer may be null. */
	static inline int compareFullPtr( TransAp *trans1, TransAp *trans2 );

	/* Compare target partitions. Either pointer may be null. */
	static inline int comparePartPtr( TransAp *trans1, TransAp *trans2 );

	/* Check marked status of target states. Either pointer may be null. */
	static inline bool shouldMarkPtr( MarkIndex &markIndex, 
			TransAp *trans1, TransAp *trans2 );

	/*
	 * Callbacks.
	 */

	/* Compare priority and function table of transitions. */
	static int compareTransData( TransAp *trans1, TransAp *trans2 );

	/* Add in the properties of srcTrans into this. */
	void addInTrans( TransAp *destTrans, TransAp *srcTrans );

	/* Compare states on data stored in the states. */
	static int compareStateData( const StateAp *state1, const StateAp *state2 );

	/* Out transition data. */
	void clearOutData( StateAp *state );
	bool hasOutData( StateAp *state );
	void transferOutData( StateAp *destState, StateAp *srcState );

	/*
	 * Allocation.
	 */

	/* New up a state and add it to the graph. */
	StateAp *addState();

	/*
	 * Building basic machines
	 */

	void concatFsm( Key c );
	void concatFsm( Key *str, int len );
	void concatFsmCI( Key *str, int len );
	void orFsm( Key *set, int len );
	void rangeFsm( Key low, Key high );
	void rangeStarFsm( Key low, Key high );
	void emptyFsm( );
	void lambdaFsm( );

	/*
	 * Fsm operators.
	 */

	void starOp( );
	void repeatOp( int times );
	void optionalRepeatOp( int times );
	void concatOp( FsmAp *other );
	void unionOp( FsmAp *other );
	void intersectOp( FsmAp *other );
	void subtractOp( FsmAp *other );
	void epsilonOp();
	void joinOp( int startId, int finalId, FsmAp **others, int numOthers );
	void globOp( FsmAp **others, int numOthers );
	void deterministicEntry();

	/*
	 * Operator workers
	 */

	/* Determine if there are any entry points into a start state other than
	 * the start state. */
	bool isStartStateIsolated();

	/* Make a new start state that has no entry points. Will not change the
	 * identity of the fsm. */
	void isolateStartState();

	/* Workers for resolving epsilon transitions. */
	bool inEptVect( EptVect *eptVect, StateAp *targ );
	void epsilonFillEptVectFrom( StateAp *root, StateAp *from, bool parentLeaving );
	void resolveEpsilonTrans( MergeData &md );

	/* Workers for concatenation and union. */
	void doConcat( FsmAp *other, StateSet *fromStates, bool optional );
	void doOr( FsmAp *other );

	/*
	 * Final states
	 */

	/* Unset any final states that are no longer to be final 
	 * due to final bits. */
	void unsetIncompleteFinals();
	void unsetKilledFinals();

	/* Bring in other's entry points. Assumes others states are going to be
	 * copied into this machine. */
	void copyInEntryPoints( FsmAp *other );

	/* Ordering states. */
	void depthFirstOrdering( StateAp *state );
	void depthFirstOrdering();
	void sortStatesByFinal();

	/* Set sqequential state numbers starting at 0. */
	void setStateNumbers( int base );

	/* Unset all final states. */
	void unsetAllFinStates();

	/* Set the bits of final states and clear the bits of non final states. */
	void setFinBits( int finStateBits );

	/*
	 * Self-consistency checks.
	 */

	/* Run a sanity check on the machine. */
	void verifyIntegrity();

	/* Verify that there are no unreachable states, or dead end states. */
	void verifyReachability();
	void verifyNoDeadEndStates();

	/*
	 * Path pruning
	 */

	/* Mark all states reachable from state. */
	void markReachableFromHereReverse( StateAp *state );

	/* Mark all states reachable from state. */
	void markReachableFromHere( StateAp *state );
	void markReachableFromHereStopFinal( StateAp *state );

	/* Removes states that cannot be reached by any path in the fsm and are
	 * thus wasted silicon. */
	void removeDeadEndStates();

	/* Removes states that cannot be reached by any path in the fsm and are
	 * thus wasted silicon. */
	void removeUnreachableStates();

	/* Remove error actions from states on which the error transition will
	 * never be taken. */
	bool outListCovers( StateAp *state );
	bool anyErrorRange( StateAp *state );

	/* Remove states that are on the misfit list. */
	void removeMisfits();

	/*
	 * FSM Minimization
	 */

	/* Minimization by partitioning. */
	void minimizePartition1();
	void minimizePartition2();

	/* Minimize the final state Machine. The result is the minimal fsm. Slow
	 * but stable, correct minimization. Uses n^2 space (lookout) and average
	 * n^2 time. Worst case n^3 time, but a that is a very rare case. */
	void minimizeStable();

	/* Minimize the final state machine. Does not find the minimal fsm, but a
	 * pretty good approximation. Does not use any extra space. Average n^2
	 * time. Worst case n^3 time, but a that is a very rare case. */
	void minimizeApproximate();

	/* This is the worker for the minimize approximate solution. It merges
	 * states that have identical out transitions. */
	bool minimizeRound( );

	/* Given an intial partioning of states, split partitions that have out trans
	 * to differing partitions. */
	int partitionRound( StateAp **statePtrs, MinPartition *parts, int numParts );

	/* Split partitions that have a transition to a previously split partition, until
	 * there are no more partitions to split. */
	int splitCandidates( StateAp **statePtrs, MinPartition *parts, int numParts );

	/* Fuse together states in the same partition. */
	void fusePartitions( MinPartition *parts, int numParts );

	/* Mark pairs where out final stateness differs, out trans data differs,
	 * trans pairs go to a marked pair or trans data differs. Should get 
	 * alot of pairs. */
	void initialMarkRound( MarkIndex &markIndex );

	/* One marking round on all state pairs. Considers if trans pairs go
	 * to a marked state only. Returns whether or not a pair was marked. */
	bool markRound( MarkIndex &markIndex );

	/* Move the in trans into src into dest. */
	void inTransMove(StateAp *dest, StateAp *src);
	
	/* Make state src and dest the same state. */
	void fuseEquivStates(StateAp *dest, StateAp *src);

	/* Find any states that didn't get marked by the marking algorithm and
	 * merge them into the primary states of their equivalence class. */
	void fuseUnmarkedPairs( MarkIndex &markIndex );

	/* Merge neighboring transitions go to the same state and have the same
	 * transitions data. */
	void compressTransitions();

	/* Returns true if there is a transtion (either explicit or by a gap) to
	 * the error state. */
	bool checkErrTrans( StateAp *state, TransAp *trans );
	bool checkErrTransFinish( StateAp *state );
	bool hasErrorTrans();

	/* Check if a machine defines a single character. This is useful in
	 * validating ranges and machines to export. */
	bool checkSingleCharMachine( );
};

#endif
