/*
 *  Copyright 2001-2006 Adrian Thurston <thurston@complang.org>
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

#ifndef _REDFSM_H
#define _REDFSM_H

#include <assert.h>
#include <string.h>
#include <string>
#include "config.h"
#include "common.h"
#include "vector.h"
#include "dlist.h"
#include "compare.h"
#include "bstmap.h"
#include "bstset.h"
#include "avlmap.h"
#include "avltree.h"
#include "avlbasic.h"
#include "mergesort.h"
#include "sbstmap.h"
#include "sbstset.h"
#include "sbsttable.h"


#define TRANS_ERR_TRANS   0
#define STATE_ERR_STATE   0
#define FUNC_NO_FUNC      0

using std::string;

struct RedStateAp;
struct GenInlineList;
struct GenAction;

/*
 * Inline code tree
 */
struct GenInlineItem
{
	enum Type 
	{
		Text, Goto, Call, Next, GotoExpr, CallExpr, NextExpr, Ret, 
		PChar, Char, Hold, Exec, Curs, Targs, Entry,
		LmSwitch, LmSetActId, LmSetTokEnd, LmGetTokEnd, LmInitTokStart,
		LmInitAct, LmSetTokStart, SubAction, Break
	};

	GenInlineItem( const InputLoc &loc, Type type ) : 
		loc(loc), data(0), targId(0), targState(0), 
		lmId(0), children(0), offset(0),
		type(type) { }
	
	InputLoc loc;
	char *data;
	int targId;
	RedStateAp *targState;
	int lmId;
	GenInlineList *children;
	int offset;
	Type type;

	GenInlineItem *prev, *next;
};

/* Normally this would be atypedef, but that would entail including DList from
 * ptreetypes, which should be just typedef forwards. */
struct GenInlineList : public DList<GenInlineItem> { };

/* Element in list of actions. Contains the string for the code to exectute. */
struct GenAction 
:
	public DListEl<GenAction>
{
	GenAction( )
	:
		name(0),
		inlineList(0), 
		actionId(0),
		numTransRefs(0),
		numToStateRefs(0),
		numFromStateRefs(0),
		numEofRefs(0)
	{
	}

	/* Data collected during parse. */
	InputLoc loc;
	const char *name;
	GenInlineList *inlineList;
	int actionId;

	string nameOrLoc();

	/* Number of references in the final machine. */
	int numRefs() 
		{ return numTransRefs + numToStateRefs + numFromStateRefs + numEofRefs; }
	int numTransRefs;
	int numToStateRefs;
	int numFromStateRefs;
	int numEofRefs;
};


/* Forwards. */
struct RedStateAp;
struct StateAp;

/* Transistion GenAction Element. */
typedef SBstMapEl< int, GenAction* > GenActionTableEl;

/* Transition GenAction Table.  */
struct GenActionTable 
	: public SBstMap< int, GenAction*, CmpOrd<int> >
{
	void setAction( int ordering, GenAction *action );
	void setActions( int *orderings, GenAction **actions, int nActs );
	void setActions( const GenActionTable &other );
};

/* Compare of a whole action table element (key & value). */
struct CmpGenActionTableEl
{
	static int compare( const GenActionTableEl &action1, 
			const GenActionTableEl &action2 )
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

/* Compare for GenActionTable. */
typedef CmpSTable< GenActionTableEl, CmpGenActionTableEl > CmpGenActionTable;

/* Set of states. */
typedef BstSet<RedStateAp*> RedStateSet;
typedef BstSet<int> IntSet;

/* Reduced action. */
struct RedAction
:
	public AvlTreeEl<RedAction>
{
	RedAction( )
	:	
		key(), 
		eofRefs(0),
		numTransRefs(0),
		numToStateRefs(0),
		numFromStateRefs(0),
		numEofRefs(0),
		bAnyNextStmt(false), 
		bAnyCurStateRef(false),
		bAnyBreakStmt(false)
	{ }
	
	const GenActionTable &getKey() 
		{ return key; }

	GenActionTable key;
	int actListId;
	int location;
	IntSet *eofRefs;

	/* Number of references in the final machine. */
	int numRefs() 
		{ return numTransRefs + numToStateRefs + numFromStateRefs + numEofRefs; }
	int numTransRefs;
	int numToStateRefs;
	int numFromStateRefs;
	int numEofRefs;

	bool anyNextStmt() { return bAnyNextStmt; }
	bool anyCurStateRef() { return bAnyCurStateRef; }
	bool anyBreakStmt() { return bAnyBreakStmt; }

	bool bAnyNextStmt;
	bool bAnyCurStateRef;
	bool bAnyBreakStmt;
};
typedef AvlTree<RedAction, GenActionTable, CmpGenActionTable> GenActionTableMap;

/* Reduced transition. */
struct RedTransAp
:
	public AvlTreeEl<RedTransAp>
{
	RedTransAp( RedStateAp *targ, RedAction *action, int id )
		: targ(targ), action(action), id(id), pos(-1), labelNeeded(true) { }

	RedStateAp *targ;
	RedAction *action;
	int id;
	int pos;
	bool partitionBoundary;
	bool labelNeeded;
};

/* Compare of transitions for the final reduction of transitions. Comparison
 * is on target and the pointer to the shared action table. It is assumed that
 * when this is used the action tables have been reduced. */
struct CmpRedTransAp
{
	static int compare( const RedTransAp &t1, const RedTransAp &t2 )
	{
		if ( t1.targ < t2.targ )
			return -1;
		else if ( t1.targ > t2.targ )
			return 1;
		else if ( t1.action < t2.action )
			return -1;
		else if ( t1.action > t2.action )
			return 1;
		else
			return 0;
	}
};

typedef AvlBasic<RedTransAp, CmpRedTransAp> TransApSet;

/* Element in out range. */
struct RedTransEl
{
	/* Constructors. */
	RedTransEl( Key lowKey, Key highKey, RedTransAp *value ) 
		: lowKey(lowKey), highKey(highKey), value(value) { }

	Key lowKey, highKey;
	RedTransAp *value;
};

typedef Vector<RedTransEl> RedTransList;
typedef Vector<RedStateAp*> RedStateVect;

typedef BstMapEl<RedStateAp*, unsigned long long> RedSpanMapEl;
typedef BstMap<RedStateAp*, unsigned long long> RedSpanMap;

/* Compare used by span map sort. Reverse sorts by the span. */
struct CmpRedSpanMapEl
{
	static int compare( const RedSpanMapEl &smel1, const RedSpanMapEl &smel2 )
	{
		if ( smel1.value > smel2.value )
			return -1;
		else if ( smel1.value < smel2.value )
			return 1;
		else
			return 0;
	}
};

/* Sorting state-span map entries by span. */
typedef MergeSort<RedSpanMapEl, CmpRedSpanMapEl> RedSpanMapSort;

/* Set of entry ids that go into this state. */
typedef Vector<int> EntryIdVect;
typedef Vector<char*> EntryNameVect;

typedef Vector< GenAction* > GenCondSet;

struct Condition
{
	Condition( )
		: key(0), baseKey(0) {}

	Key key;
	Key baseKey;
	GenCondSet condSet;

	Condition *next, *prev;
};
typedef DList<Condition> ConditionList;

struct GenCondSpace
{
	Key baseKey;
	GenCondSet condSet;
	int condSpaceId;

	GenCondSpace *next, *prev;
};
typedef DList<GenCondSpace> CondSpaceList;

struct GenStateCond
{
	Key lowKey;
	Key highKey;

	GenCondSpace *condSpace;

	GenStateCond *prev, *next;
};
typedef DList<GenStateCond> GenStateCondList;
typedef Vector<GenStateCond*> StateCondVect;

/* Reduced state. */
struct RedStateAp
{
	RedStateAp()
	: 
		defTrans(0), 
		condList(0),
		transList(0), 
		isFinal(false), 
		labelNeeded(false), 
		outNeeded(false), 
		onStateList(false), 
		toStateAction(0), 
		fromStateAction(0), 
		eofAction(0), 
		eofTrans(0), 
		id(0), 
		bAnyRegCurStateRef(false),
		partitionBoundary(false),
		inTrans(0),
		numInTrans(0)
	{ }

	/* Transitions out. */
	RedTransList outSingle;
	RedTransList outRange;
	RedTransAp *defTrans;

	/* For flat conditions. */
	Key condLowKey, condHighKey;
	GenCondSpace **condList;

	/* For flat keys. */
	Key lowKey, highKey;
	RedTransAp **transList;

	/* The list of states that transitions from this state go to. */
	RedStateVect targStates;

	bool isFinal;
	bool labelNeeded;
	bool outNeeded;
	bool onStateList;
	RedAction *toStateAction;
	RedAction *fromStateAction;
	RedAction *eofAction;
	RedTransAp *eofTrans;
	int id;
	GenStateCondList stateCondList;
	StateCondVect stateCondVect;

	/* Pointers for the list of states. */
	RedStateAp *prev, *next;

	bool anyRegCurStateRef() { return bAnyRegCurStateRef; }
	bool bAnyRegCurStateRef;

	int partition;
	bool partitionBoundary;

	RedTransAp **inTrans;
	int numInTrans;
};

/* List of states. */
typedef DList<RedStateAp> RedStateList;

/* Set of reduced transitons. Comparison is by pointer. */
typedef BstSet< RedTransAp*, CmpOrd<RedTransAp*> > RedTransSet;

/* Next version of the fsm machine. */
struct RedFsmAp
{
	RedFsmAp();

	bool forcedErrorState;

	int nextActionId;
	int nextTransId;

	/* Next State Id doubles as the total number of state ids. */
	int nextStateId;

	TransApSet transSet;
	GenActionTableMap actionMap;
	RedStateList stateList;
	RedStateSet entryPoints;
	RedStateAp *startState;
	RedStateAp *errState;
	RedTransAp *errTrans;
	RedTransAp *errActionTrans;
	RedStateAp *firstFinState;
	int numFinStates;
	int nParts;

	bool bAnyToStateActions;
	bool bAnyFromStateActions;
	bool bAnyRegActions;
	bool bAnyEofActions;
	bool bAnyEofTrans;
	bool bAnyActionGotos;
	bool bAnyActionCalls;
	bool bAnyActionRets;
	bool bAnyActionByValControl;
	bool bAnyRegActionRets;
	bool bAnyRegActionByValControl;
	bool bAnyRegNextStmt;
	bool bAnyRegCurStateRef;
	bool bAnyRegBreak;
	bool bAnyConditions;

	int maxState;
	int maxSingleLen;
	int maxRangeLen;
	int maxKeyOffset;
	int maxIndexOffset;
	int maxIndex;
	int maxActListId;
	int maxActionLoc;
	int maxActArrItem;
	unsigned long long maxSpan;
	unsigned long long maxCondSpan;
	int maxFlatIndexOffset;
	Key maxKey;
	int maxCondOffset;
	int maxCondLen;
	int maxCondSpaceId;
	int maxCondIndexOffset;
	int maxCond;

	bool anyActions();
	bool anyToStateActions()        { return bAnyToStateActions; }
	bool anyFromStateActions()      { return bAnyFromStateActions; }
	bool anyRegActions()            { return bAnyRegActions; }
	bool anyEofActions()            { return bAnyEofActions; }
	bool anyEofTrans()              { return bAnyEofTrans; }
	bool anyActionGotos()           { return bAnyActionGotos; }
	bool anyActionCalls()           { return bAnyActionCalls; }
	bool anyActionRets()            { return bAnyActionRets; }
	bool anyActionByValControl()    { return bAnyActionByValControl; }
	bool anyRegActionRets()         { return bAnyRegActionRets; }
	bool anyRegActionByValControl() { return bAnyRegActionByValControl; }
	bool anyRegNextStmt()           { return bAnyRegNextStmt; }
	bool anyRegCurStateRef()        { return bAnyRegCurStateRef; }
	bool anyRegBreak()              { return bAnyRegBreak; }
	bool anyConditions()            { return bAnyConditions; }


	/* Is is it possible to extend a range by bumping ranges that span only
	 * one character to the singles array. */
	bool canExtend( const RedTransList &list, int pos );

	/* Pick single transitions from the ranges. */
	void moveTransToSingle( RedStateAp *state );
	void chooseSingle();

	void makeFlat();

	/* Move a selected transition from ranges to default. */
	void moveToDefault( RedTransAp *defTrans, RedStateAp *state );

	/* Pick a default transition by largest span. */
	RedTransAp *chooseDefaultSpan( RedStateAp *state );
	void chooseDefaultSpan();

	/* Pick a default transition by most number of ranges. */
	RedTransAp *chooseDefaultNumRanges( RedStateAp *state );
	void chooseDefaultNumRanges();

	/* Pick a default transition tailored towards goto driven machine. */
	RedTransAp *chooseDefaultGoto( RedStateAp *state );
	void chooseDefaultGoto();

	/* Ordering states by transition connections. */
	void optimizeStateOrdering( RedStateAp *state );
	void optimizeStateOrdering();

	/* Ordering states by transition connections. */
	void depthFirstOrdering( RedStateAp *state );
	void depthFirstOrdering();

	/* Set state ids. */
	void sequentialStateIds();
	void sortStateIdsByFinal();

	/* Arrange states in by final id. This is a stable sort. */
	void sortStatesByFinal();

	/* Sorting states by id. */
	void sortByStateId();

	/* Locating the first final state. This is the final state with the lowest
	 * id. */
	void findFirstFinState();

	void assignActionLocs();

	RedTransAp *getErrorTrans();
	RedStateAp *getErrorState();

	/* Is every char in the alphabet covered? */
	bool alphabetCovered( RedTransList &outRange );

	RedTransAp *allocateTrans( RedStateAp *targState, RedAction *actionTable );

	void partitionFsm( int nParts );

	void setInTrans();
};


#endif
