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

#include "redfsm.h"
#include "avlmap.h"
#include "mergesort.h"
#include <iostream>
#include <sstream>

using std::ostringstream;

string GenAction::nameOrLoc()
{
	if ( name != 0 )
		return string(name);
	else {
		ostringstream ret;
		ret << loc.line << ":" << loc.col;
		return ret.str();
	}
}

RedFsmAp::RedFsmAp()
:
	forcedErrorState(false),
	nextActionId(0),
	nextTransId(0),
	startState(0),
	errState(0),
	errTrans(0),
	firstFinState(0),
	numFinStates(0),
	bAnyToStateActions(false),
	bAnyFromStateActions(false),
	bAnyRegActions(false),
	bAnyEofActions(false),
	bAnyEofTrans(false),
	bAnyActionGotos(false),
	bAnyActionCalls(false),
	bAnyActionRets(false),
	bAnyActionByValControl(false),
	bAnyRegActionRets(false),
	bAnyRegActionByValControl(false),
	bAnyRegNextStmt(false),
	bAnyRegCurStateRef(false),
	bAnyRegBreak(false),
	bAnyConditions(false)
{
}

/* Does the machine have any actions. */
bool RedFsmAp::anyActions()
{
	return actionMap.length() > 0;
}

void RedFsmAp::depthFirstOrdering( RedStateAp *state )
{
	/* Nothing to do if the state is already on the list. */
	if ( state->onStateList )
		return;

	/* Doing depth first, put state on the list. */
	state->onStateList = true;
	stateList.append( state );
	
	/* At this point transitions should only be in ranges. */
	assert( state->outSingle.length() == 0 );
	assert( state->defTrans == 0 );

	/* Recurse on everything ranges. */
	for ( RedTransList::Iter rtel = state->outRange; rtel.lte(); rtel++ ) {
		if ( rtel->value->targ != 0 )
			depthFirstOrdering( rtel->value->targ );
	}
}

/* Ordering states by transition connections. */
void RedFsmAp::depthFirstOrdering()
{
	/* Init on state list flags. */
	for ( RedStateList::Iter st = stateList; st.lte(); st++ )
		st->onStateList = false;
	
	/* Clear out the state list, we will rebuild it. */
	int stateListLen = stateList.length();
	stateList.abandon();

	/* Add back to the state list from the start state and all other entry
	 * points. */
	if ( startState != 0 )
		depthFirstOrdering( startState );
	for ( RedStateSet::Iter en = entryPoints; en.lte(); en++ )
		depthFirstOrdering( *en );
	if ( forcedErrorState )
		depthFirstOrdering( errState );
	
	/* Make sure we put everything back on. */
	assert( stateListLen == stateList.length() );
}

/* Assign state ids by appearance in the state list. */
void RedFsmAp::sequentialStateIds()
{
	/* Table based machines depend on the state numbers starting at zero. */
	nextStateId = 0;
	for ( RedStateList::Iter st = stateList; st.lte(); st++ )
		st->id = nextStateId++;
}

/* Stable sort the states by final state status. */
void RedFsmAp::sortStatesByFinal()
{
	/* Move forward through the list and throw final states onto the end. */
	RedStateAp *state = 0;
	RedStateAp *next = stateList.head;
	RedStateAp *last = stateList.tail;
	while ( state != last ) {
		/* Move forward and load up the next. */
		state = next;
		next = state->next;

		/* Throw to the end? */
		if ( state->isFinal ) {
			stateList.detach( state );
			stateList.append( state );
		}
	}
}

/* Assign state ids by final state state status. */
void RedFsmAp::sortStateIdsByFinal()
{
	/* Table based machines depend on this starting at zero. */
	nextStateId = 0;

	/* First pass to assign non final ids. */
	for ( RedStateList::Iter st = stateList; st.lte(); st++ ) {
		if ( ! st->isFinal ) 
			st->id = nextStateId++;
	}

	/* Second pass to assign final ids. */
	for ( RedStateList::Iter st = stateList; st.lte(); st++ ) {
		if ( st->isFinal ) 
			st->id = nextStateId++;
	}
}

struct CmpStateById
{
	static int compare( RedStateAp *st1, RedStateAp *st2 )
	{
		if ( st1->id < st2->id )
			return -1;
		else if ( st1->id > st2->id )
			return 1;
		else
			return 0;
	}
};

void RedFsmAp::sortByStateId()
{
	/* Make the array. */
	int pos = 0;
	RedStateAp **ptrList = new RedStateAp*[stateList.length()];
	for ( RedStateList::Iter st = stateList; st.lte(); st++, pos++ )
		ptrList[pos] = st;
	
	MergeSort<RedStateAp*, CmpStateById> mergeSort;
	mergeSort.sort( ptrList, stateList.length() );

	stateList.abandon();
	for ( int st = 0; st < pos; st++ )
		stateList.append( ptrList[st] );

	delete[] ptrList;
}

/* Find the final state with the lowest id. */
void RedFsmAp::findFirstFinState()
{
	for ( RedStateList::Iter st = stateList; st.lte(); st++ ) {
		if ( st->isFinal && (firstFinState == 0 || st->id < firstFinState->id) )
			firstFinState = st;
	}
}

void RedFsmAp::assignActionLocs()
{
	int nextLocation = 0;
	for ( GenActionTableMap::Iter act = actionMap; act.lte(); act++ ) {
		/* Store the loc, skip over the array and a null terminator. */
		act->location = nextLocation;
		nextLocation += act->key.length() + 1;		
	}
}

/* Check if we can extend the current range by displacing any ranges
 * ahead to the singles. */
bool RedFsmAp::canExtend( const RedTransList &list, int pos )
{
	/* Get the transition that we want to extend. */
	RedTransAp *extendTrans = list[pos].value;

	/* Look ahead in the transition list. */
	for ( int next = pos + 1; next < list.length(); pos++, next++ ) {
		/* If they are not continuous then cannot extend. */
		Key nextKey = list[next].lowKey;
		nextKey.decrement();
		if ( list[pos].highKey != nextKey )
			break;

		/* Check for the extenstion property. */
		if ( extendTrans == list[next].value )
			return true;

		/* If the span of the next element is more than one, then don't keep
		 * checking, it won't be moved to single. */
		unsigned long long nextSpan = keyOps->span( list[next].lowKey, list[next].highKey );
		if ( nextSpan > 1 )
			break;
	}
	return false;
}

/* Move ranges to the singles list. */
void RedFsmAp::moveTransToSingle( RedStateAp *state )
{
	RedTransList &range = state->outRange;
	RedTransList &single = state->outSingle;
	for ( int rpos = 0; rpos < range.length(); ) {
		/* Check if this is a range we can extend. */
		if ( canExtend( range, rpos ) ) {
			/* Transfer singles over. */
			while ( range[rpos].value != range[rpos+1].value ) {
				/* Transfer the range to single. */
				single.append( range[rpos+1] );
				range.remove( rpos+1 );
			}
			
			/* Extend. */
			range[rpos].highKey = range[rpos+1].highKey;
			range.remove( rpos+1 );
		}
		/* Maybe move it to the singles. */
		else if ( keyOps->span( range[rpos].lowKey, range[rpos].highKey ) == 1 ) {
			single.append( range[rpos] );
			range.remove( rpos );
		}
		else {
			/* Keeping it in the ranges. */
			rpos += 1;
		}
	}
}

/* Look through ranges and choose suitable single character transitions. */
void RedFsmAp::chooseSingle()
{
	/* Loop the states. */
	for ( RedStateList::Iter st = stateList; st.lte(); st++ ) {
		/* Rewrite the transition list taking out the suitable single
		 * transtions. */
		moveTransToSingle( st );
	}
}

void RedFsmAp::makeFlat()
{
	for ( RedStateList::Iter st = stateList; st.lte(); st++ ) {
		if ( st->stateCondList.length() == 0 ) {
			st->condLowKey = 0;
			st->condHighKey = 0;
		}
		else {
			st->condLowKey = st->stateCondList.head->lowKey;
			st->condHighKey = st->stateCondList.tail->highKey;

			unsigned long long span = keyOps->span( st->condLowKey, st->condHighKey );
			st->condList = new GenCondSpace*[ span ];
			memset( st->condList, 0, sizeof(GenCondSpace*)*span );

			for ( GenStateCondList::Iter sci = st->stateCondList; sci.lte(); sci++ ) {
				unsigned long long base, trSpan;
				base = keyOps->span( st->condLowKey, sci->lowKey )-1;
				trSpan = keyOps->span( sci->lowKey, sci->highKey );
				for ( unsigned long long pos = 0; pos < trSpan; pos++ )
					st->condList[base+pos] = sci->condSpace;
			}
		}

		if ( st->outRange.length() == 0 ) {
			st->lowKey = st->highKey = 0;
			st->transList = 0;
		}
		else {
			st->lowKey = st->outRange[0].lowKey;
			st->highKey = st->outRange[st->outRange.length()-1].highKey;
			unsigned long long span = keyOps->span( st->lowKey, st->highKey );
			st->transList = new RedTransAp*[ span ];
			memset( st->transList, 0, sizeof(RedTransAp*)*span );
			
			for ( RedTransList::Iter trans = st->outRange; trans.lte(); trans++ ) {
				unsigned long long base, trSpan;
				base = keyOps->span( st->lowKey, trans->lowKey )-1;
				trSpan = keyOps->span( trans->lowKey, trans->highKey );
				for ( unsigned long long pos = 0; pos < trSpan; pos++ )
					st->transList[base+pos] = trans->value;
			}

			/* Fill in the gaps with the default transition. */
			for ( unsigned long long pos = 0; pos < span; pos++ ) {
				if ( st->transList[pos] == 0 )
					st->transList[pos] = st->defTrans;
			}
		}
	}
}


/* A default transition has been picked, move it from the outRange to the
 * default pointer. */
void RedFsmAp::moveToDefault( RedTransAp *defTrans, RedStateAp *state )
{
	/* Rewrite the outRange, omitting any ranges that use 
	 * the picked default. */
	RedTransList outRange;
	for ( RedTransList::Iter rtel = state->outRange; rtel.lte(); rtel++ ) {
		/* If it does not take the default, copy it over. */
		if ( rtel->value != defTrans )
			outRange.append( *rtel );
	}

	/* Save off the range we just created into the state's range. */
	state->outRange.transfer( outRange );

	/* Store the default. */
	state->defTrans = defTrans;
}

bool RedFsmAp::alphabetCovered( RedTransList &outRange )
{
	/* Cannot cover without any out ranges. */
	if ( outRange.length() == 0 )
		return false;

	/* If the first range doesn't start at the the lower bound then the
	 * alphabet is not covered. */
	RedTransList::Iter rtel = outRange;
	if ( keyOps->minKey < rtel->lowKey )
		return false;

	/* Check that every range is next to the previous one. */
	rtel.increment();
	for ( ; rtel.lte(); rtel++ ) {
		Key highKey = rtel[-1].highKey;
		highKey.increment();
		if ( highKey != rtel->lowKey )
			return false;
	}

	/* The last must extend to the upper bound. */
	RedTransEl *last = &outRange[outRange.length()-1];
	if ( last->highKey < keyOps->maxKey )
		return false;

	return true;
}

RedTransAp *RedFsmAp::chooseDefaultSpan( RedStateAp *state )
{
	/* Make a set of transitions from the outRange. */
	RedTransSet stateTransSet;
	for ( RedTransList::Iter rtel = state->outRange; rtel.lte(); rtel++ )
		stateTransSet.insert( rtel->value );
	
	/* For each transition in the find how many alphabet characters the
	 * transition spans. */
	unsigned long long *span = new unsigned long long[stateTransSet.length()];
	memset( span, 0, sizeof(unsigned long long) * stateTransSet.length() );
	for ( RedTransList::Iter rtel = state->outRange; rtel.lte(); rtel++ ) {
		/* Lookup the transition in the set. */
		RedTransAp **inSet = stateTransSet.find( rtel->value );
		int pos = inSet - stateTransSet.data;
		span[pos] += keyOps->span( rtel->lowKey, rtel->highKey );
	}

	/* Find the max span, choose it for making the default. */
	RedTransAp *maxTrans = 0;
	unsigned long long maxSpan = 0;
	for ( RedTransSet::Iter rtel = stateTransSet; rtel.lte(); rtel++ ) {
		if ( span[rtel.pos()] > maxSpan ) {
			maxSpan = span[rtel.pos()];
			maxTrans = *rtel;
		}
	}

	delete[] span;
	return maxTrans;
}

/* Pick default transitions from ranges for the states. */
void RedFsmAp::chooseDefaultSpan()
{
	/* Loop the states. */
	for ( RedStateList::Iter st = stateList; st.lte(); st++ ) {
		/* Only pick a default transition if the alphabet is covered. This
		 * avoids any transitions in the out range that go to error and avoids
		 * the need for an ERR state. */
		if ( alphabetCovered( st->outRange ) ) {
			/* Pick a default transition by largest span. */
			RedTransAp *defTrans = chooseDefaultSpan( st );

			/* Rewrite the transition list taking out the transition we picked
			 * as the default and store the default. */
			moveToDefault( defTrans, st );
		}
	}
}

RedTransAp *RedFsmAp::chooseDefaultGoto( RedStateAp *state )
{
	/* Make a set of transitions from the outRange. */
	RedTransSet stateTransSet;
	for ( RedTransList::Iter rtel = state->outRange; rtel.lte(); rtel++ ) {
		if ( rtel->value->targ == state->next )
			return rtel->value;
	}
	return 0;
}

void RedFsmAp::chooseDefaultGoto()
{
	/* Loop the states. */
	for ( RedStateList::Iter st = stateList; st.lte(); st++ ) {
		/* Pick a default transition. */
		RedTransAp *defTrans = chooseDefaultGoto( st );
		if ( defTrans == 0 )
			defTrans = chooseDefaultSpan( st );

		/* Rewrite the transition list taking out the transition we picked
		 * as the default and store the default. */
		moveToDefault( defTrans, st );
	}
}

RedTransAp *RedFsmAp::chooseDefaultNumRanges( RedStateAp *state )
{
	/* Make a set of transitions from the outRange. */
	RedTransSet stateTransSet;
	for ( RedTransList::Iter rtel = state->outRange; rtel.lte(); rtel++ )
		stateTransSet.insert( rtel->value );
	
	/* For each transition in the find how many ranges use the transition. */
	int *numRanges = new int[stateTransSet.length()];
	memset( numRanges, 0, sizeof(int) * stateTransSet.length() );
	for ( RedTransList::Iter rtel = state->outRange; rtel.lte(); rtel++ ) {
		/* Lookup the transition in the set. */
		RedTransAp **inSet = stateTransSet.find( rtel->value );
		numRanges[inSet - stateTransSet.data] += 1;
	}

	/* Find the max number of ranges. */
	RedTransAp *maxTrans = 0;
	int maxNumRanges = 0;
	for ( RedTransSet::Iter rtel = stateTransSet; rtel.lte(); rtel++ ) {
		if ( numRanges[rtel.pos()] > maxNumRanges ) {
			maxNumRanges = numRanges[rtel.pos()];
			maxTrans = *rtel;
		}
	}

	delete[] numRanges;
	return maxTrans;
}

void RedFsmAp::chooseDefaultNumRanges()
{
	/* Loop the states. */
	for ( RedStateList::Iter st = stateList; st.lte(); st++ ) {
		/* Pick a default transition. */
		RedTransAp *defTrans = chooseDefaultNumRanges( st );

		/* Rewrite the transition list taking out the transition we picked
		 * as the default and store the default. */
		moveToDefault( defTrans, st );
	}
}

RedTransAp *RedFsmAp::getErrorTrans( )
{
	/* If the error trans has not been made aready, make it. */
	if ( errTrans == 0 ) {
		/* This insert should always succeed since no transition created by
		 * the user can point to the error state. */
		errTrans = new RedTransAp( getErrorState(), 0, nextTransId++ );
		RedTransAp *inRes = transSet.insert( errTrans );
		assert( inRes != 0 );
	}
	return errTrans;
}

RedStateAp *RedFsmAp::getErrorState()
{
	/* Something went wrong. An error state is needed but one was not supplied
	 * by the frontend. */
	assert( errState != 0 );
	return errState;
}


RedTransAp *RedFsmAp::allocateTrans( RedStateAp *targ, RedAction *action )
{
	/* Create a reduced trans and look for it in the transiton set. */
	RedTransAp redTrans( targ, action, 0 );
	RedTransAp *inDict = transSet.find( &redTrans );
	if ( inDict == 0 ) {
		inDict = new RedTransAp( targ, action, nextTransId++ );
		transSet.insert( inDict );
	}
	return inDict;
}

void RedFsmAp::partitionFsm( int nparts )
{
	/* At this point the states are ordered by a depth-first traversal. We
	 * will allocate to partitions based on this ordering. */
	this->nParts = nparts;
	int partSize = stateList.length() / nparts;
	int remainder = stateList.length() % nparts;
	int numInPart = partSize;
	int partition = 0;
	if ( remainder-- > 0 )
		numInPart += 1;
	for ( RedStateList::Iter st = stateList; st.lte(); st++ ) {
		st->partition = partition;

		numInPart -= 1;
		if ( numInPart == 0 ) {
			partition += 1;
			numInPart = partSize;
			if ( remainder-- > 0 )
				numInPart += 1;
		}
	}
}

void RedFsmAp::setInTrans()
{
	/* First pass counts the number of transitions. */
	for ( TransApSet::Iter trans = transSet; trans.lte(); trans++ )
		trans->targ->numInTrans += 1;

	/* Pass over states to allocate the needed memory. Reset the counts so we
	 * can use them as the current size. */
	for ( RedStateList::Iter st = stateList; st.lte(); st++ ) {
		st->inTrans = new RedTransAp*[st->numInTrans];
		st->numInTrans = 0;
	}

	/* Second pass over transitions copies pointers into the in trans list. */
	for ( TransApSet::Iter trans = transSet; trans.lte(); trans++ )
		trans->targ->inTrans[trans->targ->numInTrans++] = trans;
}
