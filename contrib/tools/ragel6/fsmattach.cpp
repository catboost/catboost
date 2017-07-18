/*
 *  Copyright 2001 Adrian Thurston <thurston@complang.org>
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

#include <string.h>
#include <assert.h>
#include "fsmgraph.h"

#include <iostream>
using namespace std;

/* Insert a transition into an inlist. The head must be supplied. */
void FsmAp::attachToInList( StateAp *from, StateAp *to, 
		TransAp *&head, TransAp *trans )
{
	trans->ilnext = head;
	trans->ilprev = 0;

	/* If in trans list is not empty, set the head->prev to trans. */
	if ( head != 0 )
		head->ilprev = trans;

	/* Now insert ourselves at the front of the list. */
	head = trans;

	/* Keep track of foreign transitions for from and to. */
	if ( from != to ) {
		if ( misfitAccounting ) {
			/* If the number of foreign in transitions is about to go up to 1 then
			 * move it from the misfit list to the main list. */
			if ( to->foreignInTrans == 0 )
				stateList.append( misfitList.detach( to ) );
		}
		
		to->foreignInTrans += 1;
	}
};

/* Detach a transition from an inlist. The head of the inlist must be supplied. */
void FsmAp::detachFromInList( StateAp *from, StateAp *to, 
		TransAp *&head, TransAp *trans )
{
	/* Detach in the inTransList. */
	if ( trans->ilprev == 0 ) 
		head = trans->ilnext; 
	else
		trans->ilprev->ilnext = trans->ilnext; 

	if ( trans->ilnext != 0 )
		trans->ilnext->ilprev = trans->ilprev; 
	
	/* Keep track of foreign transitions for from and to. */
	if ( from != to ) {
		to->foreignInTrans -= 1;
		
		if ( misfitAccounting ) {
			/* If the number of foreign in transitions goes down to 0 then move it
			 * from the main list to the misfit list. */
			if ( to->foreignInTrans == 0 )
				misfitList.append( stateList.detach( to ) );
		}
	}
}

/* Attach states on the default transition, range list or on out/in list key.
 * First makes a new transition. If there is already a transition out from
 * fromState on the default, then will assertion fail. */
TransAp *FsmAp::attachNewTrans( StateAp *from, StateAp *to, Key lowKey, Key highKey )
{
	/* Make the new transition. */
	TransAp *retVal = new TransAp();

	/* The transition is now attached. Remember the parties involved. */
	retVal->fromState = from;
	retVal->toState = to;

	/* Make the entry in the out list for the transitions. */
	from->outList.append( retVal );

	/* Set the the keys of the new trans. */
	retVal->lowKey = lowKey;
	retVal->highKey = highKey;

	/* Attach using inList as the head pointer. */
	if ( to != 0 )
		attachToInList( from, to, to->inList.head, retVal );

	return retVal;
}

/* Attach for range lists or for the default transition.  This attach should
 * be used when a transition already is allocated and must be attached to a
 * target state.  Does not handle adding the transition into the out list. */
void FsmAp::attachTrans( StateAp *from, StateAp *to, TransAp *trans )
{
	assert( trans->fromState == 0 && trans->toState == 0 );
	trans->fromState = from;
	trans->toState = to;

	if ( to != 0 ) { 
		/* Attach using the inList pointer as the head pointer. */
		attachToInList( from, to, to->inList.head, trans );
	}
}

/* Redirect a transition away from error and towards some state. This is just
 * like attachTrans except it requires fromState to be set and does not touch
 * it. */
void FsmAp::redirectErrorTrans( StateAp *from, StateAp *to, TransAp *trans )
{
	assert( trans->fromState != 0 && trans->toState == 0 );
	trans->toState = to;

	if ( to != 0 ) { 
		/* Attach using the inList pointer as the head pointer. */
		attachToInList( from, to, to->inList.head, trans );
	}
}

/* Detach for out/in lists or for default transition. */
void FsmAp::detachTrans( StateAp *from, StateAp *to, TransAp *trans )
{
	assert( trans->fromState == from && trans->toState == to );
	trans->fromState = 0;
	trans->toState = 0;

	if ( to != 0 ) {
		/* Detach using to's inList pointer as the head. */
		detachFromInList( from, to, to->inList.head, trans );
	}
}


/* Detach a state from the graph. Detaches and deletes transitions in and out
 * of the state. Empties inList and outList. Removes the state from the final
 * state set. A detached state becomes useless and should be deleted. */
void FsmAp::detachState( StateAp *state )
{
	/* Detach the in transitions from the inList list of transitions. */
	while ( state->inList.head != 0 ) {
		/* Get pointers to the trans and the state. */
		TransAp *trans = state->inList.head;
		StateAp *fromState = trans->fromState;

		/* Detach the transitions from the source state. */
		detachTrans( fromState, state, trans );

		/* Ok to delete the transition. */
		fromState->outList.detach( trans );
		delete trans;
	}

	/* Remove the entry points in on the machine. */
	while ( state->entryIds.length() > 0 )
		unsetEntry( state->entryIds[0], state );

	/* Detach out range transitions. */
	for ( TransList::Iter trans = state->outList; trans.lte(); ) {
		TransList::Iter next = trans.next();
		detachTrans( state, trans->toState, trans );
		delete trans;
		trans = next;
	}

	/* Delete all of the out range pointers. */
	state->outList.abandon();

	/* Unset final stateness before detaching from graph. */
	if ( state->stateBits & STB_ISFINAL )
		finStateSet.remove( state );
}


/* Duplicate a transition. Makes a new transition that is attached to the same
 * dest as srcTrans. The new transition has functions and priority taken from
 * srcTrans. Used for merging a transition in to a free spot. The trans can
 * just be dropped in. It does not conflict with an existing trans and need
 * not be crossed. Returns the new transition. */
TransAp *FsmAp::dupTrans( StateAp *from, TransAp *srcTrans )
{
	/* Make a new transition. */
	TransAp *newTrans = new TransAp();

	/* We can attach the transition, one does not exist. */
	attachTrans( from, srcTrans->toState, newTrans );
		
	/* Call the user callback to add in the original source transition. */
	addInTrans( newTrans, srcTrans );

	return newTrans;
}

/* In crossing, src trans and dest trans both go to existing states. Make one
 * state from the sets of states that src and dest trans go to. */
TransAp *FsmAp::fsmAttachStates( MergeData &md, StateAp *from,
			TransAp *destTrans, TransAp *srcTrans )
{
	/* The priorities are equal. We must merge the transitions. Does the
	 * existing trans go to the state we are to attach to? ie, are we to
	 * simply double up the transition? */
	StateAp *toState = srcTrans->toState;
	StateAp *existingState = destTrans->toState;

	if ( existingState == toState ) {
		/* The transition is a double up to the same state.  Copy the src
		 * trans into itself. We don't need to merge in the from out trans
		 * data, that was done already. */
		addInTrans( destTrans, srcTrans );
	}
	else {
		/* The trans is not a double up. Dest trans cannot be the same as src
		 * trans. Set up the state set. */
		StateSet stateSet;

		/* We go to all the states the existing trans goes to, plus... */
		if ( existingState->stateDictEl == 0 )
			stateSet.insert( existingState );
		else
			stateSet.insert( existingState->stateDictEl->stateSet );

		/* ... all the states that we have been told to go to. */
		if ( toState->stateDictEl == 0 )
			stateSet.insert( toState );
		else
			stateSet.insert( toState->stateDictEl->stateSet );

		/* Look for the state. If it is not there already, make it. */
		StateDictEl *lastFound;
		if ( md.stateDict.insert( stateSet, &lastFound ) ) {
			/* Make a new state representing the combination of states in
			 * stateSet. It gets added to the fill list.  This means that we
			 * need to fill in it's transitions sometime in the future.  We
			 * don't do that now (ie, do not recurse). */
			StateAp *combinState = addState();

			/* Link up the dict element and the state. */
			lastFound->targState = combinState;
			combinState->stateDictEl = lastFound;

			/* Add to the fill list. */
			md.fillListAppend( combinState );
		}

		/* Get the state insertted/deleted. */
		StateAp *targ = lastFound->targState;

		/* Detach the state from existing state. */
		detachTrans( from, existingState, destTrans );

		/* Re-attach to the new target. */
		attachTrans( from, targ, destTrans );

		/* Add in src trans to the existing transition that we redirected to
		 * the new state. We don't need to merge in the from out trans data,
		 * that was done already. */
		addInTrans( destTrans, srcTrans );
	}

	return destTrans;
}

/* Two transitions are to be crossed, handle the possibility of either going
 * to the error state. */
TransAp *FsmAp::mergeTrans( MergeData &md, StateAp *from,
			TransAp *destTrans, TransAp *srcTrans )
{
	TransAp *retTrans = 0;
	if ( destTrans->toState == 0 && srcTrans->toState == 0 ) {
		/* Error added into error. */
		addInTrans( destTrans, srcTrans );
		retTrans = destTrans;
	}
	else if ( destTrans->toState == 0 && srcTrans->toState != 0 ) {
		/* Non error added into error we need to detach and reattach, */
		detachTrans( from, destTrans->toState, destTrans );
		attachTrans( from, srcTrans->toState, destTrans );
		addInTrans( destTrans, srcTrans );
		retTrans = destTrans;
	}
	else if ( srcTrans->toState == 0 ) {
		/* Dest goes somewhere but src doesn't, just add it it in. */
		addInTrans( destTrans, srcTrans );
		retTrans = destTrans;
	}
	else {
		/* Both go somewhere, run the actual cross. */
		retTrans = fsmAttachStates( md, from, destTrans, srcTrans );
	}

	return retTrans;
}

/* Find the trans with the higher priority. If src is lower priority then dest then
 * src is ignored. If src is higher priority than dest, then src overwrites dest. If
 * the priorities are equal, then they are merged. */
TransAp *FsmAp::crossTransitions( MergeData &md, StateAp *from,
		TransAp *destTrans, TransAp *srcTrans )
{
	TransAp *retTrans;

	/* Compare the priority of the dest and src transitions. */
	int compareRes = comparePrior( destTrans->priorTable, srcTrans->priorTable );
	if ( compareRes < 0 ) {
		/* Src trans has a higher priority than dest, src overwrites dest.
		 * Detach dest and return a copy of src. */
		detachTrans( from, destTrans->toState, destTrans );
		retTrans = dupTrans( from, srcTrans );
	}
	else if ( compareRes > 0 ) {
		/* The dest trans has a higher priority, use dest. */
		retTrans = destTrans;
	}
	else {
		/* Src trans and dest trans have the same priority, they must be merged. */
		retTrans = mergeTrans( md, from, destTrans, srcTrans );
	}

	/* Return the transition that resulted from the cross. */
	return retTrans;
}

/* Copy the transitions in srcList to the outlist of dest. The srcList should
 * not be the outList of dest, otherwise you would be copying the contents of
 * srcList into itself as it's iterated: bad news. */
void FsmAp::outTransCopy( MergeData &md, StateAp *dest, TransAp *srcList )
{
	/* The destination list. */
	TransList destList;

	/* Set up an iterator to stop at breaks. */
	PairIter<TransAp> outPair( dest->outList.head, srcList );
	for ( ; !outPair.end(); outPair++ ) {
		switch ( outPair.userState ) {
		case RangeInS1: {
			/* The pair iter is the authority on the keys. It may have needed
			 * to break the dest range. */
			TransAp *destTrans = outPair.s1Tel.trans;
			destTrans->lowKey = outPair.s1Tel.lowKey;
			destTrans->highKey = outPair.s1Tel.highKey;
			destList.append( destTrans );
			break;
		}
		case RangeInS2: {
			/* Src range may get crossed with dest's default transition. */
			TransAp *newTrans = dupTrans( dest, outPair.s2Tel.trans );

			/* Set up the transition's keys and append to the dest list. */
			newTrans->lowKey = outPair.s2Tel.lowKey;
			newTrans->highKey = outPair.s2Tel.highKey;
			destList.append( newTrans );
			break;
		}
		case RangeOverlap: {
			/* Exact overlap, cross them. */
			TransAp *newTrans = crossTransitions( md, dest,
				outPair.s1Tel.trans, outPair.s2Tel.trans );

			/* Set up the transition's keys and append to the dest list. */
			newTrans->lowKey = outPair.s1Tel.lowKey;
			newTrans->highKey = outPair.s1Tel.highKey;
			destList.append( newTrans );
			break;
		}
		case BreakS1: {
			/* Since we are always writing to the dest trans, the dest needs
			 * to be copied when it is broken. The copy goes into the first
			 * half of the break to "break it off". */
			outPair.s1Tel.trans = dupTrans( dest, outPair.s1Tel.trans );
			break;
		}
		case BreakS2:
			break;
		}
	}

	/* Abandon the old outList and transfer destList into it. */
	dest->outList.transfer( destList );
}


/* Move all the transitions that go into src so that they go into dest.  */
void FsmAp::inTransMove( StateAp *dest, StateAp *src )
{
	/* Do not try to move in trans to and from the same state. */
	assert( dest != src );

	/* If src is the start state, dest becomes the start state. */
	if ( src == startState ) {
		unsetStartState();
		setStartState( dest );
	}

	/* For each entry point into, create an entry point into dest, when the
	 * state is detached, the entry points to src will be removed. */
	for ( EntryIdSet::Iter enId = src->entryIds; enId.lte(); enId++ )
		changeEntry( *enId, dest, src );

	/* Move the transitions in inList. */
	while ( src->inList.head != 0 ) {
		/* Get trans and from state. */
		TransAp *trans = src->inList.head;
		StateAp *fromState = trans->fromState;

		/* Detach from src, reattach to dest. */
		detachTrans( fromState, src, trans );
		attachTrans( fromState, dest, trans );
	}
}
