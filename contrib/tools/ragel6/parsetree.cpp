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

#include <iostream>
#include <iomanip>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>

/* Parsing. */
#include "ragel.h"
#include "rlparse.h"
#include "parsetree.h"

using namespace std;
ostream &operator<<( ostream &out, const NameRef &nameRef );
ostream &operator<<( ostream &out, const NameInst &nameInst );

/* Convert the literal string which comes in from the scanner into an array of
 * characters with escapes and options interpreted. Also null terminates the
 * string. Though this null termination should not be relied on for
 * interpreting literals in the parser because the string may contain \0 */
char *prepareLitString( const InputLoc &loc, const char *data, long length, 
		long &resLen, bool &caseInsensitive )
{
	char *resData = new char[length+1];
	caseInsensitive = false;

	const char *src = data + 1;
	const char *end = data + length - 1;

	while ( *end != '\'' && *end != '\"' ) {
		if ( *end == 'i' )
			caseInsensitive = true;
		else {
			error( loc ) << "literal string '" << *end << 
					"' option not supported" << endl;
		}
		end -= 1;
	}

	char *dest = resData;
	long len = 0;
	while ( src != end ) {
		if ( *src == '\\' ) {
			switch ( src[1] ) {
			case '0': dest[len++] = '\0'; break;
			case 'a': dest[len++] = '\a'; break;
			case 'b': dest[len++] = '\b'; break;
			case 't': dest[len++] = '\t'; break;
			case 'n': dest[len++] = '\n'; break;
			case 'v': dest[len++] = '\v'; break;
			case 'f': dest[len++] = '\f'; break;
			case 'r': dest[len++] = '\r'; break;
			case '\n':  break;
			default: dest[len++] = src[1]; break;
			}
			src += 2;
		}
		else {
			dest[len++] = *src++;
		}
	}

	resLen = len;
	resData[resLen] = 0;
	return resData;
}

FsmAp *VarDef::walk( ParseData *pd )
{
	/* We enter into a new name scope. */
	NameFrame nameFrame = pd->enterNameScope( true, 1 );

	/* Recurse on the expression. */
	FsmAp *rtnVal = machineDef->walk( pd );
	
	/* Do the tranfer of local error actions. */
	LocalErrDictEl *localErrDictEl = pd->localErrDict.find( name );
	if ( localErrDictEl != 0 ) {
		for ( StateList::Iter state = rtnVal->stateList; state.lte(); state++ )
			rtnVal->transferErrorActions( state, localErrDictEl->value );
	}

	/* If the expression below is a join operation with multiple expressions
	 * then it just had epsilon transisions resolved. If it is a join
	 * with only a single expression then run the epsilon op now. */
	if ( machineDef->type == MachineDef::JoinType && machineDef->join->exprList.length() == 1 )
		rtnVal->epsilonOp();

	/* We can now unset entry points that are not longer used. */
	pd->unsetObsoleteEntries( rtnVal );

	/* If the name of the variable is referenced then add the entry point to
	 * the graph. */
	if ( pd->curNameInst->numRefs > 0 )
		rtnVal->setEntry( pd->curNameInst->id, rtnVal->startState );

	/* Pop the name scope. */
	pd->popNameScope( nameFrame );
	return rtnVal;
}

void VarDef::makeNameTree( const InputLoc &loc, ParseData *pd )
{
	/* The variable definition enters a new scope. */
	NameInst *prevNameInst = pd->curNameInst;
	pd->curNameInst = pd->addNameInst( loc, name, false );

	if ( machineDef->type == MachineDef::LongestMatchType )
		pd->curNameInst->isLongestMatch = true;

	/* Recurse. */
	machineDef->makeNameTree( pd );

	/* The name scope ends, pop the name instantiation. */
	pd->curNameInst = prevNameInst;
}

void VarDef::resolveNameRefs( ParseData *pd )
{
	/* Entering into a new scope. */
	NameFrame nameFrame = pd->enterNameScope( true, 1 );

	/* Recurse. */
	machineDef->resolveNameRefs( pd );
	
	/* The name scope ends, pop the name instantiation. */
	pd->popNameScope( nameFrame );
}

InputLoc LongestMatchPart::getLoc()
{ 
	return action != 0 ? action->loc : semiLoc;
}

/*
 * If there are any LMs then all of the following entry points must reset
 * tokstart:
 *
 *  1. fentry(StateRef)
 *  2. ftoto(StateRef), fcall(StateRef), fnext(StateRef)
 *  3. targt of any transition that has an fcall (the return loc).
 *  4. start state of all longest match routines.
 */

Action *LongestMatch::newAction( ParseData *pd, const InputLoc &loc, 
		const char *name, InlineList *inlineList )
{
	Action *action = new Action( loc, name, inlineList, pd->nextCondId++ );
	action->actionRefs.append( pd->curNameInst );
	pd->actionList.append( action );
	action->isLmAction = true;
	return action;
}

void LongestMatch::makeActions( ParseData *pd )
{
	/* Make actions that set the action id. */
	for ( LmPartList::Iter lmi = *longestMatchList; lmi.lte(); lmi++ ) {
		/* For each part create actions for setting the match type.  We need
		 * to do this so that the actions will go into the actionIndex. */
		InlineList *inlineList = new InlineList;
		inlineList->append( new InlineItem( lmi->getLoc(), this, lmi, 
				InlineItem::LmSetActId ) );
		char *actName = new char[50];
		sprintf( actName, "store%i", lmi->longestMatchId );
		lmi->setActId = newAction( pd, lmi->getLoc(), actName, inlineList );
	}

	/* Make actions that execute the user action and restart on the last
	 * character. */
	for ( LmPartList::Iter lmi = *longestMatchList; lmi.lte(); lmi++ ) {
		/* For each part create actions for setting the match type.  We need
		 * to do this so that the actions will go into the actionIndex. */
		InlineList *inlineList = new InlineList;
		inlineList->append( new InlineItem( lmi->getLoc(), this, lmi, 
				InlineItem::LmOnLast ) );
		char *actName = new char[50];
		sprintf( actName, "last%i", lmi->longestMatchId );
		lmi->actOnLast = newAction( pd, lmi->getLoc(), actName, inlineList );
	}

	/* Make actions that execute the user action and restart on the next
	 * character.  These actions will set tokend themselves (it is the current
	 * char). */
	for ( LmPartList::Iter lmi = *longestMatchList; lmi.lte(); lmi++ ) {
		/* For each part create actions for setting the match type.  We need
		 * to do this so that the actions will go into the actionIndex. */
		InlineList *inlineList = new InlineList;
		inlineList->append( new InlineItem( lmi->getLoc(), this, lmi, 
				InlineItem::LmOnNext ) );
		char *actName = new char[50];
		sprintf( actName, "next%i", lmi->longestMatchId );
		lmi->actOnNext = newAction( pd, lmi->getLoc(), actName, inlineList );
	}

	/* Make actions that execute the user action and restart at tokend. These
	 * actions execute some time after matching the last char. */
	for ( LmPartList::Iter lmi = *longestMatchList; lmi.lte(); lmi++ ) {
		/* For each part create actions for setting the match type.  We need
		 * to do this so that the actions will go into the actionIndex. */
		InlineList *inlineList = new InlineList;
		inlineList->append( new InlineItem( lmi->getLoc(), this, lmi, 
				InlineItem::LmOnLagBehind ) );
		char *actName = new char[50];
		sprintf( actName, "lag%i", lmi->longestMatchId );
		lmi->actLagBehind = newAction( pd, lmi->getLoc(), actName, inlineList );
	}

	InputLoc loc;
	loc.line = 1;
	loc.col = 1;
	loc.fileName = "NONE";

	/* Create the error action. */
	InlineList *il6 = new InlineList;
	il6->append( new InlineItem( loc, this, 0, InlineItem::LmSwitch ) );
	lmActSelect = newAction( pd, loc, "switch", il6 );
}

void LongestMatch::findName( ParseData *pd )
{
	NameInst *nameInst = pd->curNameInst;
	while ( nameInst->name == 0 ) {
		nameInst = nameInst->parent;
		/* Since every machine must must have a name, we should always find a
		 * name for the longest match. */
		assert( nameInst != 0 );
	}
	name = nameInst->name;
}

void LongestMatch::makeNameTree( ParseData *pd )
{
	/* Create an anonymous scope for the longest match. Will be used for
	 * restarting machine after matching a token. */
	NameInst *prevNameInst = pd->curNameInst;
	pd->curNameInst = pd->addNameInst( loc, 0, false );

	/* Recurse into all parts of the longest match operator. */
	for ( LmPartList::Iter lmi = *longestMatchList; lmi.lte(); lmi++ )
		lmi->join->makeNameTree( pd );

	/* Traverse the name tree upwards to find a name for this lm. */
	findName( pd );

	/* Also make the longest match's actions at this point. */
	makeActions( pd );

	/* The name scope ends, pop the name instantiation. */
	pd->curNameInst = prevNameInst;
}

void LongestMatch::resolveNameRefs( ParseData *pd )
{
	/* The longest match gets its own name scope. */
	NameFrame nameFrame = pd->enterNameScope( true, 1 );

	/* Take an action reference for each longest match item and recurse. */
	for ( LmPartList::Iter lmi = *longestMatchList; lmi.lte(); lmi++ ) {
		/* Record the reference if the item has an action. */
		if ( lmi->action != 0 )
			lmi->action->actionRefs.append( pd->localNameScope );

		/* Recurse down the join. */
		lmi->join->resolveNameRefs( pd );
	}

	/* The name scope ends, pop the name instantiation. */
	pd->popNameScope( nameFrame );
}

void LongestMatch::restart( FsmAp *graph, TransAp *trans )
{
	StateAp *fromState = trans->fromState;
	graph->detachTrans( fromState, trans->toState, trans );
	graph->attachTrans( fromState, graph->startState, trans );
}

void LongestMatch::runLongestMatch( ParseData *pd, FsmAp *graph )
{
	graph->markReachableFromHereStopFinal( graph->startState );
	for ( StateList::Iter ms = graph->stateList; ms.lte(); ms++ ) {
		if ( ms->stateBits & STB_ISMARKED ) {
			ms->lmItemSet.insert( 0 );
			ms->stateBits &= ~ STB_ISMARKED;
		}
	}

	/* Transfer the first item of non-empty lmAction tables to the item sets
	 * of the states that follow. Exclude states that have no transitions out.
	 * This must happen on a separate pass so that on each iteration of the
	 * next pass we have the item set entries from all lmAction tables. */
	for ( StateList::Iter st = graph->stateList; st.lte(); st++ ) {
		for ( TransList::Iter trans = st->outList; trans.lte(); trans++ ) {
			if ( trans->lmActionTable.length() > 0 ) {
				LmActionTableEl *lmAct = trans->lmActionTable.data;
				StateAp *toState = trans->toState;
				assert( toState );

				/* Can only optimize this if there are no transitions out.
				 * Note there can be out transitions going nowhere with
				 * actions and they too must inhibit this optimization. */
				if ( toState->outList.length() > 0 ) {
					/* Fill the item sets. */
					graph->markReachableFromHereStopFinal( toState );
					for ( StateList::Iter ms = graph->stateList; ms.lte(); ms++ ) {
						if ( ms->stateBits & STB_ISMARKED ) {
							ms->lmItemSet.insert( lmAct->value );
							ms->stateBits &= ~ STB_ISMARKED;
						}
					}
				}
			}
		}
	}

	/* The lmItem sets are now filled, telling us which longest match rules
	 * can succeed in which states. First determine if we need to make sure
	 * act is defaulted to zero. We need to do this if there are any states
	 * with lmItemSet.length() > 1 and NULL is included. That is, that the
	 * switch may get called when in fact nothing has been matched. */
	int maxItemSetLength = 0;
	graph->markReachableFromHereStopFinal( graph->startState );
	for ( StateList::Iter ms = graph->stateList; ms.lte(); ms++ ) {
		if ( ms->stateBits & STB_ISMARKED ) {
			if ( ms->lmItemSet.length() > maxItemSetLength )
				maxItemSetLength = ms->lmItemSet.length();
			ms->stateBits &= ~ STB_ISMARKED;
		}
	}

	/* The actions executed on starting to match a token. */
	graph->isolateStartState();
	graph->startState->toStateActionTable.setAction( pd->initTokStartOrd, pd->initTokStart );
	graph->startState->fromStateActionTable.setAction( pd->setTokStartOrd, pd->setTokStart );
	if ( maxItemSetLength > 1 ) {
		/* The longest match action switch may be called when tokens are
		 * matched, in which case act must be initialized, there must be a
		 * case to handle the error, and the generated machine will require an
		 * error state. */
		lmSwitchHandlesError = true;
		pd->lmRequiresErrorState = true;
		graph->startState->toStateActionTable.setAction( pd->initActIdOrd, pd->initActId );
	}

	/* The place to store transitions to restart. It maybe possible for the
	 * restarting to affect the searching through the graph that follows. For
	 * now take the safe route and save the list of transitions to restart
	 * until after all searching is done. */
	Vector<TransAp*> restartTrans;

	/* Set actions that do immediate token recognition, set the longest match part
	 * id and set the token ending. */
	for ( StateList::Iter st = graph->stateList; st.lte(); st++ ) {
		for ( TransList::Iter trans = st->outList; trans.lte(); trans++ ) {
			if ( trans->lmActionTable.length() > 0 ) {
				LmActionTableEl *lmAct = trans->lmActionTable.data;
				StateAp *toState = trans->toState;
				assert( toState );

				/* Can only optimize this if there are no transitions out.
				 * Note there can be out transitions going nowhere with
				 * actions and they too must inhibit this optimization. */
				if ( toState->outList.length() == 0 ) {
					/* Can execute the immediate action for the longest match
					 * part. Redirect the action to the start state.
					 *
					 * NOTE: When we need to inhibit on_last due to leaving
					 * actions the above test suffices. If the state has out
					 * actions then it will fail because the out action will
					 * have been transferred to an error transition, which
					 * makes the outlist non-empty. */
					trans->actionTable.setAction( lmAct->key, 
							lmAct->value->actOnLast );
					restartTrans.append( trans );
				}
				else {
					/* Look for non final states that have a non-empty item
					 * set. If these are present then we need to record the
					 * end of the token.  Also Find the highest item set
					 * length reachable from here (excluding at transtions to
					 * final states). */
					bool nonFinalNonEmptyItemSet = false;
					maxItemSetLength = 0;
					graph->markReachableFromHereStopFinal( toState );
					for ( StateList::Iter ms = graph->stateList; ms.lte(); ms++ ) {
						if ( ms->stateBits & STB_ISMARKED ) {
							if ( ms->lmItemSet.length() > 0 && !ms->isFinState() )
								nonFinalNonEmptyItemSet = true;
							if ( ms->lmItemSet.length() > maxItemSetLength )
								maxItemSetLength = ms->lmItemSet.length();
							ms->stateBits &= ~ STB_ISMARKED;
						}
					}

					/* If there are reachable states that are not final and
					 * have non empty item sets or that have an item set
					 * length greater than one then we need to set tokend
					 * because the error action that matches the token will
					 * require it. */
					if ( nonFinalNonEmptyItemSet || maxItemSetLength > 1 )
						trans->actionTable.setAction( pd->setTokEndOrd, pd->setTokEnd );

					/* Some states may not know which longest match item to
					 * execute, must set it. */
					if ( maxItemSetLength > 1 ) {
						/* There are transitions out, another match may come. */
						trans->actionTable.setAction( lmAct->key, 
								lmAct->value->setActId );
					}
				}
			}
		}
	}

	/* Now that all graph searching is done it certainly safe set the
	 * restarting. It may be safe above, however this must be verified. */
	for ( Vector<TransAp*>::Iter pt = restartTrans; pt.lte(); pt++ )
		restart( graph, *pt );

	int lmErrActionOrd = pd->curActionOrd++;

	/* Embed the error for recognizing a char. */
	for ( StateList::Iter st = graph->stateList; st.lte(); st++ ) {
		if ( st->lmItemSet.length() == 1 && st->lmItemSet[0] != 0 ) {
			if ( st->isFinState() ) {
				/* On error execute the onActNext action, which knows that
				 * the last character of the token was one back and restart. */
				graph->setErrorTarget( st, graph->startState, &lmErrActionOrd, 
						&st->lmItemSet[0]->actOnNext, 1 );
				st->eofActionTable.setAction( lmErrActionOrd, 
						st->lmItemSet[0]->actOnNext );
				st->eofTarget = graph->startState;
			}
			else {
				graph->setErrorTarget( st, graph->startState, &lmErrActionOrd, 
						&st->lmItemSet[0]->actLagBehind, 1 );
				st->eofActionTable.setAction( lmErrActionOrd, 
						st->lmItemSet[0]->actLagBehind );
				st->eofTarget = graph->startState;
			}
		}
		else if ( st->lmItemSet.length() > 1 ) {
			/* Need to use the select. Take note of which items the select
			 * is needed for so only the necessary actions are included. */
			for ( LmItemSet::Iter plmi = st->lmItemSet; plmi.lte(); plmi++ ) {
				if ( *plmi != 0 )
					(*plmi)->inLmSelect = true;
			}
			/* On error, execute the action select and go to the start state. */
			graph->setErrorTarget( st, graph->startState, &lmErrActionOrd, 
					&lmActSelect, 1 );
			st->eofActionTable.setAction( lmErrActionOrd, lmActSelect );
			st->eofTarget = graph->startState;
		}
	}
	
	/* Finally, the start state should be made final. */
	graph->setFinState( graph->startState );
}

void LongestMatch::transferScannerLeavingActions( FsmAp *graph )
{
	for ( StateList::Iter st = graph->stateList; st.lte(); st++ ) {
		if ( st->outActionTable.length() > 0 )
			graph->setErrorActions( st, st->outActionTable );
	}
}

FsmAp *LongestMatch::walk( ParseData *pd )
{
	/* The longest match has it's own name scope. */
	NameFrame nameFrame = pd->enterNameScope( true, 1 );

	/* Make each part of the longest match. */
	FsmAp **parts = new FsmAp*[longestMatchList->length()];
	LmPartList::Iter lmi = *longestMatchList; 
	for ( int i = 0; lmi.lte(); lmi++, i++ ) {
		/* Create the machine and embed the setting of the longest match id. */
		parts[i] = lmi->join->walk( pd );
		parts[i]->longMatchAction( pd->curActionOrd++, lmi );
	}

	/* Before we union the patterns we need to deal with leaving actions. They
	 * are transfered to error transitions out of the final states (like local
	 * error actions) and to eof actions. In the scanner we need to forbid
	 * on_last for any final state that has an leaving action. */
	for ( int i = 0; i < longestMatchList->length(); i++ )
		transferScannerLeavingActions( parts[i] );

	/* Union machines one and up with machine zero. The grammar dictates that
	 * there will always be at least one part. */
	FsmAp *rtnVal = parts[0];
	for ( int i = 1; i < longestMatchList->length(); i++ ) {
		rtnVal->unionOp( parts[i] );
		afterOpMinimize( rtnVal );
	}

	runLongestMatch( pd, rtnVal );

	/* Pop the name scope. */
	pd->popNameScope( nameFrame );

	delete[] parts;
	return rtnVal;
}

FsmAp *MachineDef::walk( ParseData *pd )
{
	FsmAp *rtnVal = 0;
	switch ( type ) {
	case JoinType:
		rtnVal = join->walk( pd );
		break;
	case LongestMatchType:
		rtnVal = longestMatch->walk( pd );
		break;
	case LengthDefType:
		condData->lastCondKey.increment();
		rtnVal = new FsmAp();
		rtnVal->concatFsm( condData->lastCondKey );
		break;
	}
	return rtnVal;
}

void MachineDef::makeNameTree( ParseData *pd )
{
	switch ( type ) {
	case JoinType:
		join->makeNameTree( pd );
		break;
	case LongestMatchType:
		longestMatch->makeNameTree( pd );
		break;
	case LengthDefType:
		break;
	}
}

void MachineDef::resolveNameRefs( ParseData *pd )
{
	switch ( type ) {
	case JoinType:
		join->resolveNameRefs( pd );
		break;
	case LongestMatchType:
		longestMatch->resolveNameRefs( pd );
		break;
	case LengthDefType:
		break;
	}
}


/* Construct with a location and the first expression. */
Join::Join( const InputLoc &loc, Expression *expr )
:
	loc(loc)
{
	exprList.append( expr );
}

/* Construct with a location and the first expression. */
Join::Join( Expression *expr )
{
	exprList.append( expr );
}

/* Walk an expression node. */
FsmAp *Join::walk( ParseData *pd )
{
	if ( exprList.length() > 1 )
		return walkJoin( pd );
	else
		return exprList.head->walk( pd );
}

/* There is a list of expressions to join. */
FsmAp *Join::walkJoin( ParseData *pd )
{
	/* We enter into a new name scope. */
	NameFrame nameFrame = pd->enterNameScope( true, 1 );

	/* Evaluate the machines. */
	FsmAp **fsms = new FsmAp*[exprList.length()];
	ExprList::Iter expr = exprList;
	for ( int e = 0; e < exprList.length(); e++, expr++ )
		fsms[e] = expr->walk( pd );
	
	/* Get the start and final names. Final is 
	 * guaranteed to exist, start is not. */
	NameInst *startName = pd->curNameInst->start;
	NameInst *finalName = pd->curNameInst->final;

	int startId = -1;
	if ( startName != 0 ) {
		/* Take note that there was an implicit link to the start machine. */
		pd->localNameScope->referencedNames.append( startName );
		startId = startName->id;
	}

	/* A final id of -1 indicates there is no epsilon that references the
	 * final state, therefor do not create one or set an entry point to it. */
	int finalId = -1;
	if ( finalName->numRefs > 0 )
		finalId = finalName->id;

	/* Join machines 1 and up onto machine 0. */
	FsmAp *retFsm = fsms[0];
	retFsm->joinOp( startId, finalId, fsms+1, exprList.length()-1 );

	/* We can now unset entry points that are not longer used. */
	pd->unsetObsoleteEntries( retFsm );

	/* Pop the name scope. */
	pd->popNameScope( nameFrame );

	delete[] fsms;
	return retFsm;
}

void Join::makeNameTree( ParseData *pd )
{
	if ( exprList.length() > 1 ) {
		/* Create the new anonymous scope. */
		NameInst *prevNameInst = pd->curNameInst;
		pd->curNameInst = pd->addNameInst( loc, 0, false );

		/* Join scopes need an implicit "final" target. */
		pd->curNameInst->final = new NameInst( InputLoc(), pd->curNameInst, "final", 
				pd->nextNameId++, false );

		/* Recurse into all expressions in the list. */
		for ( ExprList::Iter expr = exprList; expr.lte(); expr++ )
			expr->makeNameTree( pd );

		/* The name scope ends, pop the name instantiation. */
		pd->curNameInst = prevNameInst;
	}
	else {
		/* Recurse into the single expression. */
		exprList.head->makeNameTree( pd );
	}
}


void Join::resolveNameRefs( ParseData *pd )
{
	/* Branch on whether or not there is to be a join. */
	if ( exprList.length() > 1 ) {
		/* The variable definition enters a new scope. */
		NameFrame nameFrame = pd->enterNameScope( true, 1 );

		/* The join scope must contain a start label. */
		NameSet resolved = pd->resolvePart( pd->localNameScope, "start", true );
		if ( resolved.length() > 0 ) {
			/* Take the first. */
			pd->curNameInst->start = resolved[0];
			if ( resolved.length() > 1 ) {
				/* Complain about the multiple references. */
				error(loc) << "join operation has multiple start labels" << endl;
				errorStateLabels( resolved );
			}
		}

		/* Make sure there is a start label. */
		if ( pd->curNameInst->start != 0 ) {
			/* There is an implicit reference to start name. */
			pd->curNameInst->start->numRefs += 1;
		}
		else {
			/* No start label. */
			error(loc) << "join operation has no start label" << endl;
		}

		/* Recurse into all expressions in the list. */
		for ( ExprList::Iter expr = exprList; expr.lte(); expr++ )
			expr->resolveNameRefs( pd );

		/* The name scope ends, pop the name instantiation. */
		pd->popNameScope( nameFrame );
	}
	else {
		/* Recurse into the single expression. */
		exprList.head->resolveNameRefs( pd );
	}
}

/* Clean up after an expression node. */
Expression::~Expression()
{
	switch ( type ) {
		case OrType: case IntersectType: case SubtractType:
		case StrongSubtractType:
			delete expression;
			delete term;
			break;
		case TermType:
			delete term;
			break;
		case BuiltinType:
			break;
	}
}

/* Evaluate a single expression node. */
FsmAp *Expression::walk( ParseData *pd, bool lastInSeq )
{
	FsmAp *rtnVal = 0;
	switch ( type ) {
		case OrType: {
			/* Evaluate the expression. */
			rtnVal = expression->walk( pd, false );
			/* Evaluate the term. */
			FsmAp *rhs = term->walk( pd );
			/* Perform union. */
			rtnVal->unionOp( rhs );
			afterOpMinimize( rtnVal, lastInSeq );
			break;
		}
		case IntersectType: {
			/* Evaluate the expression. */
			rtnVal = expression->walk( pd );
			/* Evaluate the term. */
			FsmAp *rhs = term->walk( pd );
			/* Perform intersection. */
			rtnVal->intersectOp( rhs );
			afterOpMinimize( rtnVal, lastInSeq );
			break;
		}
		case SubtractType: {
			/* Evaluate the expression. */
			rtnVal = expression->walk( pd );
			/* Evaluate the term. */
			FsmAp *rhs = term->walk( pd );
			/* Perform subtraction. */
			rtnVal->subtractOp( rhs );
			afterOpMinimize( rtnVal, lastInSeq );
			break;
		}
		case StrongSubtractType: {
			/* Evaluate the expression. */
			rtnVal = expression->walk( pd );

			/* Evaluate the term and pad it with any* machines. */
			FsmAp *rhs = dotStarFsm( pd );
			FsmAp *termFsm = term->walk( pd );
			FsmAp *trailAnyStar = dotStarFsm( pd );
			rhs->concatOp( termFsm );
			rhs->concatOp( trailAnyStar );

			/* Perform subtraction. */
			rtnVal->subtractOp( rhs );
			afterOpMinimize( rtnVal, lastInSeq );
			break;
		}
		case TermType: {
			/* Return result of the term. */
			rtnVal = term->walk( pd );
			break;
		}
		case BuiltinType: {
			/* Duplicate the builtin. */
			rtnVal = makeBuiltin( builtin, pd );
			break;
		}
	}

	return rtnVal;
}

void Expression::makeNameTree( ParseData *pd )
{
	switch ( type ) {
	case OrType:
	case IntersectType:
	case SubtractType:
	case StrongSubtractType:
		expression->makeNameTree( pd );
		term->makeNameTree( pd );
		break;
	case TermType:
		term->makeNameTree( pd );
		break;
	case BuiltinType:
		break;
	}
}

void Expression::resolveNameRefs( ParseData *pd )
{
	switch ( type ) {
	case OrType:
	case IntersectType:
	case SubtractType:
	case StrongSubtractType:
		expression->resolveNameRefs( pd );
		term->resolveNameRefs( pd );
		break;
	case TermType:
		term->resolveNameRefs( pd );
		break;
	case BuiltinType:
		break;
	}
}

/* Clean up after a term node. */
Term::~Term()
{
	switch ( type ) {
		case ConcatType:
		case RightStartType:
		case RightFinishType:
		case LeftType:
			delete term;
			delete factorWithAug;
			break;
		case FactorWithAugType:
			delete factorWithAug;
			break;
	}
}

/* Evaluate a term node. */
FsmAp *Term::walk( ParseData *pd, bool lastInSeq )
{
	FsmAp *rtnVal = 0;
	switch ( type ) {
		case ConcatType: {
			/* Evaluate the Term. */
			rtnVal = term->walk( pd, false );
			/* Evaluate the FactorWithRep. */
			FsmAp *rhs = factorWithAug->walk( pd );
			/* Perform concatenation. */
			rtnVal->concatOp( rhs );
			afterOpMinimize( rtnVal, lastInSeq );
			break;
		}
		case RightStartType: {
			/* Evaluate the Term. */
			rtnVal = term->walk( pd );

			/* Evaluate the FactorWithRep. */
			FsmAp *rhs = factorWithAug->walk( pd );

			/* Set up the priority descriptors. The left machine gets the
			 * lower priority where as the right get the higher start priority. */
			priorDescs[0].key = pd->nextPriorKey++;
			priorDescs[0].priority = 0;
			rtnVal->allTransPrior( pd->curPriorOrd++, &priorDescs[0] );

			/* The start transitions of the right machine gets the higher
			 * priority. Use the same unique key. */
			priorDescs[1].key = priorDescs[0].key;
			priorDescs[1].priority = 1;
			rhs->startFsmPrior( pd->curPriorOrd++, &priorDescs[1] );

			/* Perform concatenation. */
			rtnVal->concatOp( rhs );
			afterOpMinimize( rtnVal, lastInSeq );
			break;
		}
		case RightFinishType: {
			/* Evaluate the Term. */
			rtnVal = term->walk( pd );

			/* Evaluate the FactorWithRep. */
			FsmAp *rhs = factorWithAug->walk( pd );

			/* Set up the priority descriptors. The left machine gets the
			 * lower priority where as the finishing transitions to the right
			 * get the higher priority. */
			priorDescs[0].key = pd->nextPriorKey++;
			priorDescs[0].priority = 0;
			rtnVal->allTransPrior( pd->curPriorOrd++, &priorDescs[0] );

			/* The finishing transitions of the right machine get the higher
			 * priority. Use the same unique key. */
			priorDescs[1].key = priorDescs[0].key;
			priorDescs[1].priority = 1;
			rhs->finishFsmPrior( pd->curPriorOrd++, &priorDescs[1] );

			/* If the right machine's start state is final we need to guard
			 * against the left machine persisting by moving through the empty
			 * string. */
			if ( rhs->startState->isFinState() ) {
				rhs->startState->outPriorTable.setPrior( 
						pd->curPriorOrd++, &priorDescs[1] );
			}

			/* Perform concatenation. */
			rtnVal->concatOp( rhs );
			afterOpMinimize( rtnVal, lastInSeq );
			break;
		}
		case LeftType: {
			/* Evaluate the Term. */
			rtnVal = term->walk( pd );

			/* Evaluate the FactorWithRep. */
			FsmAp *rhs = factorWithAug->walk( pd );

			/* Set up the priority descriptors. The left machine gets the
			 * higher priority. */
			priorDescs[0].key = pd->nextPriorKey++;
			priorDescs[0].priority = 1;
			rtnVal->allTransPrior( pd->curPriorOrd++, &priorDescs[0] );

			/* The right machine gets the lower priority. We cannot use
			 * allTransPrior here in case the start state of the right machine
			 * is final. It would allow the right machine thread to run along
			 * with the left if just passing through the start state. Using
			 * startFsmPrior prevents this. */
			priorDescs[1].key = priorDescs[0].key;
			priorDescs[1].priority = 0;
			rhs->startFsmPrior( pd->curPriorOrd++, &priorDescs[1] );

			/* Perform concatenation. */
			rtnVal->concatOp( rhs );
			afterOpMinimize( rtnVal, lastInSeq );
			break;
		}
		case FactorWithAugType: {
			rtnVal = factorWithAug->walk( pd );
			break;
		}
	}
	return rtnVal;
}

void Term::makeNameTree( ParseData *pd )
{
	switch ( type ) {
	case ConcatType:
	case RightStartType:
	case RightFinishType:
	case LeftType:
		term->makeNameTree( pd );
		factorWithAug->makeNameTree( pd );
		break;
	case FactorWithAugType:
		factorWithAug->makeNameTree( pd );
		break;
	}
}

void Term::resolveNameRefs( ParseData *pd )
{
	switch ( type ) {
	case ConcatType:
	case RightStartType:
	case RightFinishType:
	case LeftType:
		term->resolveNameRefs( pd );
		factorWithAug->resolveNameRefs( pd );
		break;
	case FactorWithAugType:
		factorWithAug->resolveNameRefs( pd );
		break;
	}
}

/* Clean up after a factor with augmentation node. */
FactorWithAug::~FactorWithAug()
{
	delete factorWithRep;

	/* Walk the vector of parser actions, deleting function names. */

	/* Clean up priority descriptors. */
	if ( priorDescs != 0 )
		delete[] priorDescs;
}

void FactorWithAug::assignActions( ParseData *pd, FsmAp *graph, int *actionOrd )
{
	/* Assign actions. */
	for ( int i = 0; i < actions.length(); i++ )  {
		switch ( actions[i].type ) {
		/* Transition actions. */
		case at_start:
			graph->startFsmAction( actionOrd[i], actions[i].action );
			afterOpMinimize( graph );
			break;
		case at_all:
			graph->allTransAction( actionOrd[i], actions[i].action );
			break;
		case at_finish:
			graph->finishFsmAction( actionOrd[i], actions[i].action );
			break;
		case at_leave:
			graph->leaveFsmAction( actionOrd[i], actions[i].action );
			break;

		/* Global error actions. */
		case at_start_gbl_error:
			graph->startErrorAction( actionOrd[i], actions[i].action, 0 );
			afterOpMinimize( graph );
			break;
		case at_all_gbl_error:
			graph->allErrorAction( actionOrd[i], actions[i].action, 0 );
			break;
		case at_final_gbl_error:
			graph->finalErrorAction( actionOrd[i], actions[i].action, 0 );
			break;
		case at_not_start_gbl_error:
			graph->notStartErrorAction( actionOrd[i], actions[i].action, 0 );
			break;
		case at_not_final_gbl_error:
			graph->notFinalErrorAction( actionOrd[i], actions[i].action, 0 );
			break;
		case at_middle_gbl_error:
			graph->middleErrorAction( actionOrd[i], actions[i].action, 0 );
			break;

		/* Local error actions. */
		case at_start_local_error:
			graph->startErrorAction( actionOrd[i], actions[i].action,
					actions[i].localErrKey );
			afterOpMinimize( graph );
			break;
		case at_all_local_error:
			graph->allErrorAction( actionOrd[i], actions[i].action,
					actions[i].localErrKey );
			break;
		case at_final_local_error:
			graph->finalErrorAction( actionOrd[i], actions[i].action,
					actions[i].localErrKey );
			break;
		case at_not_start_local_error:
			graph->notStartErrorAction( actionOrd[i], actions[i].action,
					actions[i].localErrKey );
			break;
		case at_not_final_local_error:
			graph->notFinalErrorAction( actionOrd[i], actions[i].action,
					actions[i].localErrKey );
			break;
		case at_middle_local_error:
			graph->middleErrorAction( actionOrd[i], actions[i].action,
					actions[i].localErrKey );
			break;

		/* EOF actions. */
		case at_start_eof:
			graph->startEOFAction( actionOrd[i], actions[i].action );
			afterOpMinimize( graph );
			break;
		case at_all_eof:
			graph->allEOFAction( actionOrd[i], actions[i].action );
			break;
		case at_final_eof:
			graph->finalEOFAction( actionOrd[i], actions[i].action );
			break;
		case at_not_start_eof:
			graph->notStartEOFAction( actionOrd[i], actions[i].action );
			break;
		case at_not_final_eof:
			graph->notFinalEOFAction( actionOrd[i], actions[i].action );
			break;
		case at_middle_eof:
			graph->middleEOFAction( actionOrd[i], actions[i].action );
			break;

		/* To State Actions. */
		case at_start_to_state:
			graph->startToStateAction( actionOrd[i], actions[i].action );
			afterOpMinimize( graph );
			break;
		case at_all_to_state:
			graph->allToStateAction( actionOrd[i], actions[i].action );
			break;
		case at_final_to_state:
			graph->finalToStateAction( actionOrd[i], actions[i].action );
			break;
		case at_not_start_to_state:
			graph->notStartToStateAction( actionOrd[i], actions[i].action );
			break;
		case at_not_final_to_state:
			graph->notFinalToStateAction( actionOrd[i], actions[i].action );
			break;
		case at_middle_to_state:
			graph->middleToStateAction( actionOrd[i], actions[i].action );
			break;

		/* From State Actions. */
		case at_start_from_state:
			graph->startFromStateAction( actionOrd[i], actions[i].action );
			afterOpMinimize( graph );
			break;
		case at_all_from_state:
			graph->allFromStateAction( actionOrd[i], actions[i].action );
			break;
		case at_final_from_state:
			graph->finalFromStateAction( actionOrd[i], actions[i].action );
			break;
		case at_not_start_from_state:
			graph->notStartFromStateAction( actionOrd[i], actions[i].action );
			break;
		case at_not_final_from_state:
			graph->notFinalFromStateAction( actionOrd[i], actions[i].action );
			break;
		case at_middle_from_state:
			graph->middleFromStateAction( actionOrd[i], actions[i].action );
			break;

		/* Remaining cases, prevented by the parser. */
		default: 
			assert( false );
			break;
		}
	}
}

void FactorWithAug::assignPriorities( FsmAp *graph, int *priorOrd )
{
	/* Assign priorities. */
	for ( int i = 0; i < priorityAugs.length(); i++ ) {
		switch ( priorityAugs[i].type ) {
		case at_start:
			graph->startFsmPrior( priorOrd[i], &priorDescs[i]);
			/* Start fsm priorities are a special case that may require
			 * minimization afterwards. */
			afterOpMinimize( graph );
			break;
		case at_all:
			graph->allTransPrior( priorOrd[i], &priorDescs[i] );
			break;
		case at_finish:
			graph->finishFsmPrior( priorOrd[i], &priorDescs[i] );
			break;
		case at_leave:
			graph->leaveFsmPrior( priorOrd[i], &priorDescs[i] );
			break;

		default:
			/* Parser Prevents this case. */
			break;
		}
	}
}

void FactorWithAug::assignConditions( FsmAp *graph )
{
	for ( int i = 0; i < conditions.length(); i++ )  {
		switch ( conditions[i].type ) {
		/* Transition actions. */
		case at_start:
			graph->startFsmCondition( conditions[i].action, conditions[i].sense );
			afterOpMinimize( graph );
			break;
		case at_all:
			graph->allTransCondition( conditions[i].action, conditions[i].sense );
			break;
		case at_leave:
			graph->leaveFsmCondition( conditions[i].action, conditions[i].sense );
			break;
		default:
			break;
		}
	}
}


/* Evaluate a factor with augmentation node. */
FsmAp *FactorWithAug::walk( ParseData *pd )
{
	/* Enter into the scopes created for the labels. */
	NameFrame nameFrame = pd->enterNameScope( false, labels.length() );

	/* Make the array of function orderings. */
	int *actionOrd = 0;
	if ( actions.length() > 0 )
		actionOrd = new int[actions.length()];
	
	/* First walk the list of actions, assigning order to all starting
	 * actions. */
	for ( int i = 0; i < actions.length(); i++ ) {
		if ( actions[i].type == at_start || 
				actions[i].type == at_start_gbl_error ||
				actions[i].type == at_start_local_error ||
				actions[i].type == at_start_to_state ||
				actions[i].type == at_start_from_state ||
				actions[i].type == at_start_eof )
			actionOrd[i] = pd->curActionOrd++;
	}

	/* Evaluate the factor with repetition. */
	FsmAp *rtnVal = factorWithRep->walk( pd );

	/* Compute the remaining action orderings. */
	for ( int i = 0; i < actions.length(); i++ ) {
		if ( actions[i].type != at_start && 
				actions[i].type != at_start_gbl_error &&
				actions[i].type != at_start_local_error &&
				actions[i].type != at_start_to_state &&
				actions[i].type != at_start_from_state &&
				actions[i].type != at_start_eof )
			actionOrd[i] = pd->curActionOrd++;
	}

	/* Embed conditions. */
	assignConditions( rtnVal );

	/* Embed actions. */
	assignActions( pd, rtnVal , actionOrd );

	/* Make the array of priority orderings. Orderings are local to this walk
	 * of the factor with augmentation. */
	int *priorOrd = 0;
	if ( priorityAugs.length() > 0 )
		priorOrd = new int[priorityAugs.length()];
	
	/* Walk all priorities, assigning the priority ordering. */
	for ( int i = 0; i < priorityAugs.length(); i++ )
		priorOrd[i] = pd->curPriorOrd++;

	/* If the priority descriptors have not been made, make them now.  Make
	 * priority descriptors for each priority asignment that will be passed to
	 * the fsm. Used to keep track of the key, value and used bit. */
	if ( priorDescs == 0 && priorityAugs.length() > 0 ) {
		priorDescs = new PriorDesc[priorityAugs.length()];
		for ( int i = 0; i < priorityAugs.length(); i++ ) {
			/* Init the prior descriptor for the priority setting. */
			priorDescs[i].key = priorityAugs[i].priorKey;
			priorDescs[i].priority = priorityAugs[i].priorValue;
		}
	}

	/* Assign priorities into the machine. */
	assignPriorities( rtnVal, priorOrd );

	/* Assign epsilon transitions. */
	for ( int e = 0; e < epsilonLinks.length(); e++ ) {
		/* Get the name, which may not exist. If it doesn't then silently
		 * ignore it because an error has already been reported. */
		NameInst *epTarg = pd->epsilonResolvedLinks[pd->nextEpsilonResolvedLink++];
		if ( epTarg != 0 ) {
			/* Make the epsilon transitions. */
			rtnVal->epsilonTrans( epTarg->id );

			/* Note that we have made a link to the name. */
			pd->localNameScope->referencedNames.append( epTarg );
		}
	}

	/* Set entry points for labels. */
	if ( labels.length() > 0 ) {
		/* Pop the names. */
		pd->resetNameScope( nameFrame );

		/* Make labels that are referenced into entry points. */
		for ( int i = 0; i < labels.length(); i++ ) {
			pd->enterNameScope( false, 1 );

			/* Will always be found. */
			NameInst *name = pd->curNameInst;

			/* If the name is referenced then set the entry point. */
			if ( name->numRefs > 0 )
				rtnVal->setEntry( name->id, rtnVal->startState );
		}

		pd->popNameScope( nameFrame );
	}

	if ( priorOrd != 0 )
		delete[] priorOrd;
	if ( actionOrd != 0 )
		delete[] actionOrd;	
	return rtnVal;
}

void FactorWithAug::makeNameTree( ParseData *pd )
{
	/* Add the labels to the tree of instantiated names. Each label
	 * makes a new scope. */
	NameInst *prevNameInst = pd->curNameInst;
	for ( int i = 0; i < labels.length(); i++ )
		pd->curNameInst = pd->addNameInst( labels[i].loc, labels[i].data, true );

	/* Recurse, then pop the names. */
	factorWithRep->makeNameTree( pd );
	pd->curNameInst = prevNameInst;
}


void FactorWithAug::resolveNameRefs( ParseData *pd )
{
	/* Enter into the name scope created by any labels. */
	NameFrame nameFrame = pd->enterNameScope( false, labels.length() );

	/* Note action references. */
	for ( int i = 0; i < actions.length(); i++ ) 
		actions[i].action->actionRefs.append( pd->localNameScope );

	/* Recurse first. IMPORTANT: we must do the exact same traversal as when
	 * the tree is constructed. */
	factorWithRep->resolveNameRefs( pd );

	/* Resolve epsilon transitions. */
	for ( int ep = 0; ep < epsilonLinks.length(); ep++ ) {
		/* Get the link. */
		EpsilonLink &link = epsilonLinks[ep];
		NameInst *resolvedName = 0;

		if ( link.target.length() == 1 && strcmp( link.target.data[0], "final" ) == 0 ) {
			/* Epsilon drawn to an implicit final state. An implicit final is
			 * only available in join operations. */
			resolvedName = pd->localNameScope->final;
		}
		else {
			/* Do an search for the name. */
			NameSet resolved;
			pd->resolveFrom( resolved, pd->localNameScope, link.target, 0 );
			if ( resolved.length() > 0 ) {
				/* Take the first one. */
				resolvedName = resolved[0];
				if ( resolved.length() > 1 ) {
					/* Complain about the multiple references. */
					error(link.loc) << "state reference " << link.target << 
							" resolves to multiple entry points" << endl;
					errorStateLabels( resolved );
				}
			}
		}

		/* This is tricky, we stuff resolved epsilon transitions into one long
		 * vector in the parse data structure. Since the name resolution and
		 * graph generation both do identical walks of the parse tree we
		 * should always find the link resolutions in the right place.  */
		pd->epsilonResolvedLinks.append( resolvedName );

		if ( resolvedName != 0 ) {
			/* Found the name, bump of the reference count on it. */
			resolvedName->numRefs += 1;
		}
		else {
			/* Complain, no recovery action, the epsilon op will ignore any
			 * epsilon transitions whose names did not resolve. */
			error(link.loc) << "could not resolve label " << link.target << endl;
		}
	}

	if ( labels.length() > 0 )
		pd->popNameScope( nameFrame );
}


/* Clean up after a factor with repetition node. */
FactorWithRep::~FactorWithRep()
{
	switch ( type ) {
		case StarType: case StarStarType: case OptionalType: case PlusType:
		case ExactType: case MaxType: case MinType: case RangeType:
			delete factorWithRep;
			break;
		case FactorWithNegType:
			delete factorWithNeg;
			break;
	}
}

/* Evaluate a factor with repetition node. */
FsmAp *FactorWithRep::walk( ParseData *pd )
{
	FsmAp *retFsm = 0;

	switch ( type ) {
	case StarType: {
		/* Evaluate the FactorWithRep. */
		retFsm = factorWithRep->walk( pd );
		if ( retFsm->startState->isFinState() ) {
			warning(loc) << "applying kleene star to a machine that "
					"accepts zero length word" << endl;
			retFsm->unsetFinState( retFsm->startState );
		}

		/* Shift over the start action orders then do the kleene star. */
		pd->curActionOrd += retFsm->shiftStartActionOrder( pd->curActionOrd );
		retFsm->starOp( );
		afterOpMinimize( retFsm );
		break;
	}
	case StarStarType: {
		/* Evaluate the FactorWithRep. */
		retFsm = factorWithRep->walk( pd );
		if ( retFsm->startState->isFinState() ) {
			warning(loc) << "applying kleene star to a machine that "
					"accepts zero length word" << endl;
		}

		/* Set up the prior descs. All gets priority one, whereas leaving gets
		 * priority zero. Make a unique key so that these priorities don't
		 * interfere with any priorities set by the user. */
		priorDescs[0].key = pd->nextPriorKey++;
		priorDescs[0].priority = 1;
		retFsm->allTransPrior( pd->curPriorOrd++, &priorDescs[0] );

		/* Leaveing gets priority 0. Use same unique key. */
		priorDescs[1].key = priorDescs[0].key;
		priorDescs[1].priority = 0;
		retFsm->leaveFsmPrior( pd->curPriorOrd++, &priorDescs[1] );

		/* Shift over the start action orders then do the kleene star. */
		pd->curActionOrd += retFsm->shiftStartActionOrder( pd->curActionOrd );
		retFsm->starOp( );
		afterOpMinimize( retFsm );
		break;
	}
	case OptionalType: {
		/* Make the null fsm. */
		FsmAp *nu = new FsmAp();
		nu->lambdaFsm( );

		/* Evaluate the FactorWithRep. */
		retFsm = factorWithRep->walk( pd );

		/* Perform the question operator. */
		retFsm->unionOp( nu );
		afterOpMinimize( retFsm );
		break;
	}
	case PlusType: {
		/* Evaluate the FactorWithRep. */
		retFsm = factorWithRep->walk( pd );
		if ( retFsm->startState->isFinState() ) {
			warning(loc) << "applying plus operator to a machine that "
					"accepts zero length word" << endl;
		}

		/* Need a duplicated for the star end. */
		FsmAp *dup = new FsmAp( *retFsm );

		/* The start func orders need to be shifted before doing the star. */
		pd->curActionOrd += dup->shiftStartActionOrder( pd->curActionOrd );

		/* Star the duplicate. */
		dup->starOp( );
		afterOpMinimize( dup );

		retFsm->concatOp( dup );
		afterOpMinimize( retFsm );
		break;
	}
	case ExactType: {
		/* Get an int from the repetition amount. */
		if ( lowerRep == 0 ) {
			/* No copies. Don't need to evaluate the factorWithRep. 
			 * This Defeats the purpose so give a warning. */
			warning(loc) << "exactly zero repetitions results "
					"in the null machine" << endl;

			retFsm = new FsmAp();
			retFsm->lambdaFsm();
		}
		else {
			/* Evaluate the first FactorWithRep. */
			retFsm = factorWithRep->walk( pd );
			if ( retFsm->startState->isFinState() ) {
				warning(loc) << "applying repetition to a machine that "
						"accepts zero length word" << endl;
			}

			/* The start func orders need to be shifted before doing the
			 * repetition. */
			pd->curActionOrd += retFsm->shiftStartActionOrder( pd->curActionOrd );

			/* Do the repetition on the machine. Already guarded against n == 0 */
			retFsm->repeatOp( lowerRep );
			afterOpMinimize( retFsm );
		}
		break;
	}
	case MaxType: {
		/* Get an int from the repetition amount. */
		if ( upperRep == 0 ) {
			/* No copies. Don't need to evaluate the factorWithRep. 
			 * This Defeats the purpose so give a warning. */
			warning(loc) << "max zero repetitions results "
					"in the null machine" << endl;

			retFsm = new FsmAp();
			retFsm->lambdaFsm();
		}
		else {
			/* Evaluate the first FactorWithRep. */
			retFsm = factorWithRep->walk( pd );
			if ( retFsm->startState->isFinState() ) {
				warning(loc) << "applying max repetition to a machine that "
						"accepts zero length word" << endl;
			}

			/* The start func orders need to be shifted before doing the 
			 * repetition. */
			pd->curActionOrd += retFsm->shiftStartActionOrder( pd->curActionOrd );

			/* Do the repetition on the machine. Already guarded against n == 0 */
			retFsm->optionalRepeatOp( upperRep );
			afterOpMinimize( retFsm );
		}
		break;
	}
	case MinType: {
		/* Evaluate the repeated machine. */
		retFsm = factorWithRep->walk( pd );
		if ( retFsm->startState->isFinState() ) {
			warning(loc) << "applying min repetition to a machine that "
					"accepts zero length word" << endl;
		}

		/* The start func orders need to be shifted before doing the repetition
		 * and the kleene star. */
		pd->curActionOrd += retFsm->shiftStartActionOrder( pd->curActionOrd );
	
		if ( lowerRep == 0 ) {
			/* Acts just like a star op on the machine to return. */
			retFsm->starOp( );
			afterOpMinimize( retFsm );
		}
		else {
			/* Take a duplicate for the plus. */
			FsmAp *dup = new FsmAp( *retFsm );

			/* Do repetition on the first half. */
			retFsm->repeatOp( lowerRep );
			afterOpMinimize( retFsm );

			/* Star the duplicate. */
			dup->starOp( );
			afterOpMinimize( dup );

			/* Tak on the kleene star. */
			retFsm->concatOp( dup );
			afterOpMinimize( retFsm );
		}
		break;
	}
	case RangeType: {
		/* Check for bogus range. */
		if ( upperRep - lowerRep < 0 ) {
			error(loc) << "invalid range repetition" << endl;

			/* Return null machine as recovery. */
			retFsm = new FsmAp();
			retFsm->lambdaFsm();
		}
		else if ( lowerRep == 0 && upperRep == 0 ) {
			/* No copies. Don't need to evaluate the factorWithRep.  This
			 * defeats the purpose so give a warning. */
			warning(loc) << "zero to zero repetitions results "
					"in the null machine" << endl;

			retFsm = new FsmAp();
			retFsm->lambdaFsm();
		}
		else {
			/* Now need to evaluate the repeated machine. */
			retFsm = factorWithRep->walk( pd );
			if ( retFsm->startState->isFinState() ) {
				warning(loc) << "applying range repetition to a machine that "
						"accepts zero length word" << endl;
			}

			/* The start func orders need to be shifted before doing both kinds
			 * of repetition. */
			pd->curActionOrd += retFsm->shiftStartActionOrder( pd->curActionOrd );

			if ( lowerRep == 0 ) {
				/* Just doing max repetition. Already guarded against n == 0. */
				retFsm->optionalRepeatOp( upperRep );
				afterOpMinimize( retFsm );
			}
			else if ( lowerRep == upperRep ) {
				/* Just doing exact repetition. Already guarded against n == 0. */
				retFsm->repeatOp( lowerRep );
				afterOpMinimize( retFsm );
			}
			else {
				/* This is the case that 0 < lowerRep < upperRep. Take a
				 * duplicate for the optional repeat. */
				FsmAp *dup = new FsmAp( *retFsm );

				/* Do repetition on the first half. */
				retFsm->repeatOp( lowerRep );
				afterOpMinimize( retFsm );

				/* Do optional repetition on the second half. */
				dup->optionalRepeatOp( upperRep - lowerRep );
				afterOpMinimize( dup );

				/* Tak on the duplicate machine. */
				retFsm->concatOp( dup );
				afterOpMinimize( retFsm );
			}
		}
		break;
	}
	case FactorWithNegType: {
		/* Evaluate the Factor. Pass it up. */
		retFsm = factorWithNeg->walk( pd );
		break;
	}}
	return retFsm;
}

void FactorWithRep::makeNameTree( ParseData *pd )
{
	switch ( type ) {
	case StarType:
	case StarStarType:
	case OptionalType:
	case PlusType:
	case ExactType:
	case MaxType:
	case MinType:
	case RangeType:
		factorWithRep->makeNameTree( pd );
		break;
	case FactorWithNegType:
		factorWithNeg->makeNameTree( pd );
		break;
	}
}

void FactorWithRep::resolveNameRefs( ParseData *pd )
{
	switch ( type ) {
	case StarType:
	case StarStarType:
	case OptionalType:
	case PlusType:
	case ExactType:
	case MaxType:
	case MinType:
	case RangeType:
		factorWithRep->resolveNameRefs( pd );
		break;
	case FactorWithNegType:
		factorWithNeg->resolveNameRefs( pd );
		break;
	}
}

/* Clean up after a factor with negation node. */
FactorWithNeg::~FactorWithNeg()
{
	switch ( type ) {
		case NegateType:
		case CharNegateType:
			delete factorWithNeg;
			break;
		case FactorType:
			delete factor;
			break;
	}
}

/* Evaluate a factor with negation node. */
FsmAp *FactorWithNeg::walk( ParseData *pd )
{
	FsmAp *retFsm = 0;

	switch ( type ) {
	case NegateType: {
		/* Evaluate the factorWithNeg. */
		FsmAp *toNegate = factorWithNeg->walk( pd );

		/* Negation is subtract from dot-star. */
		retFsm = dotStarFsm( pd );
		retFsm->subtractOp( toNegate );
		afterOpMinimize( retFsm );
		break;
	}
	case CharNegateType: {
		/* Evaluate the factorWithNeg. */
		FsmAp *toNegate = factorWithNeg->walk( pd );

		/* CharNegation is subtract from dot. */
		retFsm = dotFsm( pd );
		retFsm->subtractOp( toNegate );
		afterOpMinimize( retFsm );
		break;
	}
	case FactorType: {
		/* Evaluate the Factor. Pass it up. */
		retFsm = factor->walk( pd );
		break;
	}}
	return retFsm;
}

void FactorWithNeg::makeNameTree( ParseData *pd )
{
	switch ( type ) {
	case NegateType:
	case CharNegateType:
		factorWithNeg->makeNameTree( pd );
		break;
	case FactorType:
		factor->makeNameTree( pd );
		break;
	}
}

void FactorWithNeg::resolveNameRefs( ParseData *pd )
{
	switch ( type ) {
	case NegateType:
	case CharNegateType:
		factorWithNeg->resolveNameRefs( pd );
		break;
	case FactorType:
		factor->resolveNameRefs( pd );
		break;
	}
}

/* Clean up after a factor node. */
Factor::~Factor()
{
	switch ( type ) {
		case LiteralType:
			delete literal;
			break;
		case RangeType:
			delete range;
			break;
		case OrExprType:
			delete reItem;
			break;
		case RegExprType:
			delete regExpr;
			break;
		case ReferenceType:
			break;
		case ParenType:
			delete join;
			break;
		case LongestMatchType:
			delete longestMatch;
			break;
	}
}

/* Evaluate a factor node. */
FsmAp *Factor::walk( ParseData *pd )
{
	FsmAp *rtnVal = 0;
	switch ( type ) {
	case LiteralType:
		rtnVal = literal->walk( pd );
		break;
	case RangeType:
		rtnVal = range->walk( pd );
		break;
	case OrExprType:
		rtnVal = reItem->walk( pd, 0 );
		break;
	case RegExprType:
		rtnVal = regExpr->walk( pd, 0 );
		break;
	case ReferenceType:
		rtnVal = varDef->walk( pd );
		break;
	case ParenType:
		rtnVal = join->walk( pd );
		break;
	case LongestMatchType:
		rtnVal = longestMatch->walk( pd );
		break;
	}

	return rtnVal;
}

void Factor::makeNameTree( ParseData *pd )
{
	switch ( type ) {
	case LiteralType:
	case RangeType:
	case OrExprType:
	case RegExprType:
		break;
	case ReferenceType:
		varDef->makeNameTree( loc, pd );
		break;
	case ParenType:
		join->makeNameTree( pd );
		break;
	case LongestMatchType:
		longestMatch->makeNameTree( pd );
		break;
	}
}

void Factor::resolveNameRefs( ParseData *pd )
{
	switch ( type ) {
	case LiteralType:
	case RangeType:
	case OrExprType:
	case RegExprType:
		break;
	case ReferenceType:
		varDef->resolveNameRefs( pd );
		break;
	case ParenType:
		join->resolveNameRefs( pd );
		break;
	case LongestMatchType:
		longestMatch->resolveNameRefs( pd );
		break;
	}
}

/* Clean up a range object. Must delete the two literals. */
Range::~Range()
{
	delete lowerLit;
	delete upperLit;
}

/* Evaluate a range. Gets the lower an upper key and makes an fsm range. */
FsmAp *Range::walk( ParseData *pd )
{
	/* Construct and verify the suitability of the lower end of the range. */
	FsmAp *lowerFsm = lowerLit->walk( pd );
	if ( !lowerFsm->checkSingleCharMachine() ) {
		error(lowerLit->token.loc) << 
			"bad range lower end, must be a single character" << endl;
	}

	/* Construct and verify the upper end. */
	FsmAp *upperFsm = upperLit->walk( pd );
	if ( !upperFsm->checkSingleCharMachine() ) {
		error(upperLit->token.loc) << 
			"bad range upper end, must be a single character" << endl;
	}

	/* Grab the keys from the machines, then delete them. */
	Key lowKey = lowerFsm->startState->outList.head->lowKey;
	Key highKey = upperFsm->startState->outList.head->lowKey;
	delete lowerFsm;
	delete upperFsm;

	/* Validate the range. */
	if ( lowKey > highKey ) {
		/* Recover by setting upper to lower; */
		error(lowerLit->token.loc) << "lower end of range is greater then upper end" << endl;
		highKey = lowKey;
	}

	/* Return the range now that it is validated. */
	FsmAp *retFsm = new FsmAp();
	retFsm->rangeFsm( lowKey, highKey );
	return retFsm;
}

/* Evaluate a literal object. */
FsmAp *Literal::walk( ParseData *pd )
{
	/* FsmAp to return, is the alphabet signed. */
	FsmAp *rtnVal = 0;

	switch ( type ) {
	case Number: {
		/* Make the fsm key in int format. */
		Key fsmKey = makeFsmKeyNum( token.data, token.loc, pd );
		/* Make the new machine. */
		rtnVal = new FsmAp();
		rtnVal->concatFsm( fsmKey );
		break;
	}
	case LitString: {
		/* Make the array of keys in int format. */
		long length;
		bool caseInsensitive;
		char *data = prepareLitString( token.loc, token.data, token.length, 
				length, caseInsensitive );
		Key *arr = new Key[length];
		makeFsmKeyArray( arr, data, length, pd );

		/* Make the new machine. */
		rtnVal = new FsmAp();
		if ( caseInsensitive )
			rtnVal->concatFsmCI( arr, length );
		else
			rtnVal->concatFsm( arr, length );
		delete[] data;
		delete[] arr;
		break;
	}}
	return rtnVal;
}

/* Clean up after a regular expression object. */
RegExpr::~RegExpr()
{
	switch ( type ) {
		case RecurseItem:
			delete regExpr;
			delete item;
			break;
		case Empty:
			break;
	}
}

/* Evaluate a regular expression object. */
FsmAp *RegExpr::walk( ParseData *pd, RegExpr *rootRegex )
{
	/* This is the root regex, pass down a pointer to this. */
	if ( rootRegex == 0 )
		rootRegex = this;

	FsmAp *rtnVal = 0;
	switch ( type ) {
		case RecurseItem: {
			/* Walk both items. */
			rtnVal = regExpr->walk( pd, rootRegex );
			FsmAp *fsm2 = item->walk( pd, rootRegex );
			rtnVal->concatOp( fsm2 );
			break;
		}
		case Empty: {
			rtnVal = new FsmAp();
			rtnVal->lambdaFsm();
			break;
		}
	}
	return rtnVal;
}

/* Clean up after an item in a regular expression. */
ReItem::~ReItem()
{
	switch ( type ) {
		case Data:
		case Dot:
			break;
		case OrBlock:
		case NegOrBlock:
			delete orBlock;
			break;
	}
}

/* Evaluate a regular expression object. */
FsmAp *ReItem::walk( ParseData *pd, RegExpr *rootRegex )
{
	/* The fsm to return, is the alphabet signed? */
	FsmAp *rtnVal = 0;

	switch ( type ) {
		case Data: {
			/* Move the data into an integer array and make a concat fsm. */
			Key *arr = new Key[token.length];
			makeFsmKeyArray( arr, token.data, token.length, pd );

			/* Make the concat fsm. */
			rtnVal = new FsmAp();
			if ( rootRegex != 0 && rootRegex->caseInsensitive )
				rtnVal->concatFsmCI( arr, token.length );
			else
				rtnVal->concatFsm( arr, token.length );
			delete[] arr;
			break;
		}
		case Dot: {
			/* Make the dot fsm. */
			rtnVal = dotFsm( pd );
			break;
		}
		case OrBlock: {
			/* Get the or block and minmize it. */
			rtnVal = orBlock->walk( pd, rootRegex );
			if ( rtnVal == 0 ) {
				rtnVal = new FsmAp();
				rtnVal->lambdaFsm();
			}
			rtnVal->minimizePartition2();
			break;
		}
		case NegOrBlock: {
			/* Get the or block and minimize it. */
			FsmAp *fsm = orBlock->walk( pd, rootRegex );
			fsm->minimizePartition2();

			/* Make a dot fsm and subtract from it. */
			rtnVal = dotFsm( pd );
			rtnVal->subtractOp( fsm );
			rtnVal->minimizePartition2();
			break;
		}
	}

	/* If the item is followed by a star, then apply the star op. */
	if ( star ) {
		if ( rtnVal->startState->isFinState() ) {
			warning(loc) << "applying kleene star to a machine that "
					"accepts zero length word" << endl;
		}

		rtnVal->starOp();
		rtnVal->minimizePartition2();
	}
	return rtnVal;
}

/* Clean up after an or block of a regular expression. */
ReOrBlock::~ReOrBlock()
{
	switch ( type ) {
		case RecurseItem:
			delete orBlock;
			delete item;
			break;
		case Empty:
			break;
	}
}


/* Evaluate an or block of a regular expression. */
FsmAp *ReOrBlock::walk( ParseData *pd, RegExpr *rootRegex )
{
	FsmAp *rtnVal = 0;
	switch ( type ) {
		case RecurseItem: {
			/* Evaluate the two fsm. */
			FsmAp *fsm1 = orBlock->walk( pd, rootRegex );
			FsmAp *fsm2 = item->walk( pd, rootRegex );
			if ( fsm1 == 0 )
				rtnVal = fsm2;
			else {
				fsm1->unionOp( fsm2 );
				rtnVal = fsm1;
			}
			break;
		}
		case Empty: {
			rtnVal = 0;
			break;
		}
	}
	return rtnVal;;
}

/* Evaluate an or block item of a regular expression. */
FsmAp *ReOrItem::walk( ParseData *pd, RegExpr *rootRegex )
{
	/* The return value, is the alphabet signed? */
	FsmAp *rtnVal = 0;
	switch ( type ) {
	case Data: {
		/* Make the or machine. */
		rtnVal = new FsmAp();

		/* Put the or data into an array of ints. Note that we find unique
		 * keys. Duplicates are silently ignored. The alternative would be to
		 * issue warning or an error but since we can't with [a0-9a] or 'a' |
		 * 'a' don't bother here. */
		KeySet keySet;
		makeFsmUniqueKeyArray( keySet, token.data, token.length, 
			rootRegex != 0 ? rootRegex->caseInsensitive : false, pd );

		/* Run the or operator. */
		rtnVal->orFsm( keySet.data, keySet.length() );
		break;
	}
	case Range: {
		/* Make the upper and lower keys. */
		Key lowKey = makeFsmKeyChar( lower, pd );
		Key highKey = makeFsmKeyChar( upper, pd );

		/* Validate the range. */
		if ( lowKey > highKey ) {
			/* Recover by setting upper to lower; */
			error(loc) << "lower end of range is greater then upper end" << endl;
			highKey = lowKey;
		}

		/* Make the range machine. */
		rtnVal = new FsmAp();
		rtnVal->rangeFsm( lowKey, highKey );

		if ( rootRegex != 0 && rootRegex->caseInsensitive ) {
			if ( lowKey <= 'Z' && 'A' <= highKey ) {
				Key otherLow = lowKey < 'A' ? Key('A') : lowKey;
				Key otherHigh = 'Z' < highKey ? Key('Z') : highKey;

				otherLow = 'a' + ( otherLow - 'A' );
				otherHigh = 'a' + ( otherHigh - 'A' );

				FsmAp *otherRange = new FsmAp();
				otherRange->rangeFsm( otherLow, otherHigh );
				rtnVal->unionOp( otherRange );
				rtnVal->minimizePartition2();
			}
			else if ( lowKey <= 'z' && 'a' <= highKey ) {
				Key otherLow = lowKey < 'a' ? Key('a') : lowKey;
				Key otherHigh = 'z' < highKey ? Key('z') : highKey;

				otherLow = 'A' + ( otherLow - 'a' );
				otherHigh = 'A' + ( otherHigh - 'a' );

				FsmAp *otherRange = new FsmAp();
				otherRange->rangeFsm( otherLow, otherHigh );
				rtnVal->unionOp( otherRange );
				rtnVal->minimizePartition2();
			}
		}

		break;
	}}
	return rtnVal;
}
