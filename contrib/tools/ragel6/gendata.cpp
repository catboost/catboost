/*
 *  Copyright 2005-2007 Adrian Thurston <thurston@complang.org>
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

#include "gendata.h"
#include "ragel.h"
#include <iostream>

/*
 * Code generators.
 */

#include "cstable.h"
#include "csftable.h"
#include "csflat.h"
#include "csfflat.h"
#include "csgoto.h"
#include "csfgoto.h"
#include "csipgoto.h"
#include "cssplit.h"

#include "cdtable.h"
#include "cdftable.h"
#include "cdflat.h"
#include "cdfflat.h"
#include "cdgoto.h"
#include "cdfgoto.h"
#include "cdipgoto.h"
#include "cdsplit.h"

#include "dotcodegen.h"

#include "javacodegen.h"

#include "gocodegen.h"
#include "gotable.h"
#include "goftable.h"
#include "goflat.h"
#include "gofflat.h"
#include "gogoto.h"
#include "gofgoto.h"
#include "goipgoto.h"

#include "mltable.h"
#include "mlftable.h"
#include "mlflat.h"
#include "mlfflat.h"
#include "mlgoto.h"
#include "mlfgoto.h"

#include "rubytable.h"
#include "rubyftable.h"
#include "rubyflat.h"
#include "rubyfflat.h"
#include "rbxgoto.h"

string itoa( int i )
{
	char buf[16];
	sprintf( buf, "%i", i );
	return buf;
}

using std::cout;
using std::cerr;
using std::endl;

/* Invoked by the parser when a ragel definition is opened. */
CodeGenData *dotMakeCodeGen( const char *sourceFileName, const char *fsmName, ostream &out )
{
	CodeGenData *codeGen = new GraphvizDotGen(out);

	codeGen->sourceFileName = sourceFileName;
	codeGen->fsmName = fsmName;

	/* For normal code generation we want a transition on every character so we never
	 * end up in an undefined state. For graphviz this just clutters the
	 * drawing so we turn it off. */
	codeGen->wantComplete = false;

	return codeGen;
}

/* Invoked by the parser when a ragel definition is opened. */
CodeGenData *cdMakeCodeGen( const char *sourceFileName, const char *fsmName, ostream &out )
{
	CodeGenData *codeGen = 0;
	switch ( hostLang->lang ) {
	case HostLang::C:
		switch ( codeStyle ) {
		case GenTables:
			codeGen = new CTabCodeGen(out);
			break;
		case GenFTables:
			codeGen = new CFTabCodeGen(out);
			break;
		case GenFlat:
			codeGen = new CFlatCodeGen(out);
			break;
		case GenFFlat:
			codeGen = new CFFlatCodeGen(out);
			break;
		case GenGoto:
			codeGen = new CGotoCodeGen(out);
			break;
		case GenFGoto:
			codeGen = new CFGotoCodeGen(out);
			break;
		case GenIpGoto:
			codeGen = new CIpGotoCodeGen(out);
			break;
		case GenSplit:
			codeGen = new CSplitCodeGen(out);
			break;
		}
		break;

	case HostLang::D:
		switch ( codeStyle ) {
		case GenTables:
			codeGen = new DTabCodeGen(out);
			break;
		case GenFTables:
			codeGen = new DFTabCodeGen(out);
			break;
		case GenFlat:
			codeGen = new DFlatCodeGen(out);
			break;
		case GenFFlat:
			codeGen = new DFFlatCodeGen(out);
			break;
		case GenGoto:
			codeGen = new DGotoCodeGen(out);
			break;
		case GenFGoto:
			codeGen = new DFGotoCodeGen(out);
			break;
		case GenIpGoto:
			codeGen = new DIpGotoCodeGen(out);
			break;
		case GenSplit:
			codeGen = new DSplitCodeGen(out);
			break;
		}
		break;

	case HostLang::D2:
		switch ( codeStyle ) {
		case GenTables:
			codeGen = new D2TabCodeGen(out);
			break;
		case GenFTables:
			codeGen = new D2FTabCodeGen(out);
			break;
		case GenFlat:
			codeGen = new D2FlatCodeGen(out);
			break;
		case GenFFlat:
			codeGen = new D2FFlatCodeGen(out);
			break;
		case GenGoto:
			codeGen = new D2GotoCodeGen(out);
			break;
		case GenFGoto:
			codeGen = new D2FGotoCodeGen(out);
			break;
		case GenIpGoto:
			codeGen = new D2IpGotoCodeGen(out);
			break;
		case GenSplit:
			codeGen = new D2SplitCodeGen(out);
			break;
		}
		break;

	default: break;
	}

	codeGen->sourceFileName = sourceFileName;
	codeGen->fsmName = fsmName;

	return codeGen;
}

/* Invoked by the parser when a ragel definition is opened. */
CodeGenData *javaMakeCodeGen( const char *sourceFileName, const char *fsmName, ostream &out )
{
	CodeGenData *codeGen = new JavaTabCodeGen(out);

	codeGen->sourceFileName = sourceFileName;
	codeGen->fsmName = fsmName;

	return codeGen;
}

/* Invoked by the parser when a ragel definition is opened. */
CodeGenData *goMakeCodeGen( const char *sourceFileName, const char *fsmName, ostream &out )
{
	CodeGenData *codeGen = 0;

	switch ( codeStyle ) {
	case GenTables:
		codeGen = new GoTabCodeGen(out);
		break;
	case GenFTables:
		codeGen = new GoFTabCodeGen(out);
		break;
	case GenFlat:
		codeGen = new GoFlatCodeGen(out);
		break;
	case GenFFlat:
		codeGen = new GoFFlatCodeGen(out);
		break;
	case GenGoto:
		codeGen = new GoGotoCodeGen(out);
		break;
	case GenFGoto:
		codeGen = new GoFGotoCodeGen(out);
		break;
	case GenIpGoto:
		codeGen = new GoIpGotoCodeGen(out);
		break;
	default:
		cerr << "Invalid output style, only -T0, -T1, -F0, -F1, -G0, -G1 and -G2 are supported for Go.\n";
		exit(1);
	}

	codeGen->sourceFileName = sourceFileName;
	codeGen->fsmName = fsmName;

	return codeGen;
}

/* Invoked by the parser when a ragel definition is opened. */
CodeGenData *rubyMakeCodeGen( const char *sourceFileName, const char *fsmName, ostream &out )
{
	CodeGenData *codeGen = 0;
	switch ( codeStyle ) {
		case GenTables: 
			codeGen = new RubyTabCodeGen(out);
			break;
		case GenFTables:
			codeGen = new RubyFTabCodeGen(out);
			break;
		case GenFlat:
			codeGen = new RubyFlatCodeGen(out);
			break;
		case GenFFlat:
			codeGen = new RubyFFlatCodeGen(out);
			break;
		case GenGoto:
			if ( rubyImpl == Rubinius ) {
				codeGen = new RbxGotoCodeGen(out);
			} else {
				cerr << "Goto style is still _very_ experimental " 
					"and only supported using Rubinius.\n"
					"You may want to enable the --rbx flag "
					" to give it a try.\n";
				exit(1);
			}
			break;
		default:
			cout << "Invalid code style\n";
			exit(1);
			break;
	}
	codeGen->sourceFileName = sourceFileName;
	codeGen->fsmName = fsmName;

	return codeGen;
}

/* Invoked by the parser when a ragel definition is opened. */
CodeGenData *csharpMakeCodeGen( const char *sourceFileName, const char *fsmName, ostream &out )
{
	CodeGenData *codeGen = 0;

	switch ( codeStyle ) {
	case GenTables:
		codeGen = new CSharpTabCodeGen(out);
		break;
	case GenFTables:
		codeGen = new CSharpFTabCodeGen(out);
		break;
	case GenFlat:
		codeGen = new CSharpFlatCodeGen(out);
		break;
	case GenFFlat:
		codeGen = new CSharpFFlatCodeGen(out);
		break;
	case GenGoto:
		codeGen = new CSharpGotoCodeGen(out);
		break;
	case GenFGoto:
		codeGen = new CSharpFGotoCodeGen(out);
		break;
	case GenIpGoto:
		codeGen = new CSharpIpGotoCodeGen(out);
		break;
	case GenSplit:
		codeGen = new CSharpSplitCodeGen(out);
		break;
	}

	codeGen->sourceFileName = sourceFileName;
	codeGen->fsmName = fsmName;

	return codeGen;
}

/* Invoked by the parser when a ragel definition is opened. */
CodeGenData *ocamlMakeCodeGen( const char *sourceFileName, const char *fsmName, ostream &out )
{
	CodeGenData *codeGen = 0;

	switch ( codeStyle ) {
	case GenTables:
		codeGen = new OCamlTabCodeGen(out);
		break;
	case GenFTables:
		codeGen = new OCamlFTabCodeGen(out);
		break;
	case GenFlat:
		codeGen = new OCamlFlatCodeGen(out);
		break;
	case GenFFlat:
		codeGen = new OCamlFFlatCodeGen(out);
		break;
	case GenGoto:
		codeGen = new OCamlGotoCodeGen(out);
		break;
	case GenFGoto:
		codeGen = new OCamlFGotoCodeGen(out);
		break;
	default:
		cerr << "I only support the -T0 -T1 -F0 -F1 -G0 and -G1 output styles for OCaml.\n";
		exit(1);
	}

	codeGen->sourceFileName = sourceFileName;
	codeGen->fsmName = fsmName;

	return codeGen;
}


CodeGenData *makeCodeGen( const char *sourceFileName, const char *fsmName, ostream &out )
{
	CodeGenData *cgd = 0;
	if ( generateDot )
		cgd = dotMakeCodeGen( sourceFileName, fsmName, out );
	else if ( hostLang == &hostLangC )
		cgd = cdMakeCodeGen( sourceFileName, fsmName, out );
	else if ( hostLang == &hostLangD )
		cgd = cdMakeCodeGen( sourceFileName, fsmName, out );
	else if ( hostLang == &hostLangD2 )
		cgd = cdMakeCodeGen( sourceFileName, fsmName, out );
	else if ( hostLang == &hostLangGo )
		cgd = goMakeCodeGen( sourceFileName, fsmName, out );
	else if ( hostLang == &hostLangJava )
		cgd = javaMakeCodeGen( sourceFileName, fsmName, out );
	else if ( hostLang == &hostLangRuby )
		cgd = rubyMakeCodeGen( sourceFileName, fsmName, out );
	else if ( hostLang == &hostLangCSharp )
		cgd = csharpMakeCodeGen( sourceFileName, fsmName, out );
	else if ( hostLang == &hostLangOCaml )
		cgd = ocamlMakeCodeGen( sourceFileName, fsmName, out );
	return cgd;
}

void lineDirective( ostream &out, const char *fileName, int line )
{
	if ( !generateDot ) {
		if ( hostLang == &hostLangC )
			cdLineDirective( out, fileName, line );
		else if ( hostLang == &hostLangD )
			cdLineDirective( out, fileName, line );
		else if ( hostLang == &hostLangD2 )
			cdLineDirective( out, fileName, line );
		else if ( hostLang == &hostLangGo )
			goLineDirective( out, fileName, line );
		else if ( hostLang == &hostLangJava )
			javaLineDirective( out, fileName, line );
		else if ( hostLang == &hostLangRuby )
			rubyLineDirective( out, fileName, line );
		else if ( hostLang == &hostLangCSharp )
			csharpLineDirective( out, fileName, line );
		else if ( hostLang == &hostLangOCaml )
			ocamlLineDirective( out, fileName, line );
	}
}

void genLineDirective( ostream &out )
{
	std::streambuf *sbuf = out.rdbuf();
	output_filter *filter = static_cast<output_filter*>(sbuf);
	lineDirective( out, filter->fileName, filter->line + 1 );
}


/* Total error count. */
/* int gblErrorCount = 0; */

CodeGenData::CodeGenData( ostream &out )
:
	sourceFileName(0),
	fsmName(0), 
	out(out),
	redFsm(0), 
	allActions(0),
	allActionTables(0),
	allConditions(0),
	allCondSpaces(0),
	allStates(0),
	nameIndex(0),
	startState(-1),
	errState(-1),
	getKeyExpr(0),
	accessExpr(0),
	prePushExpr(0),
	postPopExpr(0),
	pExpr(0),
	peExpr(0),
	eofExpr(0),
	csExpr(0),
	topExpr(0),
	stackExpr(0),
	actExpr(0),
	tokstartExpr(0),
	tokendExpr(0),
	dataExpr(0),
	wantComplete(true),
	hasLongestMatch(false),
	noEnd(false),
	noPrefix(false),
	noFinal(false),
	noError(false),
	noEntry(false),
	noCS(false)
{}


void CodeGenData::createMachine()
{
	redFsm = new RedFsmAp();
}

void CodeGenData::initActionList( unsigned long length )
{ 
	allActions = new GenAction[length];
	for ( unsigned long a = 0; a < length; a++ )
		actionList.append( allActions+a );
}

void CodeGenData::newAction( int anum, const char *name,
		const InputLoc &loc, GenInlineList *inlineList )
{
	allActions[anum].actionId = anum;
	allActions[anum].name = name;
	allActions[anum].loc = loc;
	allActions[anum].inlineList = inlineList;
}

void CodeGenData::initActionTableList( unsigned long length )
{ 
	allActionTables = new RedAction[length];
}

void CodeGenData::initStateList( unsigned long length )
{
	allStates = new RedStateAp[length];
	for ( unsigned long s = 0; s < length; s++ )
		redFsm->stateList.append( allStates+s );

	/* We get the start state as an offset, set the pointer now. */
	if ( startState >= 0 )
		redFsm->startState = allStates + startState;
	if ( errState >= 0 )
		redFsm->errState = allStates + errState;
	for ( EntryIdVect::Iter en = entryPointIds; en.lte(); en++ )
		redFsm->entryPoints.insert( allStates + *en );

	/* The nextStateId is no longer used to assign state ids (they come in set
	 * from the frontend now), however generation code still depends on it.
	 * Should eventually remove this variable. */
	redFsm->nextStateId = redFsm->stateList.length();
}

void CodeGenData::setStartState( unsigned long startState )
{
	this->startState = startState;
}

void CodeGenData::setErrorState( unsigned long errState )
{
	this->errState = errState;
}

void CodeGenData::addEntryPoint( char *name, unsigned long entryState )
{
	entryPointIds.append( entryState );
	entryPointNames.append( name );
}

void CodeGenData::initTransList( int snum, unsigned long length )
{
	/* Could preallocate the out range to save time growing it. For now do
	 * nothing. */
}

void CodeGenData::newTrans( int snum, int tnum, Key lowKey, 
		Key highKey, long targ, long action )
{
	/* Get the current state and range. */
	RedStateAp *curState = allStates + snum;
	RedTransList &destRange = curState->outRange;

	if ( curState == redFsm->errState )
		return;

	/* Make the new transitions. */
	RedStateAp *targState = targ >= 0 ? (allStates + targ) : 
			wantComplete ? redFsm->getErrorState() : 0;
	RedAction *actionTable = action >= 0 ? (allActionTables + action) : 0;
	RedTransAp *trans = redFsm->allocateTrans( targState, actionTable );
	RedTransEl transEl( lowKey, highKey, trans );

	if ( wantComplete ) {
		/* If the machine is to be complete then we need to fill any gaps with
		 * the error transitions. */
		if ( destRange.length() == 0 ) {
			/* Range is currently empty. */
			if ( keyOps->minKey < lowKey ) {
				/* The first range doesn't start at the low end. */
				Key fillHighKey = lowKey;
				fillHighKey.decrement();

				/* Create the filler with the state's error transition. */
				RedTransEl newTel( keyOps->minKey, fillHighKey, redFsm->getErrorTrans() );
				destRange.append( newTel );
			}
		}
		else {
			/* The range list is not empty, get the the last range. */
			RedTransEl *last = &destRange[destRange.length()-1];
			Key nextKey = last->highKey;
			nextKey.increment();
			if ( nextKey < lowKey ) {
				/* There is a gap to fill. Make the high key. */
				Key fillHighKey = lowKey;
				fillHighKey.decrement();

				/* Create the filler with the state's error transtion. */
				RedTransEl newTel( nextKey, fillHighKey, redFsm->getErrorTrans() );
				destRange.append( newTel );
			}
		}
	}

	/* Filler taken care of. Append the range. */
	destRange.append( RedTransEl( lowKey, highKey, trans ) );
}

void CodeGenData::finishTransList( int snum )
{
	/* Get the current state and range. */
	RedStateAp *curState = allStates + snum;
	RedTransList &destRange = curState->outRange;

	if ( curState == redFsm->errState )
		return;

	/* If building a complete machine we may need filler on the end. */
	if ( wantComplete ) {
		/* Check if there are any ranges already. */
		if ( destRange.length() == 0 ) {
			/* Fill with the whole alphabet. */
			/* Add the range on the lower and upper bound. */
			RedTransEl newTel( keyOps->minKey, keyOps->maxKey, redFsm->getErrorTrans() );
			destRange.append( newTel );
		}
		else {
			/* Get the last and check for a gap on the end. */
			RedTransEl *last = &destRange[destRange.length()-1];
			if ( last->highKey < keyOps->maxKey ) {
				/* Make the high key. */
				Key fillLowKey = last->highKey;
				fillLowKey.increment();

				/* Create the new range with the error trans and append it. */
				RedTransEl newTel( fillLowKey, keyOps->maxKey, redFsm->getErrorTrans() );
				destRange.append( newTel );
			}
		}
	}
}

void CodeGenData::setId( int snum, int id )
{
	RedStateAp *curState = allStates + snum;
	curState->id = id;
}

void CodeGenData::setFinal( int snum )
{
	RedStateAp *curState = allStates + snum;
	curState->isFinal = true;
}


void CodeGenData::setStateActions( int snum, long toStateAction, 
		long fromStateAction, long eofAction )
{
	RedStateAp *curState = allStates + snum;
	if ( toStateAction >= 0 )
		curState->toStateAction = allActionTables + toStateAction;
	if ( fromStateAction >= 0 )
		curState->fromStateAction = allActionTables + fromStateAction;
	if ( eofAction >= 0 )
		curState->eofAction = allActionTables + eofAction;
}

void CodeGenData::setEofTrans( int snum, long eofTarget, long actId )
{
	RedStateAp *curState = allStates + snum;
	RedStateAp *targState = allStates + eofTarget;
	RedAction *eofAct = allActionTables + actId;
	curState->eofTrans = redFsm->allocateTrans( targState, eofAct );
}

void CodeGenData::resolveTargetStates( GenInlineList *inlineList )
{
	for ( GenInlineList::Iter item = *inlineList; item.lte(); item++ ) {
		switch ( item->type ) {
		case GenInlineItem::Goto: case GenInlineItem::Call:
		case GenInlineItem::Next: case GenInlineItem::Entry:
			item->targState = allStates + item->targId;
			break;
		default:
			break;
		}

		if ( item->children != 0 )
			resolveTargetStates( item->children );
	}
}

void CodeGenData::closeMachine()
{
	for ( GenActionList::Iter a = actionList; a.lte(); a++ )
		resolveTargetStates( a->inlineList );

	/* Note that even if we want a complete graph we do not give the error
	 * state a default transition. All machines break out of the processing
	 * loop when in the error state. */

	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		for ( GenStateCondList::Iter sci = st->stateCondList; sci.lte(); sci++ )
			st->stateCondVect.append( sci );
	}
}


bool CodeGenData::setAlphType( const char *data )
{
	HostType *alphType = findAlphTypeInternal( data );
	if ( alphType == 0 )
		return false;

	thisKeyOps.setAlphType( alphType );
	return true;
}

void CodeGenData::initCondSpaceList( ulong length )
{
	allCondSpaces = new GenCondSpace[length];
	for ( ulong c = 0; c < length; c++ )
		condSpaceList.append( allCondSpaces + c );
}

void CodeGenData::newCondSpace( int cnum, int condSpaceId, Key baseKey )
{
	GenCondSpace *cond = allCondSpaces + cnum;
	cond->condSpaceId = condSpaceId;
	cond->baseKey = baseKey;
}

void CodeGenData::condSpaceItem( int cnum, long condActionId )
{
	GenCondSpace *cond = allCondSpaces + cnum;
	cond->condSet.append( allActions + condActionId );
}

void CodeGenData::initStateCondList( int snum, ulong length )
{
	/* Could preallocate these, as we could with transitions. */
}

void CodeGenData::addStateCond( int snum, Key lowKey, Key highKey, long condNum )
{
	RedStateAp *curState = allStates + snum;

	/* Create the new state condition. */
	GenStateCond *stateCond = new GenStateCond;
	stateCond->lowKey = lowKey;
	stateCond->highKey = highKey;

	/* Assign it a cond space. */
	GenCondSpace *condSpace = allCondSpaces + condNum;
	stateCond->condSpace = condSpace;

	curState->stateCondList.append( stateCond );
}


GenCondSpace *CodeGenData::findCondSpace( Key lowKey, Key highKey )
{
	for ( CondSpaceList::Iter cs = condSpaceList; cs.lte(); cs++ ) {
		Key csHighKey = cs->baseKey;
		csHighKey += keyOps->alphSize() * (1 << cs->condSet.length());

		if ( lowKey >= cs->baseKey && highKey <= csHighKey )
			return cs;
	}
	return 0;
}

Condition *CodeGenData::findCondition( Key key )
{
	for ( ConditionList::Iter cond = conditionList; cond.lte(); cond++ ) {
		Key upperKey = cond->baseKey + (1 << cond->condSet.length());
		if ( cond->baseKey <= key && key <= upperKey )
			return cond;
	}
	return 0;
}

Key CodeGenData::findMaxKey()
{
	Key maxKey = keyOps->maxKey;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		assert( st->outSingle.length() == 0 );
		assert( st->defTrans == 0 );

		long rangeLen = st->outRange.length();
		if ( rangeLen > 0 ) {
			Key highKey = st->outRange[rangeLen-1].highKey;
			if ( highKey > maxKey )
				maxKey = highKey;
		}
	}
	return maxKey;
}

void CodeGenData::findFinalActionRefs()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Rerence count out of single transitions. */
		for ( RedTransList::Iter rtel = st->outSingle; rtel.lte(); rtel++ ) {
			if ( rtel->value->action != 0 ) {
				rtel->value->action->numTransRefs += 1;
				for ( GenActionTable::Iter item = rtel->value->action->key; item.lte(); item++ )
					item->value->numTransRefs += 1;
			}
		}

		/* Reference count out of range transitions. */
		for ( RedTransList::Iter rtel = st->outRange; rtel.lte(); rtel++ ) {
			if ( rtel->value->action != 0 ) {
				rtel->value->action->numTransRefs += 1;
				for ( GenActionTable::Iter item = rtel->value->action->key; item.lte(); item++ )
					item->value->numTransRefs += 1;
			}
		}

		/* Reference count default transition. */
		if ( st->defTrans != 0 && st->defTrans->action != 0 ) {
			st->defTrans->action->numTransRefs += 1;
			for ( GenActionTable::Iter item = st->defTrans->action->key; item.lte(); item++ )
				item->value->numTransRefs += 1;
		}

		/* Reference count eof transitions. */
		if ( st->eofTrans != 0 && st->eofTrans->action != 0 ) {
			st->eofTrans->action->numTransRefs += 1;
			for ( GenActionTable::Iter item = st->eofTrans->action->key; item.lte(); item++ )
				item->value->numTransRefs += 1;
		}

		/* Reference count to state actions. */
		if ( st->toStateAction != 0 ) {
			st->toStateAction->numToStateRefs += 1;
			for ( GenActionTable::Iter item = st->toStateAction->key; item.lte(); item++ )
				item->value->numToStateRefs += 1;
		}

		/* Reference count from state actions. */
		if ( st->fromStateAction != 0 ) {
			st->fromStateAction->numFromStateRefs += 1;
			for ( GenActionTable::Iter item = st->fromStateAction->key; item.lte(); item++ )
				item->value->numFromStateRefs += 1;
		}

		/* Reference count EOF actions. */
		if ( st->eofAction != 0 ) {
			st->eofAction->numEofRefs += 1;
			for ( GenActionTable::Iter item = st->eofAction->key; item.lte(); item++ )
				item->value->numEofRefs += 1;
		}
	}
}

void CodeGenData::analyzeAction( GenAction *act, GenInlineList *inlineList )
{
	for ( GenInlineList::Iter item = *inlineList; item.lte(); item++ ) {
		/* Only consider actions that are referenced. */
		if ( act->numRefs() > 0 ) {
			if ( item->type == GenInlineItem::Goto || item->type == GenInlineItem::GotoExpr )
				redFsm->bAnyActionGotos = true;
			else if ( item->type == GenInlineItem::Call || item->type == GenInlineItem::CallExpr )
				redFsm->bAnyActionCalls = true;
			else if ( item->type == GenInlineItem::Ret )
				redFsm->bAnyActionRets = true;

			if ( item->type == GenInlineItem::CallExpr || item->type == GenInlineItem::GotoExpr )
				redFsm->bAnyActionByValControl = true;

		}

		/* Check for various things in regular actions. */
		if ( act->numTransRefs > 0 || act->numToStateRefs > 0 || act->numFromStateRefs > 0 ) {
			/* Any returns in regular actions? */
			if ( item->type == GenInlineItem::Ret )
				redFsm->bAnyRegActionRets = true;

			/* Any next statements in the regular actions? */
			if ( item->type == GenInlineItem::Next || item->type == GenInlineItem::NextExpr )
				redFsm->bAnyRegNextStmt = true;

			/* Any by value control in regular actions? */
			if ( item->type == GenInlineItem::CallExpr || item->type == GenInlineItem::GotoExpr )
				redFsm->bAnyRegActionByValControl = true;

			/* Any references to the current state in regular actions? */
			if ( item->type == GenInlineItem::Curs )
				redFsm->bAnyRegCurStateRef = true;

			if ( item->type == GenInlineItem::Break )
				redFsm->bAnyRegBreak = true;
		}

		if ( item->children != 0 )
			analyzeAction( act, item->children );
	}
}

void CodeGenData::analyzeActionList( RedAction *redAct, GenInlineList *inlineList )
{
	for ( GenInlineList::Iter item = *inlineList; item.lte(); item++ ) {
		/* Any next statements in the action table? */
		if ( item->type == GenInlineItem::Next || item->type == GenInlineItem::NextExpr )
			redAct->bAnyNextStmt = true;

		/* Any references to the current state. */
		if ( item->type == GenInlineItem::Curs )
			redAct->bAnyCurStateRef = true;

		if ( item->type == GenInlineItem::Break )
			redAct->bAnyBreakStmt = true;

		if ( item->children != 0 )
			analyzeActionList( redAct, item->children );
	}
}

/* Assign ids to referenced actions. */
void CodeGenData::assignActionIds()
{
	int nextActionId = 0;
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Only ever interested in referenced actions. */
		if ( act->numRefs() > 0 )
			act->actionId = nextActionId++;
	}
}

void CodeGenData::setValueLimits()
{
	redFsm->maxSingleLen = 0;
	redFsm->maxRangeLen = 0;
	redFsm->maxKeyOffset = 0;
	redFsm->maxIndexOffset = 0;
	redFsm->maxActListId = 0;
	redFsm->maxActionLoc = 0;
	redFsm->maxActArrItem = 0;
	redFsm->maxSpan = 0;
	redFsm->maxCondSpan = 0;
	redFsm->maxFlatIndexOffset = 0;
	redFsm->maxCondOffset = 0;
	redFsm->maxCondLen = 0;
	redFsm->maxCondSpaceId = 0;
	redFsm->maxCondIndexOffset = 0;

	/* In both of these cases the 0 index is reserved for no value, so the max
	 * is one more than it would be if they started at 0. */
	redFsm->maxIndex = redFsm->transSet.length();
	redFsm->maxCond = condSpaceList.length(); 

	/* The nextStateId - 1 is the last state id assigned. */
	redFsm->maxState = redFsm->nextStateId - 1;

	for ( CondSpaceList::Iter csi = condSpaceList; csi.lte(); csi++ ) {
		if ( csi->condSpaceId > redFsm->maxCondSpaceId )
			redFsm->maxCondSpaceId = csi->condSpaceId;
	}

	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Maximum cond length. */
		if ( st->stateCondList.length() > redFsm->maxCondLen )
			redFsm->maxCondLen = st->stateCondList.length();

		/* Maximum single length. */
		if ( st->outSingle.length() > redFsm->maxSingleLen )
			redFsm->maxSingleLen = st->outSingle.length();

		/* Maximum range length. */
		if ( st->outRange.length() > redFsm->maxRangeLen )
			redFsm->maxRangeLen = st->outRange.length();

		/* The key offset index offset for the state after last is not used, skip it.. */
		if ( ! st.last() ) {
			redFsm->maxCondOffset += st->stateCondList.length();
			redFsm->maxKeyOffset += st->outSingle.length() + st->outRange.length()*2;
			redFsm->maxIndexOffset += st->outSingle.length() + st->outRange.length() + 2;
		}

		/* Max cond span. */
		if ( st->condList != 0 ) {
			unsigned long long span = keyOps->span( st->condLowKey, st->condHighKey );
			if ( span > redFsm->maxCondSpan )
				redFsm->maxCondSpan = span;
		}

		/* Max key span. */
		if ( st->transList != 0 ) {
			unsigned long long span = keyOps->span( st->lowKey, st->highKey );
			if ( span > redFsm->maxSpan )
				redFsm->maxSpan = span;
		}

		/* Max cond index offset. */
		if ( ! st.last() ) {
			if ( st->condList != 0 )
				redFsm->maxCondIndexOffset += keyOps->span( st->condLowKey, st->condHighKey );
		}

		/* Max flat index offset. */
		if ( ! st.last() ) {
			if ( st->transList != 0 )
				redFsm->maxFlatIndexOffset += keyOps->span( st->lowKey, st->highKey );
			redFsm->maxFlatIndexOffset += 1;
		}
	}

	for ( GenActionTableMap::Iter at = redFsm->actionMap; at.lte(); at++ ) {
		/* Maximum id of action lists. */
		if ( at->actListId+1 > redFsm->maxActListId )
			redFsm->maxActListId = at->actListId+1;

		/* Maximum location of items in action array. */
		if ( at->location+1 > redFsm->maxActionLoc )
			redFsm->maxActionLoc = at->location+1;

		/* Maximum values going into the action array. */
		if ( at->key.length() > redFsm->maxActArrItem )
			redFsm->maxActArrItem = at->key.length();
		for ( GenActionTable::Iter item = at->key; item.lte(); item++ ) {
			if ( item->value->actionId > redFsm->maxActArrItem )
				redFsm->maxActArrItem = item->value->actionId;
		}
	}
}



/* Gather various info on the machine. */
void CodeGenData::analyzeMachine()
{
	/* Find the true count of action references.  */
	findFinalActionRefs();

	/* Check if there are any calls in action code. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Record the occurrence of various kinds of actions. */
		if ( act->numToStateRefs > 0 )
			redFsm->bAnyToStateActions = true;
		if ( act->numFromStateRefs > 0 )
			redFsm->bAnyFromStateActions = true;
		if ( act->numEofRefs > 0 )
			redFsm->bAnyEofActions = true;
		if ( act->numTransRefs > 0 )
			redFsm->bAnyRegActions = true;

		/* Recurse through the action's parse tree looking for various things. */
		analyzeAction( act, act->inlineList );
	}

	/* Analyze reduced action lists. */
	for ( GenActionTableMap::Iter redAct = redFsm->actionMap; redAct.lte(); redAct++ ) {
		for ( GenActionTable::Iter act = redAct->key; act.lte(); act++ )
			analyzeActionList( redAct, act->value->inlineList );
	}

	/* Find states that have transitions with actions that have next
	 * statements. */
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Check any actions out of outSinge. */
		for ( RedTransList::Iter rtel = st->outSingle; rtel.lte(); rtel++ ) {
			if ( rtel->value->action != 0 && rtel->value->action->anyCurStateRef() )
				st->bAnyRegCurStateRef = true;
		}

		/* Check any actions out of outRange. */
		for ( RedTransList::Iter rtel = st->outRange; rtel.lte(); rtel++ ) {
			if ( rtel->value->action != 0 && rtel->value->action->anyCurStateRef() )
				st->bAnyRegCurStateRef = true;
		}

		/* Check any action out of default. */
		if ( st->defTrans != 0 && st->defTrans->action != 0 && 
				st->defTrans->action->anyCurStateRef() )
			st->bAnyRegCurStateRef = true;
		
		if ( st->stateCondList.length() > 0 )
			redFsm->bAnyConditions = true;

		if ( st->eofTrans != 0 )
			redFsm->bAnyEofTrans = true;
	}

	/* Assign ids to actions that are referenced. */
	assignActionIds();

	/* Set the maximums of various values used for deciding types. */
	setValueLimits();
}

void CodeGenData::write_option_error( InputLoc &loc, char *arg )
{
	source_warning(loc) << "unrecognized write option \"" << arg << "\"" << endl;
}

/* returns true if the following section should generate line directives. */
bool CodeGenData::writeStatement( InputLoc &loc, int nargs, char **args )
{
	bool followLineDirective = false;

	if ( strcmp( args[0], "data" ) == 0 ) {
		out << '\n';
		genLineDirective( out );
		followLineDirective = true;

		for ( int i = 1; i < nargs; i++ ) {
			if ( strcmp( args[i], "noerror" ) == 0 )
				noError = true;
			else if ( strcmp( args[i], "noprefix" ) == 0 )
				noPrefix = true;
			else if ( strcmp( args[i], "nofinal" ) == 0 )
				noFinal = true;
			else if ( strcmp( args[i], "noentry" ) == 0 )
				noEntry = true;
			else
				write_option_error( loc, args[i] );
		}
		writeData();
	}
	else if ( strcmp( args[0], "init" ) == 0 ) {
		out << '\n';
		genLineDirective( out );
		followLineDirective = true;

		for ( int i = 1; i < nargs; i++ ) {
			if ( strcmp( args[i], "nocs" ) == 0 )
				noCS = true;
			else
				write_option_error( loc, args[i] );
		}
		writeInit();
	}
	else if ( strcmp( args[0], "exec" ) == 0 ) {
		out << '\n';
		genLineDirective( out );
		followLineDirective = true;

		for ( int i = 1; i < nargs; i++ ) {
			if ( strcmp( args[i], "noend" ) == 0 )
				noEnd = true;
			else
				write_option_error( loc, args[i] );
		}
		writeExec();
	}
	else if ( strcmp( args[0], "exports" ) == 0 ) {
		out << '\n';
		genLineDirective( out );
		followLineDirective = true;

		for ( int i = 1; i < nargs; i++ )
			write_option_error( loc, args[i] );
		writeExports();
	}
	else if ( strcmp( args[0], "start" ) == 0 ) {
		for ( int i = 1; i < nargs; i++ )
			write_option_error( loc, args[i] );
		writeStart();
	}
	else if ( strcmp( args[0], "first_final" ) == 0 ) {
		for ( int i = 1; i < nargs; i++ )
			write_option_error( loc, args[i] );
		writeFirstFinal();
	}
	else if ( strcmp( args[0], "error" ) == 0 ) {
		for ( int i = 1; i < nargs; i++ )
			write_option_error( loc, args[i] );
		writeError();
	}
	else {
		/* EMIT An error here. */
		source_error(loc) << "unrecognized write command \"" << 
				args[0] << "\"" << endl;
	}
	return followLineDirective;
}

ostream &CodeGenData::source_warning( const InputLoc &loc )
{
	cerr << sourceFileName << ":" << loc.line << ":" << loc.col << ": warning: ";
	return cerr;
}

ostream &CodeGenData::source_error( const InputLoc &loc )
{
	gblErrorCount += 1;
	assert( sourceFileName != 0 );
	cerr << sourceFileName << ":" << loc.line << ":" << loc.col << ": ";
	return cerr;
}


