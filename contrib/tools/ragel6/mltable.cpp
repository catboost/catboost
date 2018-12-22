/*
 *  Copyright 2001-2006 Adrian Thurston <thurston@complang.org>
 *            2004 Erich Ocean <eric.ocean@ampede.com>
 *            2005 Alan West <alan@alanz.com>
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

#include "ragel.h"
#include "mltable.h"
#include "redfsm.h"
#include "gendata.h"

/* Determine if we should use indicies or not. */
void OCamlTabCodeGen::calcIndexSize()
{
	int sizeWithInds = 0, sizeWithoutInds = 0;

	/* Calculate cost of using with indicies. */
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		int totalIndex = st->outSingle.length() + st->outRange.length() + 
				(st->defTrans == 0 ? 0 : 1);
		sizeWithInds += arrayTypeSize(redFsm->maxIndex) * totalIndex;
	}
	sizeWithInds += arrayTypeSize(redFsm->maxState) * redFsm->transSet.length();
	if ( redFsm->anyActions() )
		sizeWithInds += arrayTypeSize(redFsm->maxActionLoc) * redFsm->transSet.length();

	/* Calculate the cost of not using indicies. */
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		int totalIndex = st->outSingle.length() + st->outRange.length() + 
				(st->defTrans == 0 ? 0 : 1);
		sizeWithoutInds += arrayTypeSize(redFsm->maxState) * totalIndex;
		if ( redFsm->anyActions() )
			sizeWithoutInds += arrayTypeSize(redFsm->maxActionLoc) * totalIndex;
	}

	/* If using indicies reduces the size, use them. */
	useIndicies = sizeWithInds < sizeWithoutInds;
}

std::ostream &OCamlTabCodeGen::TO_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->toStateAction != 0 )
		act = state->toStateAction->location+1;
	out << act;
	return out;
}

std::ostream &OCamlTabCodeGen::FROM_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->fromStateAction != 0 )
		act = state->fromStateAction->location+1;
	out << act;
	return out;
}

std::ostream &OCamlTabCodeGen::EOF_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->eofAction != 0 )
		act = state->eofAction->location+1;
	out << act;
	return out;
}


std::ostream &OCamlTabCodeGen::TRANS_ACTION( RedTransAp *trans )
{
	/* If there are actions, emit them. Otherwise emit zero. */
	int act = 0;
	if ( trans->action != 0 )
		act = trans->action->location+1;
	out << act;
	return out;
}

std::ostream &OCamlTabCodeGen::TO_STATE_ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numToStateRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << "\t| " << act->actionId << " ->\n";
      ACTION( out, act, 0, false );
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &OCamlTabCodeGen::FROM_STATE_ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numFromStateRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << "\t| " << act->actionId << " ->\n";
			ACTION( out, act, 0, false );
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &OCamlTabCodeGen::EOF_ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numEofRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << "\t| " << act->actionId << " ->\n";
			ACTION( out, act, 0, true );
		}
	}

	genLineDirective( out );
	return out;
}


std::ostream &OCamlTabCodeGen::ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numTransRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << "\t| " << act->actionId << " -> \n";
			ACTION( out, act, 0, false );
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &OCamlTabCodeGen::COND_OFFSETS()
{
	out << "\t";
	int totalStateNum = 0, curKeyOffset = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write the key offset. */
		out << curKeyOffset;
		if ( !st.last() ) {
			out << ARR_SEP();
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}

		/* Move the key offset ahead. */
		curKeyOffset += st->stateCondList.length();
	}
	out << "\n";
	return out;
}

std::ostream &OCamlTabCodeGen::KEY_OFFSETS()
{
	out << "\t";
	int totalStateNum = 0, curKeyOffset = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write the key offset. */
		out << curKeyOffset;
		if ( !st.last() ) {
			out << ARR_SEP();
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}

		/* Move the key offset ahead. */
		curKeyOffset += st->outSingle.length() + st->outRange.length()*2;
	}
	out << "\n";
	return out;
}


std::ostream &OCamlTabCodeGen::INDEX_OFFSETS()
{
	out << "\t";
	int totalStateNum = 0, curIndOffset = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write the index offset. */
		out << curIndOffset;
		if ( !st.last() ) {
			out << ARR_SEP();
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}

		/* Move the index offset ahead. */
		curIndOffset += st->outSingle.length() + st->outRange.length();
		if ( st->defTrans != 0 )
			curIndOffset += 1;
	}
	out << "\n";
	return out;
}

std::ostream &OCamlTabCodeGen::COND_LENS()
{
	out << "\t";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write singles length. */
		out << st->stateCondList.length();
		if ( !st.last() ) {
			out << ARR_SEP();
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}


std::ostream &OCamlTabCodeGen::SINGLE_LENS()
{
	out << "\t";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write singles length. */
		out << st->outSingle.length();
		if ( !st.last() ) {
			out << ARR_SEP();
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}

std::ostream &OCamlTabCodeGen::RANGE_LENS()
{
	out << "\t";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Emit length of range index. */
		out << st->outRange.length();
		if ( !st.last() ) {
			out << ARR_SEP();
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}

std::ostream &OCamlTabCodeGen::TO_STATE_ACTIONS()
{
	out << "\t";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write any eof action. */
		TO_STATE_ACTION(st);
		if ( !st.last() ) {
			out << ARR_SEP();
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}

std::ostream &OCamlTabCodeGen::FROM_STATE_ACTIONS()
{
	out << "\t";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write any eof action. */
		FROM_STATE_ACTION(st);
		if ( !st.last() ) {
			out << ARR_SEP();
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}

std::ostream &OCamlTabCodeGen::EOF_ACTIONS()
{
	out << "\t";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write any eof action. */
		EOF_ACTION(st);
		if ( !st.last() ) {
			out << ARR_SEP();
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}

std::ostream &OCamlTabCodeGen::EOF_TRANS()
{
	out << "\t";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write any eof action. */
		long trans = 0;
		if ( st->eofTrans != 0 ) {
			assert( st->eofTrans->pos >= 0 );
			trans = st->eofTrans->pos+1;
		}
		out << trans;

		if ( !st.last() ) {
			out << ARR_SEP();
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}


std::ostream &OCamlTabCodeGen::COND_KEYS()
{
	out << '\t';
	int totalTrans = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Loop the state's transitions. */
		for ( GenStateCondList::Iter sc = st->stateCondList; sc.lte(); sc++ ) {
			/* Lower key. */
			out << ALPHA_KEY( sc->lowKey ) << ARR_SEP();
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";

			/* Upper key. */
			out << ALPHA_KEY( sc->highKey ) << ARR_SEP();
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}
	}

	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	out << 0 << "\n";
	return out;
}

std::ostream &OCamlTabCodeGen::COND_SPACES()
{
	out << '\t';
	int totalTrans = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Loop the state's transitions. */
		for ( GenStateCondList::Iter sc = st->stateCondList; sc.lte(); sc++ ) {
			/* Cond Space id. */
			out << sc->condSpace->condSpaceId << ARR_SEP();
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}
	}

	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	out << 0 << "\n";
	return out;
}

std::ostream &OCamlTabCodeGen::KEYS()
{
	out << '\t';
	int totalTrans = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Loop the singles. */
		for ( RedTransList::Iter stel = st->outSingle; stel.lte(); stel++ ) {
			out << ALPHA_KEY( stel->lowKey ) << ARR_SEP();
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}

		/* Loop the state's transitions. */
		for ( RedTransList::Iter rtel = st->outRange; rtel.lte(); rtel++ ) {
			/* Lower key. */
			out << ALPHA_KEY( rtel->lowKey ) << ARR_SEP();
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";

			/* Upper key. */
			out << ALPHA_KEY( rtel->highKey ) << ARR_SEP();
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}
	}

	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	out << 0 << "\n";
	return out;
}

std::ostream &OCamlTabCodeGen::INDICIES()
{
	int totalTrans = 0;
	out << '\t';
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Walk the singles. */
		for ( RedTransList::Iter stel = st->outSingle; stel.lte(); stel++ ) {
			out << stel->value->id << ARR_SEP();
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}

		/* Walk the ranges. */
		for ( RedTransList::Iter rtel = st->outRange; rtel.lte(); rtel++ ) {
			out << rtel->value->id << ARR_SEP();
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}

		/* The state's default index goes next. */
		if ( st->defTrans != 0 ) {
			out << st->defTrans->id << ARR_SEP();
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}
	}

	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	out << 0 << "\n";
	return out;
}

std::ostream &OCamlTabCodeGen::TRANS_TARGS()
{
	int totalTrans = 0;
	out << '\t';
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Walk the singles. */
		for ( RedTransList::Iter stel = st->outSingle; stel.lte(); stel++ ) {
			RedTransAp *trans = stel->value;
			out << trans->targ->id << ARR_SEP();
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}

		/* Walk the ranges. */
		for ( RedTransList::Iter rtel = st->outRange; rtel.lte(); rtel++ ) {
			RedTransAp *trans = rtel->value;
			out << trans->targ->id << ARR_SEP();
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}

		/* The state's default target state. */
		if ( st->defTrans != 0 ) {
			RedTransAp *trans = st->defTrans;
			out << trans->targ->id << ARR_SEP();
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}
	}

	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st->eofTrans != 0 ) {
			RedTransAp *trans = st->eofTrans;
			trans->pos = totalTrans;
			out << trans->targ->id << ARR_SEP();
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}
	}


	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	out << 0 << "\n";
	return out;
}


std::ostream &OCamlTabCodeGen::TRANS_ACTIONS()
{
	int totalTrans = 0;
	out << '\t';
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Walk the singles. */
		for ( RedTransList::Iter stel = st->outSingle; stel.lte(); stel++ ) {
			RedTransAp *trans = stel->value;
			TRANS_ACTION( trans ) << ARR_SEP();
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}

		/* Walk the ranges. */
		for ( RedTransList::Iter rtel = st->outRange; rtel.lte(); rtel++ ) {
			RedTransAp *trans = rtel->value;
			TRANS_ACTION( trans ) << ARR_SEP();
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}

		/* The state's default index goes next. */
		if ( st->defTrans != 0 ) {
			RedTransAp *trans = st->defTrans;
			TRANS_ACTION( trans ) << ARR_SEP();
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}
	}

	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st->eofTrans != 0 ) {
			RedTransAp *trans = st->eofTrans;
			TRANS_ACTION( trans ) << ARR_SEP();
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}
	}

	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	out << 0 << "\n";
	return out;
}

std::ostream &OCamlTabCodeGen::TRANS_TARGS_WI()
{
	/* Transitions must be written ordered by their id. */
	RedTransAp **transPtrs = new RedTransAp*[redFsm->transSet.length()];
	for ( TransApSet::Iter trans = redFsm->transSet; trans.lte(); trans++ )
		transPtrs[trans->id] = trans;

	/* Keep a count of the num of items in the array written. */
	out << '\t';
	int totalStates = 0;
	for ( int t = 0; t < redFsm->transSet.length(); t++ ) {
		/* Record the position, need this for eofTrans. */
		RedTransAp *trans = transPtrs[t];
		trans->pos = t;

		/* Write out the target state. */
		out << trans->targ->id;
		if ( t < redFsm->transSet.length()-1 ) {
			out << ARR_SEP();
			if ( ++totalStates % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	delete[] transPtrs;
	return out;
}


std::ostream &OCamlTabCodeGen::TRANS_ACTIONS_WI()
{
	/* Transitions must be written ordered by their id. */
	RedTransAp **transPtrs = new RedTransAp*[redFsm->transSet.length()];
	for ( TransApSet::Iter trans = redFsm->transSet; trans.lte(); trans++ )
		transPtrs[trans->id] = trans;

	/* Keep a count of the num of items in the array written. */
	out << '\t';
	int totalAct = 0;
	for ( int t = 0; t < redFsm->transSet.length(); t++ ) {
		/* Write the function for the transition. */
		RedTransAp *trans = transPtrs[t];
		TRANS_ACTION( trans );
		if ( t < redFsm->transSet.length()-1 ) {
			out << ARR_SEP();
			if ( ++totalAct % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	delete[] transPtrs;
	return out;
}

void OCamlTabCodeGen::GOTO( ostream &ret, int gotoDest, bool inFinish )
{
	ret << "begin " << vCS() << " <- " << gotoDest << "; " << 
			CTRL_FLOW() << "raise Goto_again end";
}

void OCamlTabCodeGen::GOTO_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish )
{
	ret << "begin " << vCS() << " <- (";
	INLINE_LIST( ret, ilItem->children, 0, inFinish );
	ret << "); " << CTRL_FLOW() << "raise Goto_again end";
}

void OCamlTabCodeGen::CURS( ostream &ret, bool inFinish )
{
	ret << "(_ps)";
}

void OCamlTabCodeGen::TARGS( ostream &ret, bool inFinish, int targState )
{
	ret << "(" << vCS() << ")";
}

void OCamlTabCodeGen::NEXT( ostream &ret, int nextDest, bool inFinish )
{
	ret << vCS() << " <- " << nextDest << ";";
}

void OCamlTabCodeGen::NEXT_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish )
{
	ret << vCS() << " <- (";
	INLINE_LIST( ret, ilItem->children, 0, inFinish );
	ret << ");";
}

void OCamlTabCodeGen::CALL( ostream &ret, int callDest, int targState, bool inFinish )
{
	if ( prePushExpr != 0 ) {
		ret << "begin ";
		INLINE_LIST( ret, prePushExpr, 0, false );
	}

	ret << "begin " << AT( STACK(), POST_INCR(TOP()) ) << " <- " << vCS() << "; ";
  ret << vCS() << " <- " << callDest << "; " << CTRL_FLOW() << "raise Goto_again end ";

	if ( prePushExpr != 0 )
		ret << "end";
}

void OCamlTabCodeGen::CALL_EXPR( ostream &ret, GenInlineItem *ilItem, int targState, bool inFinish )
{
	if ( prePushExpr != 0 ) {
		ret << "begin ";
		INLINE_LIST( ret, prePushExpr, 0, false );
	}

	ret << "begin " << AT(STACK(), POST_INCR(TOP()) ) << " <- " << vCS() << "; " << vCS() << " <- (";
	INLINE_LIST( ret, ilItem->children, targState, inFinish );
	ret << "); " << CTRL_FLOW() << "raise Goto_again end ";

	if ( prePushExpr != 0 )
		ret << "end";
}

void OCamlTabCodeGen::RET( ostream &ret, bool inFinish )
{
	ret << "begin " << vCS() << " <- " << AT(STACK(), PRE_DECR(TOP()) ) << "; ";

	if ( postPopExpr != 0 ) {
		ret << "begin ";
		INLINE_LIST( ret, postPopExpr, 0, false );
		ret << "end ";
	}

	ret << CTRL_FLOW() <<  "raise Goto_again end";
}

void OCamlTabCodeGen::BREAK( ostream &ret, int targState )
{
	outLabelUsed = true;
	ret << "begin " << P() << " <- " << P() << " + 1; " << CTRL_FLOW() << "raise Goto_out end";
}

void OCamlTabCodeGen::writeData()
{
	/* If there are any transtion functions then output the array. If there
	 * are none, don't bother emitting an empty array that won't be used. */
	if ( redFsm->anyActions() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActArrItem), A() );
		ACTIONS_ARRAY();
		CLOSE_ARRAY() <<
		"\n";
	}

	if ( redFsm->anyConditions() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxCondOffset), CO() );
		COND_OFFSETS();
		CLOSE_ARRAY() <<
		"\n";

		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxCondLen), CL() );
		COND_LENS();
		CLOSE_ARRAY() <<
		"\n";

		OPEN_ARRAY( WIDE_ALPH_TYPE(), CK() );
		COND_KEYS();
		CLOSE_ARRAY() <<
		"\n";

		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxCondSpaceId), C() );
		COND_SPACES();
		CLOSE_ARRAY() <<
		"\n";
	}

	OPEN_ARRAY( ARRAY_TYPE(redFsm->maxKeyOffset), KO() );
	KEY_OFFSETS();
	CLOSE_ARRAY() <<
	"\n";

	OPEN_ARRAY( WIDE_ALPH_TYPE(), K() );
	KEYS();
	CLOSE_ARRAY() <<
	"\n";

	OPEN_ARRAY( ARRAY_TYPE(redFsm->maxSingleLen), SL() );
	SINGLE_LENS();
	CLOSE_ARRAY() <<
	"\n";

	OPEN_ARRAY( ARRAY_TYPE(redFsm->maxRangeLen), RL() );
	RANGE_LENS();
	CLOSE_ARRAY() <<
	"\n";

	OPEN_ARRAY( ARRAY_TYPE(redFsm->maxIndexOffset), IO() );
	INDEX_OFFSETS();
	CLOSE_ARRAY() <<
	"\n";

	if ( useIndicies ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxIndex), I() );
		INDICIES();
		CLOSE_ARRAY() <<
		"\n";

		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxState), TT() );
		TRANS_TARGS_WI();
		CLOSE_ARRAY() <<
		"\n";

		if ( redFsm->anyActions() ) {
			OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActionLoc), TA() );
			TRANS_ACTIONS_WI();
			CLOSE_ARRAY() <<
			"\n";
		}
	}
	else {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxState), TT() );
		TRANS_TARGS();
		CLOSE_ARRAY() <<
		"\n";

		if ( redFsm->anyActions() ) {
			OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActionLoc), TA() );
			TRANS_ACTIONS();
			CLOSE_ARRAY() <<
			"\n";
		}
	}

	if ( redFsm->anyToStateActions() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActionLoc), TSA() );
		TO_STATE_ACTIONS();
		CLOSE_ARRAY() <<
		"\n";
	}

	if ( redFsm->anyFromStateActions() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActionLoc), FSA() );
		FROM_STATE_ACTIONS();
		CLOSE_ARRAY() <<
		"\n";
	}

	if ( redFsm->anyEofActions() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActionLoc), EA() );
		EOF_ACTIONS();
		CLOSE_ARRAY() <<
		"\n";
	}

	if ( redFsm->anyEofTrans() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxIndexOffset+1), ET() );
		EOF_TRANS();
		CLOSE_ARRAY() <<
		"\n";
	}

	STATE_IDS();

  out << "type " << TYPE_STATE() << " = { mutable keys : int; mutable trans : int; mutable acts : int; mutable nacts : int; }"
    << TOP_SEP();

  out << "exception Goto_match" << TOP_SEP();
  out << "exception Goto_again" << TOP_SEP();
  out << "exception Goto_eof_trans" << TOP_SEP();
}

void OCamlTabCodeGen::LOCATE_TRANS()
{
	out <<
		"	state.keys <- " << AT( KO(), vCS() ) << ";\n"
		"	state.trans <- " << CAST(transType) << AT( IO(), vCS() ) << ";\n"
		"\n"
		"	let klen = " << AT( SL(), vCS() ) << " in\n"
		"	if klen > 0 then begin\n"
		"		let lower : " << signedKeysType << " ref = ref state.keys in\n"
		"		let upper : " << signedKeysType << " ref = ref " << CAST(signedKeysType) << 
			"(state.keys + klen - 1) in\n"
		"		while !upper >= !lower do\n"
		"			let mid = " << CAST(signedKeysType) << " (!lower + ((!upper - !lower) / 2)) in\n"
		"			if " << GET_WIDE_KEY() << " < " << AT( K(), "mid" ) << " then\n"
		"				upper := " << CAST(signedKeysType) << " (mid - 1)\n"
		"			else if " << GET_WIDE_KEY() << " > " << AT( K(), "mid" ) << " then\n"
		"				lower := " << CAST(signedKeysType) << " (mid + 1)\n"
		"			else begin\n"
		"				state.trans <- state.trans + " << CAST(transType) << " (mid - state.keys);\n"
		"				raise Goto_match;\n"
		"			end\n"
		"		done;\n"
		"		state.keys <- state.keys + " << CAST(keysType) << " klen;\n"
		"		state.trans <- state.trans + " << CAST(transType) << " klen;\n"
		"	end;\n"
		"\n"
		"	let klen = " << AT( RL(), vCS() ) << " in\n"
		"	if klen > 0 then begin\n"
		"		let lower : " << signedKeysType << " ref = ref state.keys in\n"
		"		let upper : " << signedKeysType << " ref = ref " << CAST(signedKeysType) <<
			"(state.keys + (klen * 2) - 2) in\n"
		"		while !upper >= !lower do\n"
		"			let mid = " << CAST(signedKeysType) << " (!lower + (((!upper - !lower) / 2) land (lnot 1))) in\n"
		"			if " << GET_WIDE_KEY() << " < " << AT( K() , "mid" ) << " then\n"
		"				upper := " << CAST(signedKeysType) << " (mid - 2)\n"
		"			else if " << GET_WIDE_KEY() << " > " << AT( K(), "mid+1" ) << " then\n"
		"				lower := " << CAST(signedKeysType) << " (mid + 2)\n"
		"			else begin\n"
		"				state.trans <- state.trans + " << CAST(transType) << "((mid - state.keys) / 2);\n"
		"				raise Goto_match;\n"
		"		  end\n"
		"		done;\n"
		"		state.trans <- state.trans + " << CAST(transType) << " klen;\n"
		"	end;\n"
		"\n";
}

void OCamlTabCodeGen::COND_TRANSLATE()
{
	out << 
		"	_widec = " << GET_KEY() << ";\n"
		"	_klen = " << CL() << "[" << vCS() << "];\n"
		"	_keys = " << CAST(keysType) << " ("<< CO() << "[" << vCS() << "]*2);\n"
		"	if ( _klen > 0 ) {\n"
		"		" << signedKeysType << " _lower = _keys;\n"
		"		" << signedKeysType << " _mid;\n"
		"		" << signedKeysType << " _upper = " << CAST(signedKeysType) << 
			" (_keys + (_klen<<1) - 2);\n"
		"		while (true) {\n"
		"			if ( _upper < _lower )\n"
		"				break;\n"
		"\n"
		"			_mid = " << CAST(signedKeysType) << 
			" (_lower + (((_upper-_lower) >> 1) & ~1));\n"
		"			if ( " << GET_WIDE_KEY() << " < " << CK() << "[_mid] )\n"
		"				_upper = " << CAST(signedKeysType) << " (_mid - 2);\n"
		"			else if ( " << GET_WIDE_KEY() << " > " << CK() << "[_mid+1] )\n"
		"				_lower = " << CAST(signedKeysType) << " (_mid + 2);\n"
		"			else {\n"
		"				switch ( " << C() << "[" << CO() << "[" << vCS() << "]"
							" + ((_mid - _keys)>>1)] ) {\n";

	for ( CondSpaceList::Iter csi = condSpaceList; csi.lte(); csi++ ) {
		GenCondSpace *condSpace = csi;
		out << "	case " << condSpace->condSpaceId << ": {\n";
		out << TABS(2) << "_widec = " << CAST(WIDE_ALPH_TYPE()) << "(" <<
				KEY(condSpace->baseKey) << " + (" << GET_KEY() << 
				" - " << KEY(keyOps->minKey) << "));\n";

		for ( GenCondSet::Iter csi = condSpace->condSet; csi.lte(); csi++ ) {
			out << TABS(2) << "if ( ";
			CONDITION( out, *csi );
			Size condValOffset = ((1 << csi.pos()) * keyOps->alphSize());
			out << " ) _widec += " << condValOffset << ";\n";
		}

		out << 
			"		break;\n"
			"	}\n";
	}

	SWITCH_DEFAULT();

	out << 
		"				}\n"
		"				break;\n"
		"			}\n"
		"		}\n"
		"	}\n"
		"\n";
}

void OCamlTabCodeGen::writeExec()
{
	testEofUsed = false;
	outLabelUsed = false;
	initVarTypes();

	out <<
		"	begin\n";
//		"	" << klenType << " _klen";

//	if ( redFsm->anyRegCurStateRef() )
//		out << ", _ps";

/*
	out << "	" << transType << " _trans;\n";

	if ( redFsm->anyConditions() )
		out << "	" << WIDE_ALPH_TYPE() << " _widec;\n";

	if ( redFsm->anyToStateActions() || redFsm->anyRegActions() 
			|| redFsm->anyFromStateActions() )
	{
		out << 
			"	int _acts;\n"
			"	int _nacts;\n";
	}

	out <<
		"	" << keysType << " _keys;\n"
		"\n";
//		"	" << PTR_CONST() << WIDE_ALPH_TYPE() << POINTER() << "_keys;\n"
*/

  out <<
    "	let state = { keys = 0; trans = 0; acts = 0; nacts = 0; } in\n"
    "	let rec do_start () =\n";
	if ( !noEnd ) {
		testEofUsed = true;
		out << 
			"	if " << P() << " = " << PE() << " then\n"
			"		do_test_eof ()\n"
      "\telse\n";
	}

	if ( redFsm->errState != 0 ) {
		outLabelUsed = true;
		out << 
			"	if " << vCS() << " = " << redFsm->errState->id << " then\n"
			"		do_out ()\n"
      "\telse\n";
	}
  out << "\tdo_resume ()\n";

	out << "and do_resume () =\n";

	if ( redFsm->anyFromStateActions() ) {
		out <<
			"	state.acts <- " << AT( FSA(), vCS() ) << ";\n"
			"	state.nacts <- " << AT( A(), POST_INCR("state.acts") ) << ";\n"
			"	while " << POST_DECR("state.nacts") << " > 0 do\n"
			"		begin match " << AT( A(), POST_INCR("state.acts") ) << " with\n";
			FROM_STATE_ACTION_SWITCH();
			SWITCH_DEFAULT() <<
			"		end\n"
			"	done;\n"
			"\n";
	}

	if ( redFsm->anyConditions() )
		COND_TRANSLATE();

  out << "\tbegin try\n";
	LOCATE_TRANS();
  out << "\twith Goto_match -> () end;\n";

  out << 
    "\tdo_match ()\n";

	out << "and do_match () =\n";

	if ( useIndicies )
		out << "	state.trans <- " << CAST(transType) << AT( I(), "state.trans" ) << ";\n";

  out << "\tdo_eof_trans ()\n";
	
//	if ( redFsm->anyEofTrans() )
  out << "and do_eof_trans () =\n";

	if ( redFsm->anyRegCurStateRef() )
		out << "	let ps = " << vCS() << " in\n";

	out <<
		"	" << vCS() << " <- " << AT( TT() ,"state.trans" ) << ";\n"
		"\n";

	if ( redFsm->anyRegActions() ) {
		out <<
			"\tbegin try\n"
      "	match " << AT( TA(), "state.trans" ) << " with\n"
			"\t| 0 -> raise Goto_again\n"
      "\t| _ ->\n"
			"	state.acts <- " << AT( TA(), "state.trans" ) << ";\n"
			"	state.nacts <- " << AT( A(), POST_INCR("state.acts") ) << ";\n"
			"	while " << POST_DECR("state.nacts") << " > 0 do\n"
			"		begin match " << AT( A(), POST_INCR("state.acts") ) << " with\n";
			ACTION_SWITCH();
			SWITCH_DEFAULT() <<
			"		end;\n"
			"	done\n"
      "\twith Goto_again -> () end;\n";
	}
  out << "\tdo_again ()\n";

//	if ( redFsm->anyRegActions() || redFsm->anyActionGotos() || 
//			redFsm->anyActionCalls() || redFsm->anyActionRets() )
  out << "\tand do_again () =\n";

	if ( redFsm->anyToStateActions() ) {
		out <<
			"	state.acts <- " << AT( TSA(), vCS() ) << ";\n"
			"	state.nacts <- " << AT( A(), POST_INCR("state.acts") ) << ";\n"
			"	while " << POST_DECR("state.nacts") << " > 0 do\n"
			"		begin match " << AT( A(), POST_INCR("state.acts") ) << " with\n";
			TO_STATE_ACTION_SWITCH();
			SWITCH_DEFAULT() <<
			"		end\n"
			"	done;\n"
			"\n";
	}

	if ( redFsm->errState != 0 ) {
		outLabelUsed = true;
		out << 
			"	match " << vCS() << " with\n"
      "\t| " << redFsm->errState->id << " -> do_out ()\n"
      "\t| _ ->\n";
	}

  out << "\t" << P() << " <- " << P() << " + 1;\n";

	if ( !noEnd ) {
		out << 
			"	if " << P() << " <> " << PE() << " then\n"
			"		do_resume ()\n"
      "\telse do_test_eof ()\n";
	}
	else {
		out << 
			"	do_resume ()\n";
	}

//	if ( testEofUsed )
	out << "and do_test_eof () =\n";
	
	if ( redFsm->anyEofTrans() || redFsm->anyEofActions() ) {
		out << 
			"	if " << P() << " = " << vEOF() << " then\n"
			"	begin try\n";

		if ( redFsm->anyEofTrans() ) {
			out <<
				"	if " << AT( ET(), vCS() ) << " > 0 then\n"
				"	begin\n"
        "   state.trans <- " << CAST(transType) << "(" << AT( ET(), vCS() ) << " - 1);\n"
				"		raise Goto_eof_trans;\n"
				"	end;\n";
		}

		if ( redFsm->anyEofActions() ) {
			out <<
				"	let __acts = ref " << AT( EA(), vCS() ) << " in\n"
				"	let __nacts = ref " << AT( A(), "!__acts" ) << " in\n"
        " incr __acts;\n"
				"	while !__nacts > 0 do\n"
        "   decr __nacts;\n"
				"		begin match " << AT( A(), POST_INCR("__acts.contents") ) << " with\n";
				EOF_ACTION_SWITCH();
				SWITCH_DEFAULT() <<
				"		end;\n"
				"	done\n";
		}

		out << 
			"	with Goto_again -> do_again ()\n"
			"	| Goto_eof_trans -> do_eof_trans () end\n"
			"\n";
	}
  else
  {
    out << "\t()\n";
  }

	if ( outLabelUsed )
		out << "	and do_out () = ()\n";

  out << "\tin do_start ()\n";
	out << "	end;\n";
}

void OCamlTabCodeGen::initVarTypes()
{
	int klenMax = MAX(MAX(redFsm->maxCondLen, redFsm->maxRangeLen),
				redFsm->maxSingleLen);
	int keysMax = MAX(MAX(redFsm->maxKeyOffset, klenMax),
				redFsm->maxCondOffset);
	int transMax = MAX(MAX(redFsm->maxIndex+1, redFsm->maxIndexOffset), keysMax);
	transMax = MAX(transMax, klenMax);
	transType = ARRAY_TYPE(transMax);
	klenType = ARRAY_TYPE(klenMax);
	keysType = ARRAY_TYPE(keysMax);
	signedKeysType = ARRAY_TYPE(keysMax, true);
}
