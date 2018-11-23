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
#include "cstable.h"
#include "redfsm.h"
#include "gendata.h"

/* Determine if we should use indicies or not. */
void CSharpTabCodeGen::calcIndexSize()
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

std::ostream &CSharpTabCodeGen::TO_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->toStateAction != 0 )
		act = state->toStateAction->location+1;
	out << act;
	return out;
}

std::ostream &CSharpTabCodeGen::FROM_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->fromStateAction != 0 )
		act = state->fromStateAction->location+1;
	out << act;
	return out;
}

std::ostream &CSharpTabCodeGen::EOF_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->eofAction != 0 )
		act = state->eofAction->location+1;
	out << act;
	return out;
}


std::ostream &CSharpTabCodeGen::TRANS_ACTION( RedTransAp *trans )
{
	/* If there are actions, emit them. Otherwise emit zero. */
	int act = 0;
	if ( trans->action != 0 )
		act = trans->action->location+1;
	out << act;
	return out;
}

std::ostream &CSharpTabCodeGen::TO_STATE_ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numToStateRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << "\tcase " << act->actionId << ":\n";
			ACTION( out, act, 0, false );
			out << "\tbreak;\n";
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &CSharpTabCodeGen::FROM_STATE_ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numFromStateRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << "\tcase " << act->actionId << ":\n";
			ACTION( out, act, 0, false );
			out << "\tbreak;\n";
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &CSharpTabCodeGen::EOF_ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numEofRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << "\tcase " << act->actionId << ":\n";
			ACTION( out, act, 0, true );
			out << "\tbreak;\n";
		}
	}

	genLineDirective( out );
	return out;
}


std::ostream &CSharpTabCodeGen::ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numTransRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << "\tcase " << act->actionId << ":\n";
			ACTION( out, act, 0, false );
			out << "\tbreak;\n";
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &CSharpTabCodeGen::COND_OFFSETS()
{
	out << "\t";
	int totalStateNum = 0, curKeyOffset = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write the key offset. */
		out << curKeyOffset;
		if ( !st.last() ) {
			out << ", ";
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}

		/* Move the key offset ahead. */
		curKeyOffset += st->stateCondList.length();
	}
	out << "\n";
	return out;
}

std::ostream &CSharpTabCodeGen::KEY_OFFSETS()
{
	out << "\t";
	int totalStateNum = 0, curKeyOffset = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write the key offset. */
		out << curKeyOffset;
		if ( !st.last() ) {
			out << ", ";
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}

		/* Move the key offset ahead. */
		curKeyOffset += st->outSingle.length() + st->outRange.length()*2;
	}
	out << "\n";
	return out;
}


std::ostream &CSharpTabCodeGen::INDEX_OFFSETS()
{
	out << "\t";
	int totalStateNum = 0, curIndOffset = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write the index offset. */
		out << curIndOffset;
		if ( !st.last() ) {
			out << ", ";
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

std::ostream &CSharpTabCodeGen::COND_LENS()
{
	out << "\t";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write singles length. */
		out << st->stateCondList.length();
		if ( !st.last() ) {
			out << ", ";
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}


std::ostream &CSharpTabCodeGen::SINGLE_LENS()
{
	out << "\t";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write singles length. */
		out << st->outSingle.length();
		if ( !st.last() ) {
			out << ", ";
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}

std::ostream &CSharpTabCodeGen::RANGE_LENS()
{
	out << "\t";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Emit length of range index. */
		out << st->outRange.length();
		if ( !st.last() ) {
			out << ", ";
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}

std::ostream &CSharpTabCodeGen::TO_STATE_ACTIONS()
{
	out << "\t";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write any eof action. */
		TO_STATE_ACTION(st);
		if ( !st.last() ) {
			out << ", ";
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}

std::ostream &CSharpTabCodeGen::FROM_STATE_ACTIONS()
{
	out << "\t";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write any eof action. */
		FROM_STATE_ACTION(st);
		if ( !st.last() ) {
			out << ", ";
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}

std::ostream &CSharpTabCodeGen::EOF_ACTIONS()
{
	out << "\t";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write any eof action. */
		EOF_ACTION(st);
		if ( !st.last() ) {
			out << ", ";
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}

std::ostream &CSharpTabCodeGen::EOF_TRANS()
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
			out << ", ";
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}


std::ostream &CSharpTabCodeGen::COND_KEYS()
{
	out << '\t';
	int totalTrans = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Loop the state's transitions. */
		for ( GenStateCondList::Iter sc = st->stateCondList; sc.lte(); sc++ ) {
			/* Lower key. */
			out << ALPHA_KEY( sc->lowKey ) << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";

			/* Upper key. */
			out << ALPHA_KEY( sc->highKey ) << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}
	}

	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	if ( keyOps->alphType->isChar )
		out << "(char) " << 0 << "\n";
	else
		out << 0 << "\n";

	return out;
}

std::ostream &CSharpTabCodeGen::COND_SPACES()
{
	out << '\t';
	int totalTrans = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Loop the state's transitions. */
		for ( GenStateCondList::Iter sc = st->stateCondList; sc.lte(); sc++ ) {
			/* Cond Space id. */
			out << sc->condSpace->condSpaceId << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}
	}

	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	out << 0 << "\n";
	return out;
}

std::ostream &CSharpTabCodeGen::KEYS()
{
	out << '\t';
	int totalTrans = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Loop the singles. */
		for ( RedTransList::Iter stel = st->outSingle; stel.lte(); stel++ ) {
			out << ALPHA_KEY( stel->lowKey ) << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}

		/* Loop the state's transitions. */
		for ( RedTransList::Iter rtel = st->outRange; rtel.lte(); rtel++ ) {
			/* Lower key. */
			out << ALPHA_KEY( rtel->lowKey ) << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";

			/* Upper key. */
			out << ALPHA_KEY( rtel->highKey ) << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}
	}

	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	if ( keyOps->alphType->isChar )
		out << "(char) " << 0 << "\n";
	else
		out << 0 << "\n";

	return out;
}

std::ostream &CSharpTabCodeGen::INDICIES()
{
	int totalTrans = 0;
	out << '\t';
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Walk the singles. */
		for ( RedTransList::Iter stel = st->outSingle; stel.lte(); stel++ ) {
			out << stel->value->id << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}

		/* Walk the ranges. */
		for ( RedTransList::Iter rtel = st->outRange; rtel.lte(); rtel++ ) {
			out << rtel->value->id << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}

		/* The state's default index goes next. */
		if ( st->defTrans != 0 ) {
			out << st->defTrans->id << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}
	}

	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	out << 0 << "\n";
	return out;
}

std::ostream &CSharpTabCodeGen::TRANS_TARGS()
{
	int totalTrans = 0;
	out << '\t';
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Walk the singles. */
		for ( RedTransList::Iter stel = st->outSingle; stel.lte(); stel++ ) {
			RedTransAp *trans = stel->value;
			out << trans->targ->id << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}

		/* Walk the ranges. */
		for ( RedTransList::Iter rtel = st->outRange; rtel.lte(); rtel++ ) {
			RedTransAp *trans = rtel->value;
			out << trans->targ->id << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}

		/* The state's default target state. */
		if ( st->defTrans != 0 ) {
			RedTransAp *trans = st->defTrans;
			out << trans->targ->id << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}
	}

	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st->eofTrans != 0 ) {
			RedTransAp *trans = st->eofTrans;
			trans->pos = totalTrans;
			out << trans->targ->id << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}
	}


	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	out << 0 << "\n";
	return out;
}


std::ostream &CSharpTabCodeGen::TRANS_ACTIONS()
{
	int totalTrans = 0;
	out << '\t';
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Walk the singles. */
		for ( RedTransList::Iter stel = st->outSingle; stel.lte(); stel++ ) {
			RedTransAp *trans = stel->value;
			TRANS_ACTION( trans ) << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}

		/* Walk the ranges. */
		for ( RedTransList::Iter rtel = st->outRange; rtel.lte(); rtel++ ) {
			RedTransAp *trans = rtel->value;
			TRANS_ACTION( trans ) << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}

		/* The state's default index goes next. */
		if ( st->defTrans != 0 ) {
			RedTransAp *trans = st->defTrans;
			TRANS_ACTION( trans ) << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}
	}

	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st->eofTrans != 0 ) {
			RedTransAp *trans = st->eofTrans;
			TRANS_ACTION( trans ) << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << "\n\t";
		}
	}

	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	out << 0 << "\n";
	return out;
}

std::ostream &CSharpTabCodeGen::TRANS_TARGS_WI()
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
			out << ", ";
			if ( ++totalStates % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	delete[] transPtrs;
	return out;
}


std::ostream &CSharpTabCodeGen::TRANS_ACTIONS_WI()
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
			out << ", ";
			if ( ++totalAct % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	delete[] transPtrs;
	return out;
}

void CSharpTabCodeGen::GOTO( ostream &ret, int gotoDest, bool inFinish )
{
	ret << "{" << vCS() << " = " << gotoDest << ";";

	ret << CTRL_FLOW() << "goto _again;";

	ret << "}";
}

void CSharpTabCodeGen::GOTO_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish )
{
	ret << "{" << vCS() << " = (";
	INLINE_LIST( ret, ilItem->children, 0, inFinish );
	ret << ");";
	
	ret << CTRL_FLOW() << "goto _again;";

	ret << "}";
}

void CSharpTabCodeGen::CURS( ostream &ret, bool inFinish )
{
	ret << "(_ps)";
}

void CSharpTabCodeGen::TARGS( ostream &ret, bool inFinish, int targState )
{
	ret << "(" << vCS() << ")";
}

void CSharpTabCodeGen::NEXT( ostream &ret, int nextDest, bool inFinish )
{
	ret << vCS() << " = " << nextDest << ";";
}

void CSharpTabCodeGen::NEXT_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish )
{
	ret << vCS() << " = (";
	INLINE_LIST( ret, ilItem->children, 0, inFinish );
	ret << ");";
}

void CSharpTabCodeGen::CALL( ostream &ret, int callDest, int targState, bool inFinish )
{
	if ( prePushExpr != 0 ) {
		ret << "{";
		INLINE_LIST( ret, prePushExpr, 0, false );
	}

	ret << "{";

	ret << STACK() << "[" << TOP() << "++] = " << vCS() << "; " << vCS() << " = " << callDest << ";";

	ret << CTRL_FLOW() << "goto _again;";

	ret << "}";

	if ( prePushExpr != 0 )
		ret << "}";
}

void CSharpTabCodeGen::CALL_EXPR( ostream &ret, GenInlineItem *ilItem, int targState, bool inFinish )
{
	if ( prePushExpr != 0 ) {
		ret << "{";
		INLINE_LIST( ret, prePushExpr, 0, false );
	}

	ret << "{";

	ret << STACK() << "[" << TOP() << "++] = " << vCS() << "; " << vCS() << " = (";
	INLINE_LIST( ret, ilItem->children, targState, inFinish );
	ret << ");";

	ret << CTRL_FLOW() << "goto _again;";

	ret << "}";

	if ( prePushExpr != 0 )
		ret << "}";
}

void CSharpTabCodeGen::RET( ostream &ret, bool inFinish )
{
	ret << "{";

	ret << vCS() << " = " << STACK() << "[--" << TOP() << "]; ";

	if ( postPopExpr != 0 ) {
		ret << "{";
		INLINE_LIST( ret, postPopExpr, 0, false );
		ret << "}";
	}

	ret << CTRL_FLOW() <<  "goto _again;";

	ret << "}";
}

void CSharpTabCodeGen::BREAK( ostream &ret, int targState )
{
	outLabelUsed = true;
	ret << "{" << P() << "++; " << CTRL_FLOW() << "goto _out; }";
}

void CSharpTabCodeGen::writeData()
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
}

void CSharpTabCodeGen::LOCATE_TRANS()
{
	out <<
		"	_keys = " << KO() + "[" + vCS() + "]" << ";\n"
		"	_trans = " << CAST(transType) << IO() << "[" << vCS() << "];\n"
		"\n"
		"	_klen = " << SL() << "[" << vCS() << "];\n"
		"	if ( _klen > 0 ) {\n"
		"		" << signedKeysType << " _lower = _keys;\n"
		"		" << signedKeysType << " _mid;\n"
		"		" << signedKeysType << " _upper = " << CAST(signedKeysType) << 
			" (_keys + _klen - 1);\n"
		"		while (true) {\n"
		"			if ( _upper < _lower )\n"
		"				break;\n"
		"\n"
		"			_mid = " << CAST(signedKeysType) << 
			" (_lower + ((_upper-_lower) >> 1));\n"
		"			if ( " << GET_WIDE_KEY() << " < " << K() << "[_mid] )\n"
		"				_upper = " << CAST(signedKeysType) << " (_mid - 1);\n"
		"			else if ( " << GET_WIDE_KEY() << " > " << K() << "[_mid] )\n"
		"				_lower = " << CAST(signedKeysType) << " (_mid + 1);\n"
		"			else {\n"
		"				_trans += " << CAST(transType) << " (_mid - _keys);\n"
		"				goto _match;\n"
		"			}\n"
		"		}\n"
		"		_keys += " << CAST(keysType) << " _klen;\n"
		"		_trans += " << CAST(transType) << " _klen;\n"
		"	}\n"
		"\n"
		"	_klen = " << RL() << "[" << vCS() << "];\n"
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
		"			if ( " << GET_WIDE_KEY() << " < " << K() << "[_mid] )\n"
		"				_upper = " << CAST(signedKeysType) << " (_mid - 2);\n"
		"			else if ( " << GET_WIDE_KEY() << " > " << K() << "[_mid+1] )\n"
		"				_lower = " << CAST(signedKeysType) << " (_mid + 2);\n"
		"			else {\n"
		"				_trans += " << CAST(transType) << "((_mid - _keys)>>1);\n"
		"				goto _match;\n"
		"			}\n"
		"		}\n"
		"		_trans += " << CAST(transType) << " _klen;\n"
		"	}\n"
		"\n";
}

void CSharpTabCodeGen::COND_TRANSLATE()
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

void CSharpTabCodeGen::writeExec()
{
	testEofUsed = false;
	outLabelUsed = false;
	initVarTypes();

	out <<
		"	{\n"
		"	" << klenType << " _klen";

	if ( redFsm->anyRegCurStateRef() )
		out << ", _ps";

	out << 
		";\n"
		"	" << transType << " _trans;\n";

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

	if ( !noEnd ) {
		testEofUsed = true;
		out << 
			"	if ( " << P() << " == " << PE() << " )\n"
			"		goto _test_eof;\n";
	}

	if ( redFsm->errState != 0 ) {
		outLabelUsed = true;
		out << 
			"	if ( " << vCS() << " == " << redFsm->errState->id << " )\n"
			"		goto _out;\n";
	}

	out << "_resume:\n";

	if ( redFsm->anyFromStateActions() ) {
		out <<
			"	_acts = " << FSA() << "[" + vCS() + "]" << ";\n"
			"	_nacts = " << A() << "[_acts++];\n"
			"	while ( _nacts-- > 0 ) {\n"
			"		switch ( " << A() << "[_acts++] ) {\n";
			FROM_STATE_ACTION_SWITCH();
			SWITCH_DEFAULT() <<
			"		}\n"
			"	}\n"
			"\n";
	}

	if ( redFsm->anyConditions() )
		COND_TRANSLATE();

	LOCATE_TRANS();

	out << "_match:\n";

	if ( useIndicies )
		out << "	_trans = " << CAST(transType) << I() << "[_trans];\n";
	
	if ( redFsm->anyEofTrans() )
		out << "_eof_trans:\n";

	if ( redFsm->anyRegCurStateRef() )
		out << "	_ps = " << vCS() << ";\n";

	out <<
		"	" << vCS() << " = " << TT() << "[_trans];\n"
		"\n";

	if ( redFsm->anyRegActions() ) {
		out <<
			"	if ( " << TA() << "[_trans] == 0 )\n"
			"		goto _again;\n"
			"\n"
			"	_acts = " << TA() << "[_trans]" << ";\n"
			"	_nacts = " << A() << "[_acts++];\n"
			"	while ( _nacts-- > 0 )\n	{\n"
			"		switch ( " << A() << "[_acts++] )\n		{\n";
			ACTION_SWITCH();
			SWITCH_DEFAULT() <<
			"		}\n"
			"	}\n"
			"\n";
	}

	if ( redFsm->anyRegActions() || redFsm->anyActionGotos() || 
			redFsm->anyActionCalls() || redFsm->anyActionRets() )
		out << "_again:\n";

	if ( redFsm->anyToStateActions() ) {
		out <<
			"	_acts = " << TSA() << "[" << vCS() << "]" << ";\n"
			"	_nacts = " << A() << "[_acts++];\n"
			"	while ( _nacts-- > 0 ) {\n"
			"		switch ( " << A() << "[_acts++] ) {\n";
			TO_STATE_ACTION_SWITCH();
			SWITCH_DEFAULT() <<
			"		}\n"
			"	}\n"
			"\n";
	}

	if ( redFsm->errState != 0 ) {
		outLabelUsed = true;
		out << 
			"	if ( " << vCS() << " == " << redFsm->errState->id << " )\n"
			"		goto _out;\n";
	}

	if ( !noEnd ) {
		out << 
			"	if ( ++" << P() << " != " << PE() << " )\n"
			"		goto _resume;\n";
	}
	else {
		out << 
			"	" << P() << " += 1;\n"
			"	goto _resume;\n";
	}
	
	if ( testEofUsed )
		out << "	_test_eof: {}\n";
	
	if ( redFsm->anyEofTrans() || redFsm->anyEofActions() ) {
		out << 
			"	if ( " << P() << " == " << vEOF() << " )\n"
			"	{\n";

		if ( redFsm->anyEofTrans() ) {
			out <<
				"	if ( " << ET() << "[" << vCS() << "] > 0 ) {\n"
				"		_trans = " << CAST(transType) << " (" << ET() <<
					"[" << vCS() << "] - 1);\n"
				"		goto _eof_trans;\n"
				"	}\n";
		}

		if ( redFsm->anyEofActions() ) {
			out <<
				"	int __acts = " << 
						EA() << "[" << vCS() << "]" << ";\n"
				"	int __nacts = " << 
				A() << "[__acts++];\n"
				"	while ( __nacts-- > 0 ) {\n"
				"		switch ( " << A() << "[__acts++] ) {\n";
				EOF_ACTION_SWITCH();
				SWITCH_DEFAULT() <<
				"		}\n"
				"	}\n";
		}
		
		out << 
			"	}\n"
			"\n";
	}

	if ( outLabelUsed )
		out << "	_out: {}\n";

	out << "	}\n";
}

void CSharpTabCodeGen::initVarTypes()
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
