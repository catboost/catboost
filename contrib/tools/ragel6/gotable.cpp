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

#include <sstream>
#include "ragel.h"
#include "gotable.h"
#include "redfsm.h"
#include "gendata.h"

using std::endl;

/* Determine if we should use indicies or not. */
void GoTabCodeGen::calcIndexSize()
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

std::ostream &GoTabCodeGen::TO_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->toStateAction != 0 )
		act = state->toStateAction->location+1;
	out << act;
	return out;
}

std::ostream &GoTabCodeGen::FROM_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->fromStateAction != 0 )
		act = state->fromStateAction->location+1;
	out << act;
	return out;
}

std::ostream &GoTabCodeGen::EOF_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->eofAction != 0 )
		act = state->eofAction->location+1;
	out << act;
	return out;
}


std::ostream &GoTabCodeGen::TRANS_ACTION( RedTransAp *trans )
{
	/* If there are actions, emit them. Otherwise emit zero. */
	int act = 0;
	if ( trans->action != 0 )
		act = trans->action->location+1;
	out << act;
	return out;
}

std::ostream &GoTabCodeGen::TO_STATE_ACTION_SWITCH( int level )
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numToStateRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << TABS(level) << "case " << act->actionId << ":" << endl;
			ACTION( out, act, 0, false, false );
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &GoTabCodeGen::FROM_STATE_ACTION_SWITCH( int level )
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numFromStateRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << TABS(level) << "case " << act->actionId << ":" << endl;
			ACTION( out, act, 0, false, false );
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &GoTabCodeGen::EOF_ACTION_SWITCH( int level )
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numEofRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << TABS(level) << "case " << act->actionId << ":" << endl;
			ACTION( out, act, 0, true, false );
		}
	}

	genLineDirective(out);
	return out;
}


std::ostream &GoTabCodeGen::ACTION_SWITCH( int level )
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numTransRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << TABS(level) << "case " << act->actionId << ":" << endl;
			ACTION( out, act, 0, false, false );
		}
	}

	genLineDirective(out);
	return out;
}

std::ostream &GoTabCodeGen::COND_OFFSETS()
{
	out << "	";
	int totalStateNum = 0, curKeyOffset = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write the key offset. */
		out << curKeyOffset;
		out << ", ";
		if ( !st.last() ) {
			if ( ++totalStateNum % IALL == 0 )
				out << endl << "	";
		}

		/* Move the key offset ahead. */
		curKeyOffset += st->stateCondList.length();
	}
	out << endl;
	return out;
}

std::ostream &GoTabCodeGen::KEY_OFFSETS()
{
	out << "	";
	int totalStateNum = 0, curKeyOffset = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write the key offset. */
		out << curKeyOffset;
		out << ", ";
		if ( !st.last() ) {
			if ( ++totalStateNum % IALL == 0 )
				out << endl << "	";
		}

		/* Move the key offset ahead. */
		curKeyOffset += st->outSingle.length() + st->outRange.length()*2;
	}
	out << endl;
	return out;
}


std::ostream &GoTabCodeGen::INDEX_OFFSETS()
{
	out << "	";
	int totalStateNum = 0, curIndOffset = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write the index offset. */
		out << curIndOffset;
		out << ", ";
		if ( !st.last() ) {
			if ( ++totalStateNum % IALL == 0 )
				out << endl << "	";
		}

		/* Move the index offset ahead. */
		curIndOffset += st->outSingle.length() + st->outRange.length();
		if ( st->defTrans != 0 )
			curIndOffset += 1;
	}
	out << endl;
	return out;
}

std::ostream &GoTabCodeGen::COND_LENS()
{
	out << "	";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write singles length. */
		out << st->stateCondList.length();
		out << ", ";
		if ( !st.last() ) {
			if ( ++totalStateNum % IALL == 0 )
				out << endl << "	";
		}
	}
	out << endl;
	return out;
}


std::ostream &GoTabCodeGen::SINGLE_LENS()
{
	out << "	";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write singles length. */
		out << st->outSingle.length();
		out << ", ";
		if ( !st.last() ) {
			if ( ++totalStateNum % IALL == 0 )
				out << endl << "	";
		}
	}
	out << endl;
	return out;
}

std::ostream &GoTabCodeGen::RANGE_LENS()
{
	out << "	";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Emit length of range index. */
		out << st->outRange.length();
		out << ", ";
		if ( !st.last() ) {
			if ( ++totalStateNum % IALL == 0 )
				out << endl << "	";
		}
	}
	out << endl;
	return out;
}

std::ostream &GoTabCodeGen::TO_STATE_ACTIONS()
{
	out << "	";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write any eof action. */
		TO_STATE_ACTION(st);
		out << ", ";
		if ( !st.last() ) {
			if ( ++totalStateNum % IALL == 0 )
				out << endl << "	";
		}
	}
	out << endl;
	return out;
}

std::ostream &GoTabCodeGen::FROM_STATE_ACTIONS()
{
	out << "	";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write any eof action. */
		FROM_STATE_ACTION(st);
		out << ", ";
		if ( !st.last() ) {
			if ( ++totalStateNum % IALL == 0 )
				out << endl << "	";
		}
	}
	out << endl;
	return out;
}

std::ostream &GoTabCodeGen::EOF_ACTIONS()
{
	out << "	";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write any eof action. */
		EOF_ACTION(st);
		out << ", ";
		if ( !st.last() ) {
			if ( ++totalStateNum % IALL == 0 )
				out << endl << "	";
		}
	}
	out << endl;
	return out;
}

std::ostream &GoTabCodeGen::EOF_TRANS()
{
	out << "	";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write any eof action. */
		long trans = 0;
		if ( st->eofTrans != 0 ) {
			assert( st->eofTrans->pos >= 0 );
			trans = st->eofTrans->pos+1;
		}
		out << trans;

		out << ", ";
		if ( !st.last() ) {
			if ( ++totalStateNum % IALL == 0 )
				out << endl << "	";
		}
	}
	out << endl;
	return out;
}


std::ostream &GoTabCodeGen::COND_KEYS()
{
	out << "	";
	int totalTrans = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Loop the state's transitions. */
		for ( GenStateCondList::Iter sc = st->stateCondList; sc.lte(); sc++ ) {
			/* Lower key. */
			out << KEY( sc->lowKey ) << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << endl << "	";

			/* Upper key. */
			out << KEY( sc->highKey ) << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << endl << "	";
		}
	}

	out << endl;
	return out;
}

std::ostream &GoTabCodeGen::COND_SPACES()
{
	out << "	";
	int totalTrans = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Loop the state's transitions. */
		for ( GenStateCondList::Iter sc = st->stateCondList; sc.lte(); sc++ ) {
			/* Cond Space id. */
			out << sc->condSpace->condSpaceId << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << endl << "	";
		}
	}

	out << endl;
	return out;
}

std::ostream &GoTabCodeGen::KEYS()
{
	out << "	";
	int totalTrans = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Loop the singles. */
		for ( RedTransList::Iter stel = st->outSingle; stel.lte(); stel++ ) {
			out << KEY( stel->lowKey ) << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << endl << "	";
		}

		/* Loop the state's transitions. */
		for ( RedTransList::Iter rtel = st->outRange; rtel.lte(); rtel++ ) {
			/* Lower key. */
			out << KEY( rtel->lowKey ) << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << endl << "	";

			/* Upper key. */
			out << KEY( rtel->highKey ) << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << endl << "	";
		}
	}

	out << endl;
	return out;
}

std::ostream &GoTabCodeGen::INDICIES()
{
	out << "	";
	int totalTrans = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Walk the singles. */
		for ( RedTransList::Iter stel = st->outSingle; stel.lte(); stel++ ) {
			out << stel->value->id << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << endl << "	";
		}

		/* Walk the ranges. */
		for ( RedTransList::Iter rtel = st->outRange; rtel.lte(); rtel++ ) {
			out << rtel->value->id << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << endl << "	";
		}

		/* The state's default index goes next. */
		if ( st->defTrans != 0 ) {
			out << st->defTrans->id << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << endl << "	";
		}
	}

	out << endl;
	return out;
}

std::ostream &GoTabCodeGen::TRANS_TARGS()
{
	out << "	";
	int totalTrans = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Walk the singles. */
		for ( RedTransList::Iter stel = st->outSingle; stel.lte(); stel++ ) {
			RedTransAp *trans = stel->value;
			out << trans->targ->id << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << endl << "	";
		}

		/* Walk the ranges. */
		for ( RedTransList::Iter rtel = st->outRange; rtel.lte(); rtel++ ) {
			RedTransAp *trans = rtel->value;
			out << trans->targ->id << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << endl << "	";
		}

		/* The state's default target state. */
		if ( st->defTrans != 0 ) {
			RedTransAp *trans = st->defTrans;
			out << trans->targ->id << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << endl << "	";
		}
	}

	/* Add any eof transitions that have not yet been written out above. */
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st->eofTrans != 0 ) {
			RedTransAp *trans = st->eofTrans;
			trans->pos = totalTrans;
			out << trans->targ->id << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << endl << "	";
		}
	}


	out << endl;
	return out;
}


std::ostream &GoTabCodeGen::TRANS_ACTIONS()
{
	out << "	";
	int totalTrans = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Walk the singles. */
		for ( RedTransList::Iter stel = st->outSingle; stel.lte(); stel++ ) {
			RedTransAp *trans = stel->value;
			TRANS_ACTION( trans ) << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << endl << "	";
		}

		/* Walk the ranges. */
		for ( RedTransList::Iter rtel = st->outRange; rtel.lte(); rtel++ ) {
			RedTransAp *trans = rtel->value;
			TRANS_ACTION( trans ) << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << endl << "	";
		}

		/* The state's default index goes next. */
		if ( st->defTrans != 0 ) {
			RedTransAp *trans = st->defTrans;
			TRANS_ACTION( trans ) << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << endl << "	";
		}
	}

	/* Add any eof transitions that have not yet been written out above. */
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st->eofTrans != 0 ) {
			RedTransAp *trans = st->eofTrans;
			TRANS_ACTION( trans ) << ", ";
			if ( ++totalTrans % IALL == 0 )
				out << endl << "	";
		}
	}

	out << endl;
	return out;
}

std::ostream &GoTabCodeGen::TRANS_TARGS_WI()
{
	/* Transitions must be written ordered by their id. */
	RedTransAp **transPtrs = new RedTransAp*[redFsm->transSet.length()];
	for ( TransApSet::Iter trans = redFsm->transSet; trans.lte(); trans++ )
		transPtrs[trans->id] = trans;

	/* Keep a count of the num of items in the array written. */
	out << "	";
	int totalStates = 0;
	for ( int t = 0; t < redFsm->transSet.length(); t++ ) {
		/* Record the position, need this for eofTrans. */
		RedTransAp *trans = transPtrs[t];
		trans->pos = t;

		/* Write out the target state. */
		out << trans->targ->id << ", ";
		if ( t < redFsm->transSet.length()-1 ) {
			if ( ++totalStates % IALL == 0 )
				out << endl << "	";
		}
	}
	out << endl;
	delete[] transPtrs;
	return out;
}


std::ostream &GoTabCodeGen::TRANS_ACTIONS_WI()
{
	/* Transitions must be written ordered by their id. */
	RedTransAp **transPtrs = new RedTransAp*[redFsm->transSet.length()];
	for ( TransApSet::Iter trans = redFsm->transSet; trans.lte(); trans++ )
		transPtrs[trans->id] = trans;

	/* Keep a count of the num of items in the array written. */
	out << "	";
	int totalAct = 0;
	for ( int t = 0; t < redFsm->transSet.length(); t++ ) {
		/* Write the function for the transition. */
		RedTransAp *trans = transPtrs[t];
		TRANS_ACTION( trans );
		out << ", ";
		if ( t < redFsm->transSet.length()-1 ) {
			if ( ++totalAct % IALL == 0 )
				out << endl << "	";
		}
	}
	out << endl;
	delete[] transPtrs;
	return out;
}

void GoTabCodeGen::writeData()
{
	/* If there are any transtion functions then output the array. If there
	 * are none, don't bother emitting an empty array that won't be used. */
	if ( redFsm->anyActions() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActArrItem), A() );
		ACTIONS_ARRAY();
		CLOSE_ARRAY() << endl;
	}

	if ( redFsm->anyConditions() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxCondOffset), CO() );
		COND_OFFSETS();
		CLOSE_ARRAY() << endl;

		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxCondLen), CL() );
		COND_LENS();
		CLOSE_ARRAY() << endl;

		OPEN_ARRAY( WIDE_ALPH_TYPE(), CK() );
		COND_KEYS();
		CLOSE_ARRAY() << endl;

		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxCondSpaceId), C() );
		COND_SPACES();
		CLOSE_ARRAY() << endl;
	}

	OPEN_ARRAY( ARRAY_TYPE(redFsm->maxKeyOffset), KO() );
	KEY_OFFSETS();
	CLOSE_ARRAY() << endl;

	OPEN_ARRAY( WIDE_ALPH_TYPE(), K() );
	KEYS();
	CLOSE_ARRAY() << endl;

	OPEN_ARRAY( ARRAY_TYPE(redFsm->maxSingleLen), SL() );
	SINGLE_LENS();
	CLOSE_ARRAY() << endl;

	OPEN_ARRAY( ARRAY_TYPE(redFsm->maxRangeLen), RL() );
	RANGE_LENS();
	CLOSE_ARRAY() << endl;

	OPEN_ARRAY( ARRAY_TYPE(redFsm->maxIndexOffset), IO() );
	INDEX_OFFSETS();
	CLOSE_ARRAY() << endl;

	if ( useIndicies ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxIndex), I() );
		INDICIES();
		CLOSE_ARRAY() << endl;

		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxState), TT() );
		TRANS_TARGS_WI();
		CLOSE_ARRAY() << endl;

		if ( redFsm->anyActions() ) {
			OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActionLoc), TA() );
			TRANS_ACTIONS_WI();
			CLOSE_ARRAY() << endl;
		}
	}
	else {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxState), TT() );
		TRANS_TARGS();
		CLOSE_ARRAY() << endl;

		if ( redFsm->anyActions() ) {
			OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActionLoc), TA() );
			TRANS_ACTIONS();
			CLOSE_ARRAY() << endl;
		}
	}

	if ( redFsm->anyToStateActions() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActionLoc), TSA() );
		TO_STATE_ACTIONS();
		CLOSE_ARRAY() << endl;
	}

	if ( redFsm->anyFromStateActions() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActionLoc), FSA() );
		FROM_STATE_ACTIONS();
		CLOSE_ARRAY() << endl;
	}

	if ( redFsm->anyEofActions() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActionLoc), EA() );
		EOF_ACTIONS();
		CLOSE_ARRAY() << endl;
	}

	if ( redFsm->anyEofTrans() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxIndexOffset+1), ET() );
		EOF_TRANS();
		CLOSE_ARRAY() << endl;
	}

	STATE_IDS();
}

void GoTabCodeGen::LOCATE_TRANS()
{
	out <<
		"	_keys = " << CAST(INT(), KO() + "[" + vCS() + "]") << endl <<
		"	_trans = " << CAST(INT(), IO() + "[" + vCS() + "]") << endl <<
		endl <<
		"	_klen = " << CAST(INT(), SL() + "[" + vCS() + "]") << endl <<
		"	if _klen > 0 {" << endl <<
		"		_lower := " << CAST(INT(), "_keys") << endl <<
		"		var _mid " << INT() << endl <<
		"		_upper := " << CAST(INT(), "_keys + _klen - 1") << endl <<
		"		for {" << endl <<
		"			if _upper < _lower {" << endl <<
		"				break" << endl <<
		"			}" << endl <<
		endl <<
		"			_mid = _lower + ((_upper - _lower) >> 1)" << endl <<
		"			switch {" << endl <<
		"			case " << GET_WIDE_KEY() << " < " << K() << "[_mid]" << ":" << endl <<
		"				_upper = _mid - 1" << endl <<
		"			case " << GET_WIDE_KEY() << " > " << K() << "[_mid]" << ":" << endl <<
		"				_lower = _mid + 1" << endl <<
		"			default:" << endl <<
		"				_trans += " << CAST(INT(), "_mid - " + CAST(INT(), "_keys")) << endl <<
		"				goto _match" << endl <<
		"			}" << endl <<
		"		}" << endl <<
		"		_keys += _klen" << endl <<
		"		_trans += _klen" << endl <<
		"	}" << endl <<
		endl <<
		"	_klen = " << CAST(INT(), RL() + "[" + vCS() + "]") << endl <<
		"	if _klen > 0 {" << endl <<
		"		_lower := " << CAST(INT(), "_keys") << endl <<
		"		var _mid " << INT() << endl <<
		"		_upper := " << CAST(INT(), "_keys + (_klen << 1) - 2") << endl <<
		"		for {" << endl <<
		"			if _upper < _lower {" << endl <<
		"				break" << endl <<
		"			}" << endl <<
		endl <<
		"			_mid = _lower + (((_upper - _lower) >> 1) & ^1)" << endl <<
		"			switch {" << endl <<
		"			case " << GET_WIDE_KEY() << " < " << K() << "[_mid]" << ":" << endl <<
		"				_upper = _mid - 2" << endl <<
		"			case " << GET_WIDE_KEY() << " > " << K() << "[_mid + 1]" << ":" << endl <<
		"				_lower = _mid + 2" << endl <<
		"			default:" << endl <<
		"				_trans += " << CAST(INT(), "(_mid - " + CAST(INT(), "_keys") + ") >> 1") << endl <<
		"				goto _match" << endl <<
		"			}" << endl <<
		"		}" << endl <<
		"		_trans += _klen" << endl <<
		"	}" << endl <<
		endl;
}

void GoTabCodeGen::COND_TRANSLATE()
{
	out <<
		"	_widec = " << CAST(WIDE_ALPH_TYPE(), GET_KEY()) << endl <<
		"	_klen = " << CAST(INT(), CL() + "[" + vCS() + "]") << endl <<
		"	_keys = " << CAST(INT(), CO() + "[" + vCS() + "] * 2") << endl <<
		"	if _klen > 0 {" << endl <<
		"		_lower := " << CAST(INT(), "_keys") << endl <<
		"		var _mid " << INT() << endl <<
		"		_upper := " << CAST(INT(), "_keys + (_klen << 1) - 2") << endl <<
		"	COND_LOOP:" << endl <<
		"		for {" << endl <<
		"			if _upper < _lower {" << endl <<
		"				break" << endl <<
		"			}" << endl <<
		endl <<
		"			_mid = _lower + (((_upper - _lower) >> 1) & ^1)" << endl <<
		"			switch {" << endl <<
		"			case " << GET_WIDE_KEY() << " < " << CAST(WIDE_ALPH_TYPE(), CK() + "[_mid]") << ":" << endl <<
		"				_upper = _mid - 2" << endl <<
		"			case " << GET_WIDE_KEY() << " > " << CAST(WIDE_ALPH_TYPE(), CK() + "[_mid + 1]") << ":" << endl <<
		"				_lower = _mid + 2" << endl <<
		"			default:" << endl <<
		"				switch " << C() << "[" << CAST(INT(), CO() + "[" + vCS() + "]") <<
							" + ((_mid - _keys)>>1)] {" << endl;

	for ( CondSpaceList::Iter csi = condSpaceList; csi.lte(); csi++ ) {
		GenCondSpace *condSpace = csi;
		out << TABS(4) << "case " << condSpace->condSpaceId << ":" << endl;
		out << TABS(5) << "_widec = " << KEY(condSpace->baseKey) << " + (" << CAST(WIDE_ALPH_TYPE(), GET_KEY()) <<
					" - " << KEY(keyOps->minKey) << ")" << endl;

		for ( GenCondSet::Iter csi = condSpace->condSet; csi.lte(); csi++ ) {
			out << TABS(5) << "if ";
			CONDITION( out, *csi );
			Size condValOffset = ((1 << csi.pos()) * keyOps->alphSize());
			out << " {" << endl << TABS(6) << "_widec += " << condValOffset << endl << TABS(5) << "}" << endl;
		}
	}

	out <<
		"				}" << endl <<
		"				break COND_LOOP" << endl <<
		"			}" << endl <<
		"		}" << endl <<
		"	}" << endl <<
		endl;
}

void GoTabCodeGen::writeExec()
{
	testEofUsed = false;
	outLabelUsed = false;

	out <<
		"	{" << endl <<
		"	var _klen " << INT() << endl;

	if ( redFsm->anyRegCurStateRef() )
		out << "	var _ps " << INT() << endl;

	out <<
		"	var _trans " << INT() << endl;

	if ( redFsm->anyConditions() )
		out << "	var _widec " << WIDE_ALPH_TYPE() << endl;

	if ( redFsm->anyToStateActions() || redFsm->anyRegActions()
			|| redFsm->anyFromStateActions() )
	{
		out <<
			"	var _acts " << INT() << endl <<
			"	var _nacts " << UINT() << endl;
	}

	out <<
		"	var _keys " << INT() << endl;

	if ( !noEnd ) {
		testEofUsed = true;
		out <<
			"	if " << P() << " == " << PE() << " {" << endl <<
			"		goto _test_eof" << endl <<
			"	}" << endl;
	}

	if ( redFsm->errState != 0 ) {
		outLabelUsed = true;
		out <<
			"	if " << vCS() << " == " << redFsm->errState->id << " {" << endl <<
			"		goto _out" << endl <<
			"	}" << endl;
	}

	out << "_resume:" << endl;

	if ( redFsm->anyFromStateActions() ) {
		out <<
			"	_acts = " << CAST(INT(), FSA() + "[" + vCS() + "]") << endl <<
			"	_nacts = " << CAST(UINT(), A() + "[_acts]") << "; _acts++" << endl <<
			"	for ; _nacts > 0; _nacts-- {" << endl <<
			"		 _acts++" << endl <<
			"		switch " << A() << "[_acts - 1]" << " {" << endl;
			FROM_STATE_ACTION_SWITCH(2);
			out <<
			"		}" << endl <<
			"	}" << endl << endl;
	}

	if ( redFsm->anyConditions() )
		COND_TRANSLATE();

	LOCATE_TRANS();

	out << "_match:" << endl;

	if ( useIndicies )
		out << "	_trans = " << CAST(INT(), I() + "[_trans]") << endl;

	if ( redFsm->anyEofTrans() )
		out << "_eof_trans:" << endl;

	if ( redFsm->anyRegCurStateRef() )
		out << "	_ps = " << vCS() << endl;

	out <<
		"	" << vCS() << " = " << CAST(INT(), TT() + "[_trans]") << endl << endl;

	if ( redFsm->anyRegActions() ) {
		out <<
			"	if " << TA() << "[_trans] == 0 {" <<  endl <<
			"		goto _again" << endl <<
			"	}" << endl <<
			endl <<
			"	_acts = " << CAST(INT(), TA() + "[_trans]") << endl <<
			"	_nacts = " << CAST(UINT(), A() + "[_acts]") << "; _acts++" << endl <<
			"	for ; _nacts > 0; _nacts-- {" << endl <<
			"		_acts++" << endl <<
			"		switch " << A() << "[_acts-1]" << " {" << endl;
			ACTION_SWITCH(2);
			out <<
			"		}" << endl <<
			"	}" << endl << endl;
	}

	if ( redFsm->anyRegActions() || redFsm->anyActionGotos() ||
			redFsm->anyActionCalls() || redFsm->anyActionRets() )
		out << "_again:" << endl;

	if ( redFsm->anyToStateActions() ) {
		out <<
			"	_acts = " << CAST(INT(), TSA() + "[" + vCS() + "]") << endl <<
			"	_nacts = " << CAST(UINT(), A() + "[_acts]") << "; _acts++" << endl <<
			"	for ; _nacts > 0; _nacts-- {" << endl <<
			"		_acts++" << endl <<
			"		switch " << A() << "[_acts-1] {" << endl;
			TO_STATE_ACTION_SWITCH(2);
			out <<
			"		}" << endl <<
			"	}" << endl << endl;
	}

	if ( redFsm->errState != 0 ) {
		outLabelUsed = true;
		out <<
			"	if " << vCS() << " == " << redFsm->errState->id << " {" << endl <<
			"		goto _out" << endl <<
			"	}" << endl;
	}

	if ( !noEnd ) {
		out <<
			"	" << P() << "++" << endl <<
			"	if " << P() << " != " << PE() << " {" << endl <<
			"		goto _resume" << endl <<
			"	}" << endl;
	}
	else {
		out <<
			"	" << P() << "++" << endl <<
			"	goto _resume" << endl;
	}

	if ( testEofUsed )
		out << "	_test_eof: {}" << endl;

	if ( redFsm->anyEofTrans() || redFsm->anyEofActions() ) {
		out <<
			"	if " << P() << " == " << vEOF() << " {" << endl;

		if ( redFsm->anyEofTrans() ) {
			out <<
				"		if " << ET() << "[" << vCS() << "] > 0 {" << endl <<
				"			_trans = " << CAST(INT(), ET() + "[" + vCS() + "] - 1") << endl <<
				"			goto _eof_trans" << endl <<
				"		}" << endl;
		}

		if ( redFsm->anyEofActions() ) {
			out <<
				"		__acts := " << EA() << "[" << vCS() << "]" << endl <<
				"		__nacts := " << CAST(UINT(), A() + "[__acts]") << "; __acts++" << endl <<
				"		for ; __nacts > 0; __nacts-- {" << endl <<
				"			__acts++" << endl <<
				"			switch " << A() << "[__acts-1] {" << endl;
				EOF_ACTION_SWITCH(3);
				out <<
				"			}" << endl <<
				"		}" << endl;
		}

		out <<
			"	}" << endl << endl;
	}

	if ( outLabelUsed )
		out << "	_out: {}" << endl;

	out << "	}" << endl;
}
