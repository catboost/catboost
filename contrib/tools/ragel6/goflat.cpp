/*
 *  Copyright 2004-2006 Adrian Thurston <thurston@complang.org>
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
#include "goflat.h"
#include "redfsm.h"
#include "gendata.h"

using std::endl;

std::ostream &GoFlatCodeGen::TO_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->toStateAction != 0 )
		act = state->toStateAction->location+1;
	out << act;
	return out;
}

std::ostream &GoFlatCodeGen::FROM_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->fromStateAction != 0 )
		act = state->fromStateAction->location+1;
	out << act;
	return out;
}

std::ostream &GoFlatCodeGen::EOF_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->eofAction != 0 )
		act = state->eofAction->location+1;
	out << act;
	return out;
}

std::ostream &GoFlatCodeGen::TRANS_ACTION( RedTransAp *trans )
{
	/* If there are actions, emit them. Otherwise emit zero. */
	int act = 0;
	if ( trans->action != 0 )
		act = trans->action->location+1;
	out << act;
	return out;
}

std::ostream &GoFlatCodeGen::TO_STATE_ACTION_SWITCH( int level )
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numToStateRefs > 0 ) {
			/* Write the case label, the action and the case break */
			out << TABS(level) << "case " << act->actionId << ":" << endl;
			ACTION( out, act, 0, false, false );
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &GoFlatCodeGen::FROM_STATE_ACTION_SWITCH( int level )
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numFromStateRefs > 0 ) {
			/* Write the case label, the action and the case break */
			out << TABS(level) << "case " << act->actionId << ":" << endl;
			ACTION( out, act, 0, false, false );
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &GoFlatCodeGen::EOF_ACTION_SWITCH( int level )
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numEofRefs > 0 ) {
			/* Write the case label, the action and the case break */
			out << TABS(level) << "case " << act->actionId << ":" << endl;
			ACTION( out, act, 0, true, false );
		}
	}

	genLineDirective( out );
	return out;
}


std::ostream &GoFlatCodeGen::ACTION_SWITCH( int level )
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numTransRefs > 0 ) {
			/* Write the case label, the action and the case break */
			out << TABS(level) << "case " << act->actionId << ":" << endl;
			ACTION( out, act, 0, false, false );
		}
	}

	genLineDirective( out );
	return out;
}


std::ostream &GoFlatCodeGen::FLAT_INDEX_OFFSET()
{
	out << "	";
	int totalStateNum = 0, curIndOffset = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write the index offset. */
		out << curIndOffset << ", ";
		if ( !st.last() ) {
			if ( ++totalStateNum % IALL == 0 )
				out << endl << "	";
		}

		/* Move the index offset ahead. */
		if ( st->transList != 0 )
			curIndOffset += keyOps->span( st->lowKey, st->highKey );

		if ( st->defTrans != 0 )
			curIndOffset += 1;
	}
	out << endl;
	return out;
}

std::ostream &GoFlatCodeGen::KEY_SPANS()
{
	out << "	";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write singles length. */
		unsigned long long span = 0;
		if ( st->transList != 0 )
			span = keyOps->span( st->lowKey, st->highKey );
		out << span << ", ";
		if ( !st.last() ) {
			if ( ++totalStateNum % IALL == 0 )
				out << endl << "	";
		}
	}
	out << endl;
	return out;
}

std::ostream &GoFlatCodeGen::TO_STATE_ACTIONS()
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

std::ostream &GoFlatCodeGen::FROM_STATE_ACTIONS()
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

std::ostream &GoFlatCodeGen::EOF_ACTIONS()
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

std::ostream &GoFlatCodeGen::EOF_TRANS()
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
		out << trans << ", ";

		if ( !st.last() ) {
			if ( ++totalStateNum % IALL == 0 )
				out << endl << "	";
		}
	}
	out << endl;
	return out;
}


std::ostream &GoFlatCodeGen::COND_KEYS()
{
	out << "	";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Emit just cond low key and cond high key. */
		out << KEY( st->condLowKey ) << ", ";
		out << KEY( st->condHighKey ) << ", ";
		if ( !st.last() ) {
			if ( ++totalStateNum % IALL == 0 )
				out << endl << "	";
		}
	}

	out << endl;
	return out;
}

std::ostream &GoFlatCodeGen::COND_KEY_SPANS()
{
	out << "	";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write singles length. */
		unsigned long long span = 0;
		if ( st->condList != 0 )
			span = keyOps->span( st->condLowKey, st->condHighKey );
		out << span << ", ";
		if ( !st.last() ) {
			if ( ++totalStateNum % IALL == 0 )
				out << endl << "	";
		}
	}
	out << endl;
	return out;
}

std::ostream &GoFlatCodeGen::CONDS()
{
	out << "	";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st->condList != 0 ) {
			/* Walk the singles. */
			unsigned long long span = keyOps->span( st->condLowKey, st->condHighKey );
			for ( unsigned long long pos = 0; pos < span; pos++ ) {
				if ( st->condList[pos] != 0 )
					out << st->condList[pos]->condSpaceId + 1 << ", ";
				else
					out << "0, ";
				if ( !st.last() ) {
					if ( ++totalStateNum % IALL == 0 )
						out << endl << "	";
				}
			}
		}
	}

	out << endl;
	return out;
}

std::ostream &GoFlatCodeGen::COND_INDEX_OFFSET()
{
	out << "	";
	int totalStateNum = 0;
	int curIndOffset = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write the index offset. */
		out << curIndOffset << ", ";
		if ( !st.last() ) {
			if ( ++totalStateNum % IALL == 0 )
				out << endl << "	";
		}

		/* Move the index offset ahead. */
		if ( st->condList != 0 )
			curIndOffset += keyOps->span( st->condLowKey, st->condHighKey );
	}
	out << endl;
	return out;
}


std::ostream &GoFlatCodeGen::KEYS()
{
	out << "	";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Emit just low key and high key. */
		out << KEY( st->lowKey ) << ", ";
		out << KEY( st->highKey ) << ", ";
		if ( !st.last() ) {
			if ( ++totalStateNum % IALL == 0 )
				out << endl << "	";
		}
	}

	out << endl;
	return out;
}

std::ostream &GoFlatCodeGen::INDICIES()
{
	out << "	";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st->transList != 0 ) {
			/* Walk the singles. */
			unsigned long long span = keyOps->span( st->lowKey, st->highKey );
			for ( unsigned long long pos = 0; pos < span; pos++ ) {
				out << st->transList[pos]->id << ", ";
				if ( ++totalStateNum % IALL == 0 )
					out << endl << "	";
			}
		}

		/* The state's default index goes next. */
		if ( st->defTrans != 0 ) {
			out << st->defTrans->id << ", ";
			if ( ++totalStateNum % IALL == 0 )
				out << endl << "	";
		}
	}

	out << endl;
	return out;
}

std::ostream &GoFlatCodeGen::TRANS_TARGS()
{
	/* Transitions must be written ordered by their id. */
	RedTransAp **transPtrs = new RedTransAp*[redFsm->transSet.length()];
	for ( TransApSet::Iter trans = redFsm->transSet; trans.lte(); trans++ )
		transPtrs[trans->id] = trans;

	/* Keep a count of the num of items in the array written. */
	out << "	";
	int totalStates = 0;
	for ( int t = 0; t < redFsm->transSet.length(); t++ ) {
		/* Save the position. Needed for eofTargs. */
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


std::ostream &GoFlatCodeGen::TRANS_ACTIONS()
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

void GoFlatCodeGen::LOCATE_TRANS()
{
	out <<
		"	_keys = " << CAST(INT(), vCS() + " << 1") << endl <<
		"	_inds = " << CAST(INT(), IO() + "[" + vCS() + "]") << endl <<
		endl <<
		"	_slen = " << CAST(INT(), SP() + "[" + vCS() + "]") << endl <<
		"	if _slen > 0 && " << K() << "[_keys] <= " << GET_WIDE_KEY() << " && " <<
			GET_WIDE_KEY() << " <= " << K() << "[_keys + 1]" << " {" << endl <<
		"		_trans = " << CAST(INT(), I() + "[_inds + " + CAST(INT(), GET_WIDE_KEY() + " - " + K() + "[_keys]") + "]") << endl <<
		"	} else {" << endl <<
		"		_trans = " << CAST(INT(), I() + "[_inds + _slen]") << endl <<
		"	}" << endl <<
		endl;
}

void GoFlatCodeGen::writeData()
{
	/* If there are any transtion functions then output the array. If there
	 * are none, don't bother emitting an empty array that won't be used. */
	if ( redFsm->anyActions() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActArrItem), A() );
		ACTIONS_ARRAY();
		CLOSE_ARRAY() <<
		endl;
	}

	if ( redFsm->anyConditions() ) {
		OPEN_ARRAY( WIDE_ALPH_TYPE(), CK() );
		COND_KEYS();
		CLOSE_ARRAY() <<
		endl;

		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxCondSpan), CSP() );
		COND_KEY_SPANS();
		CLOSE_ARRAY() <<
		endl;

		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxCond), C() );
		CONDS();
		CLOSE_ARRAY() <<
		endl;

		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxCondIndexOffset), CO() );
		COND_INDEX_OFFSET();
		CLOSE_ARRAY() <<
		endl;
	}

	OPEN_ARRAY( WIDE_ALPH_TYPE(), K() );
	KEYS();
	CLOSE_ARRAY() <<
	endl;

	OPEN_ARRAY( ARRAY_TYPE(redFsm->maxSpan), SP() );
	KEY_SPANS();
	CLOSE_ARRAY() <<
	endl;

	OPEN_ARRAY( ARRAY_TYPE(redFsm->maxFlatIndexOffset), IO() );
	FLAT_INDEX_OFFSET();
	CLOSE_ARRAY() <<
	endl;

	OPEN_ARRAY( ARRAY_TYPE(redFsm->maxIndex), I() );
	INDICIES();
	CLOSE_ARRAY() <<
	endl;

	OPEN_ARRAY( ARRAY_TYPE(redFsm->maxState), TT() );
	TRANS_TARGS();
	CLOSE_ARRAY() <<
	endl;

	if ( redFsm->anyActions() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActionLoc), TA() );
		TRANS_ACTIONS();
		CLOSE_ARRAY() <<
		endl;
	}

	if ( redFsm->anyToStateActions() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActionLoc), TSA() );
		TO_STATE_ACTIONS();
		CLOSE_ARRAY() <<
		endl;
	}

	if ( redFsm->anyFromStateActions() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActionLoc), FSA() );
		FROM_STATE_ACTIONS();
		CLOSE_ARRAY() <<
		endl;
	}

	if ( redFsm->anyEofActions() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActionLoc), EA() );
		EOF_ACTIONS();
		CLOSE_ARRAY() <<
		endl;
	}

	if ( redFsm->anyEofTrans() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxIndexOffset+1), ET() );
		EOF_TRANS();
		CLOSE_ARRAY() <<
		endl;
	}

	STATE_IDS();
}

void GoFlatCodeGen::COND_TRANSLATE()
{
	out <<
		"	_widec = " << CAST(WIDE_ALPH_TYPE(), GET_KEY()) << endl;

	out <<
		"	_keys = " << CAST(INT(), vCS() + " << 1") << endl <<
		"	_conds = " << CAST(INT(), CO() + "[" + vCS() + "]") << endl <<
		endl <<
		"	_slen = " << CAST(INT(), CSP() + "[" + vCS() + "]") << endl <<
		"	if _slen > 0 && " << CK() << "[_keys]" << " <= " << GET_WIDE_KEY() << " && " <<
				GET_WIDE_KEY() << " <= " << CK() << "[_keys + 1] {" << endl <<
		"		_cond = " << CAST(INT(), C() + "[_conds + " + CAST(INT(), GET_WIDE_KEY() + " - " + CK() + "[_keys]") + "]") << endl <<
		"	} else {" << endl <<
		"		_cond = 0" << endl <<
		"	}" << endl <<
		endl;

	out <<
		"	switch _cond {" << endl;
	for ( CondSpaceList::Iter csi = condSpaceList; csi.lte(); csi++ ) {
		GenCondSpace *condSpace = csi;
		out << "	case " << condSpace->condSpaceId + 1 << ":" << endl;
		out << TABS(2) << "_widec = " <<
				KEY(condSpace->baseKey) << " + (" << CAST(WIDE_ALPH_TYPE(), GET_KEY()) <<
				" - " << KEY(keyOps->minKey) << ")" << endl;

		for ( GenCondSet::Iter csi = condSpace->condSet; csi.lte(); csi++ ) {
			out << TABS(2) << "if ";
			CONDITION( out, *csi );
			Size condValOffset = ((1 << csi.pos()) * keyOps->alphSize());
			out << " {" << endl <<
				"			_widec += " << condValOffset << endl <<
				"		}" << endl;
		}
	}

	out <<
		"	}" << endl;
}

void GoFlatCodeGen::writeExec()
{
	testEofUsed = false;
	outLabelUsed = false;

	out <<
		"	{" << endl <<
		"	var _slen " << INT() << endl;

	if ( redFsm->anyRegCurStateRef() )
		out << "	var _ps " << INT() << endl;

	out <<
		"	var _trans " << INT() << endl;

	if ( redFsm->anyConditions() )
		out << "	var _cond " << INT() << endl;

	if ( redFsm->anyToStateActions() ||
			redFsm->anyRegActions() || redFsm->anyFromStateActions() )
	{
		out <<
			"	var _acts " << INT() << endl <<
			"	var _nacts " << UINT() << endl;
	}

	out <<
		"	var _keys " << INT() << endl <<
		"	var _inds " << INT() << endl;

	if ( redFsm->anyConditions() ) {
		out <<
			"	var _conds " << INT() << endl <<
			"	var _widec " << WIDE_ALPH_TYPE() << endl;
	}

	out << endl;

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
			"		_acts++" << endl <<
			"		switch " << A() << "[_acts - 1]" << " {" << endl;
			FROM_STATE_ACTION_SWITCH(2);
			out <<
			"		}" << endl <<
			"	}" << endl <<
			endl;
	}

	if ( redFsm->anyConditions() )
		COND_TRANSLATE();

	LOCATE_TRANS();

	if ( redFsm->anyEofTrans() )
		out << "_eof_trans:" << endl;

	if ( redFsm->anyRegCurStateRef() )
		out << "	_ps = " << vCS() << endl;

	out <<
		"	" << vCS() << " = " << CAST(INT(), TT() + "[_trans]") << endl <<
		endl;

	if ( redFsm->anyRegActions() ) {
		out <<
			"	if " << TA() << "[_trans] == 0 {" << endl <<
			"		goto _again" << endl <<
			"	}" << endl <<
			endl <<
			"	_acts = " << CAST(INT(), TA() + "[_trans]") << endl <<
			"	_nacts = " << CAST(UINT(), A() + "[_acts]") << "; _acts++" << endl <<
			"	for ; _nacts > 0; _nacts-- {" << endl <<
			"		_acts++" << endl <<
			"		switch " << A() << "[_acts - 1]" << " {" << endl;
			ACTION_SWITCH(2);
			out <<
			"		}" << endl <<
			"	}" << endl <<
			endl;
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
			"		switch " << A() << "[_acts - 1]" << " {" << endl;
			TO_STATE_ACTION_SWITCH(2);
			out <<
			"		}" << endl <<
			"	}" << endl <<
			endl;
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
			"	if " << P() << "++; " << P() << " != " << PE() << " {" << endl <<
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
				"		__acts := " << CAST(INT(), EA() + "[" + vCS() + "]") << endl <<
				"		__nacts := " << CAST(UINT(), A() + "[__acts]") << "; __acts++" << endl <<
				"		for ; __nacts > 0; __nacts-- {" << endl <<
				"			__acts++" << endl <<
				"			switch " << A() << "[__acts - 1]" << " {" << endl;
				EOF_ACTION_SWITCH(3);
				out <<
				"			}" << endl <<
				"		}" << endl;
		}

		out <<
			"	}" << endl <<
			endl;
	}

	if ( outLabelUsed )
		out << "	_out: {}" << endl;

	out << "	}" << endl;
}
