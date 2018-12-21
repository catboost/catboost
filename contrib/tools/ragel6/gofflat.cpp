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
#include "gofflat.h"
#include "redfsm.h"
#include "gendata.h"

using std::endl;

std::ostream &GoFFlatCodeGen::TO_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->toStateAction != 0 )
		act = state->toStateAction->actListId+1;
	out << act;
	return out;
}

std::ostream &GoFFlatCodeGen::FROM_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->fromStateAction != 0 )
		act = state->fromStateAction->actListId+1;
	out << act;
	return out;
}

std::ostream &GoFFlatCodeGen::EOF_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->eofAction != 0 )
		act = state->eofAction->actListId+1;
	out << act;
	return out;
}

/* Write out the function for a transition. */
std::ostream &GoFFlatCodeGen::TRANS_ACTION( RedTransAp *trans )
{
	int action = 0;
	if ( trans->action != 0 )
		action = trans->action->actListId+1;
	out << action;
	return out;
}

/* Write out the function switch. This switch is keyed on the values
 * of the func index. */
std::ostream &GoFFlatCodeGen::TO_STATE_ACTION_SWITCH( int level )
{
	/* Loop the actions. */
	for ( GenActionTableMap::Iter redAct = redFsm->actionMap; redAct.lte(); redAct++ ) {
		if ( redAct->numToStateRefs > 0 ) {
			/* Write the entry label. */
			out << TABS(level) << "case " << redAct->actListId+1 << ":" << endl;

			/* Write each action in the list of action items. */
			for ( GenActionTable::Iter item = redAct->key; item.lte(); item++ )
				ACTION( out, item->value, 0, false, false );
		}
	}

	genLineDirective( out );
	return out;
}

/* Write out the function switch. This switch is keyed on the values
 * of the func index. */
std::ostream &GoFFlatCodeGen::FROM_STATE_ACTION_SWITCH( int level )
{
	/* Loop the actions. */
	for ( GenActionTableMap::Iter redAct = redFsm->actionMap; redAct.lte(); redAct++ ) {
		if ( redAct->numFromStateRefs > 0 ) {
			/* Write the entry label. */
			out << TABS(level) << "case " << redAct->actListId+1 << ":" << endl;

			/* Write each action in the list of action items. */
			for ( GenActionTable::Iter item = redAct->key; item.lte(); item++ )
				ACTION( out, item->value, 0, false, false );
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &GoFFlatCodeGen::EOF_ACTION_SWITCH( int level )
{
	/* Loop the actions. */
	for ( GenActionTableMap::Iter redAct = redFsm->actionMap; redAct.lte(); redAct++ ) {
		if ( redAct->numEofRefs > 0 ) {
			/* Write the entry label. */
			out << TABS(level) << "case " << redAct->actListId+1 << ":" << endl;

			/* Write each action in the list of action items. */
			for ( GenActionTable::Iter item = redAct->key; item.lte(); item++ )
				ACTION( out, item->value, 0, true, false );
		}
	}

	genLineDirective( out );
	return out;
}

/* Write out the function switch. This switch is keyed on the values
 * of the func index. */
std::ostream &GoFFlatCodeGen::ACTION_SWITCH( int level )
{
	/* Loop the actions. */
	for ( GenActionTableMap::Iter redAct = redFsm->actionMap; redAct.lte(); redAct++ ) {
		if ( redAct->numTransRefs > 0 ) {
			/* Write the entry label. */
			out << TABS(level) << "case " << redAct->actListId+1 << ":" << endl;

			/* Write each action in the list of action items. */
			for ( GenActionTable::Iter item = redAct->key; item.lte(); item++ )
				ACTION( out, item->value, 0, false, false );
		}
	}

	genLineDirective( out );
	return out;
}

void GoFFlatCodeGen::writeData()
{
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
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActListId), TA() );
		TRANS_ACTIONS();
		CLOSE_ARRAY() <<
		endl;
	}

	if ( redFsm->anyToStateActions() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActionLoc),  TSA() );
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
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActListId), EA() );
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

void GoFFlatCodeGen::writeExec()
{
	testEofUsed = false;
	outLabelUsed = false;

	out <<
		"	{" << endl <<
		"	var _slen " << INT() << endl;

	if ( redFsm->anyRegCurStateRef() )
		out << "	var _ps " << INT() << endl;

	out << "	var _trans " << INT() << endl;

	if ( redFsm->anyConditions() )
		out << "	var _cond " << INT() << endl;

	out <<
		"	var _keys " << INT() << endl <<
		"	var _inds " << INT() << endl;

	if ( redFsm->anyConditions() ) {
		out <<
			"	var _conds " << INT() << endl <<
			"	var _widec " << WIDE_ALPH_TYPE() << endl;
	}

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
			"	switch " << FSA() << "[" << vCS() << "] {" << endl;
			FROM_STATE_ACTION_SWITCH(1);
			out <<
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
			"	switch " << TA() << "[_trans] {" << endl;
			ACTION_SWITCH(1);
			out <<
			"	}" << endl <<
			endl;
	}

	if ( redFsm->anyRegActions() || redFsm->anyActionGotos() ||
			redFsm->anyActionCalls() || redFsm->anyActionRets() )
		out << "_again:" << endl;

	if ( redFsm->anyToStateActions() ) {
		out <<
			"	switch " << TSA() << "[" << vCS() << "] {" << endl;
			TO_STATE_ACTION_SWITCH(1);
			out <<
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
			"	if " << P() << "++; " << P() << " != " << PE() << " {"
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
				"		switch " << EA() << "[" << vCS() << "] {" << endl;
				EOF_ACTION_SWITCH(2);
				out <<
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
