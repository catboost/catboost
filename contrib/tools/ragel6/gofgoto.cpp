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
#include "gofgoto.h"
#include "redfsm.h"
#include "gendata.h"
#include "bstmap.h"

using std::endl;

std::ostream &GoFGotoCodeGen::EXEC_ACTIONS()
{
	/* Loop the actions. */
	for ( GenActionTableMap::Iter redAct = redFsm->actionMap; redAct.lte(); redAct++ ) {
		if ( redAct->numTransRefs > 0 ) {
			/* 	We are at the start of a glob, write the case. */
			out << "f" << redAct->actListId << ":" << endl;

			/* Write each action in the list of action items. */
			for ( GenActionTable::Iter item = redAct->key; item.lte(); item++ )
				ACTION( out, item->value, 0, false, false );

			out << TABS(1) << "goto _again" << endl;
		}
	}
	return out;
}

/* Write out the function switch. This switch is keyed on the values
 * of the func index. */
std::ostream &GoFGotoCodeGen::TO_STATE_ACTION_SWITCH( int level )
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
std::ostream &GoFGotoCodeGen::FROM_STATE_ACTION_SWITCH( int level )
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

std::ostream &GoFGotoCodeGen::EOF_ACTION_SWITCH( int level )
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


std::ostream &GoFGotoCodeGen::FINISH_CASES()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* States that are final and have an out action need a case. */
		if ( st->eofAction != 0 ) {
			/* Write the case label. */
			out << TABS(2) << "case " << st->id << ":" << endl;

			/* Jump to the func. */
			out << TABS(3) << "goto f" << st->eofAction->actListId << endl;
		}
	}

	return out;
}

unsigned int GoFGotoCodeGen::TO_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->toStateAction != 0 )
		act = state->toStateAction->actListId+1;
	return act;
}

unsigned int GoFGotoCodeGen::FROM_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->fromStateAction != 0 )
		act = state->fromStateAction->actListId+1;
	return act;
}

unsigned int GoFGotoCodeGen::EOF_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->eofAction != 0 )
		act = state->eofAction->actListId+1;
	return act;
}

void GoFGotoCodeGen::writeData()
{
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

	STATE_IDS();
}

void GoFGotoCodeGen::writeExec()
{
	testEofUsed = false;
	outLabelUsed = false;

	out << "	{" << endl;

	if ( redFsm->anyRegCurStateRef() )
		out << "	var _ps " << INT() << " = 0" << endl;

	if ( redFsm->anyConditions() )
		out << "	var _widec " << WIDE_ALPH_TYPE() << endl;

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

	out <<
		"	switch " << vCS() << " {" << endl;
		STATE_GOTOS(1);
		out <<
		"	}" << endl <<
		endl;
		TRANSITIONS() <<
		endl;

	if ( redFsm->anyRegActions() )
		EXEC_ACTIONS() << endl;

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
				"		switch " << vCS() << " {" << endl;

			for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
				if ( st->eofTrans != 0 )
					out <<
						"		case " << st->id << ":" << endl <<
						"			goto tr" << st->eofTrans->id << endl;
			}

			out <<
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
