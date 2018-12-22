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
#include "goipgoto.h"
#include "redfsm.h"
#include "gendata.h"
#include "bstmap.h"

using std::endl;

bool GoIpGotoCodeGen::useAgainLabel()
{
	return 	redFsm->anyRegActionRets() ||
			redFsm->anyRegActionByValControl() ||
			redFsm->anyRegNextStmt();
}

void GoIpGotoCodeGen::GOTO( ostream &ret, int gotoDest, bool inFinish )
{
	ret << "{" << "goto st" << gotoDest << " }";
}

void GoIpGotoCodeGen::CALL( ostream &ret, int callDest, int targState, bool inFinish )
{
	if ( prePushExpr != 0 ) {
		ret << "{";
		INLINE_LIST( ret, prePushExpr, 0, false, false );
	}

	ret << "{" << STACK() << "[" << TOP() << "] = " << targState <<
			"; " << TOP() << "++; " << "goto st" << callDest << " }";

	if ( prePushExpr != 0 )
		ret << "}";
}

void GoIpGotoCodeGen::CALL_EXPR( ostream &ret, GenInlineItem *ilItem, int targState, bool inFinish )
{
	if ( prePushExpr != 0 ) {
		ret << "{";
		INLINE_LIST( ret, prePushExpr, 0, false, false );
	}

	ret << "{" << STACK() << "[" << TOP() << "] = " << targState << "; " << TOP() << "++; " << vCS() << " = (";
	INLINE_LIST( ret, ilItem->children, 0, inFinish, false );
	ret << "); " << "goto _again }";

	if ( prePushExpr != 0 )
		ret << "}";
}

void GoIpGotoCodeGen::RET( ostream &ret, bool inFinish )
{
	ret << "{" << TOP() << "--; " << vCS() << " = " << STACK() << "[" << TOP() << "];";

	if ( postPopExpr != 0 ) {
		ret << "{";
		INLINE_LIST( ret, postPopExpr, 0, false, false );
		ret << "}";
	}

	ret << "goto _again }";
}

void GoIpGotoCodeGen::GOTO_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish )
{
	ret << "{" << vCS() << " = (";
	INLINE_LIST( ret, ilItem->children, 0, inFinish, false );
	ret << "); " << "goto _again }";
}

void GoIpGotoCodeGen::NEXT( ostream &ret, int nextDest, bool inFinish )
{
	ret << vCS() << " = " << nextDest << ";";
}

void GoIpGotoCodeGen::NEXT_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish )
{
	ret << vCS() << " = (";
	INLINE_LIST( ret, ilItem->children, 0, inFinish, false );
	ret << ");";
}

void GoIpGotoCodeGen::CURS( ostream &ret, bool inFinish )
{
	ret << "(_ps)";
}

void GoIpGotoCodeGen::TARGS( ostream &ret, bool inFinish, int targState )
{
	ret << targState;
}

void GoIpGotoCodeGen::BREAK( ostream &ret, int targState, bool csForced )
{
	outLabelUsed = true;
	ret << "{" << P() << "++; ";
	if ( !csForced )
		ret << vCS() << " = " << targState << "; ";
	ret << "goto _out }";
}

bool GoIpGotoCodeGen::IN_TRANS_ACTIONS( RedStateAp *state )
{
	bool anyWritten = false;

	/* Emit any transitions that have actions and that go to this state. */
	for ( int it = 0; it < state->numInTrans; it++ ) {
		RedTransAp *trans = state->inTrans[it];
		if ( trans->action != 0 && trans->labelNeeded ) {
			/* Remember that we wrote an action so we know to write the
			 * line directive for going back to the output. */
			anyWritten = true;

			/* Write the label for the transition so it can be jumped to. */
			out << "tr" << trans->id << ":" << endl;

			/* If the action contains a next, then we must preload the current
			 * state since the action may or may not set it. */
			if ( trans->action->anyNextStmt() )
				out << "	" << vCS() << " = " << trans->targ->id << endl;

			/* Write each action in the list. */
			for ( GenActionTable::Iter item = trans->action->key; item.lte(); item++ ) {
				ACTION( out, item->value, trans->targ->id, false,
						trans->action->anyNextStmt() );
			}

			/* If the action contains a next then we need to reload, otherwise
			 * jump directly to the target state. */
			if ( trans->action->anyNextStmt() )
				out << "	goto _again" << endl;
			else
				out << "	goto st" << trans->targ->id << endl;
		}
	}

	return anyWritten;
}

std::ostream &GoIpGotoCodeGen::STATE_GOTOS_SWITCH( int level )
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		out << TABS(level) << "case " << st->id << ":" << endl;
		out << TABS(level + 1) << "goto st_case_" << st->id << endl;
	}
	return out;
}

/* Called from GotoCodeGen::STATE_GOTOS just before writing the gotos for each
 * state. */
void GoIpGotoCodeGen::GOTO_HEADER( RedStateAp *state, int level )
{
	bool anyWritten = IN_TRANS_ACTIONS( state );

	if ( state->labelNeeded )
		out << TABS(level) << "st" << state->id << ":" << endl;

	if ( state->toStateAction != 0 ) {
		/* Remember that we wrote an action. Write every action in the list. */
		anyWritten = true;
		for ( GenActionTable::Iter item = state->toStateAction->key; item.lte(); item++ ) {
			ACTION( out, item->value, state->id, false,
					state->toStateAction->anyNextStmt() );
		}
	}

	/* Advance and test buffer pos. */
	if ( state->labelNeeded ) {
		if ( !noEnd ) {
			out <<
				TABS(level + 1) << "if " << P() << "++; " << P() << " == " << PE() << " {" << endl <<
				TABS(level + 2) << "goto _test_eof" << state->id << endl <<
				TABS(level + 1) << "}" << endl;
		}
		else {
			out <<
				TABS(level + 1) << P() << "++" << endl;
		}
	}

	/* Give the state a label. */
	out << TABS(level) << "st_case_" << state->id << ":" << endl;

	if ( state->fromStateAction != 0 ) {
		/* Remember that we wrote an action. Write every action in the list. */
		anyWritten = true;
		for ( GenActionTable::Iter item = state->fromStateAction->key; item.lte(); item++ ) {
			ACTION( out, item->value, state->id, false,
					state->fromStateAction->anyNextStmt() );
		}
	}

	if ( anyWritten )
		genLineDirective( out );

	/* Record the prev state if necessary. */
	if ( state->anyRegCurStateRef() )
		out << TABS(level + 1) << "_ps = " << state->id << endl;
}

void GoIpGotoCodeGen::STATE_GOTO_ERROR( int level )
{
	/* In the error state we need to emit some stuff that usually goes into
	 * the header. */
	RedStateAp *state = redFsm->errState;
	bool anyWritten = IN_TRANS_ACTIONS( state );

	/* No case label needed since we don't switch on the error state. */
	if ( anyWritten )
		genLineDirective( out );

	out << "st_case_" << state->id << ":" << endl;
	if ( state->labelNeeded )
		out << TABS(level) << "st" << state->id << ":" << endl;

	/* Break out here. */
	outLabelUsed = true;
	out << TABS(level + 1) << vCS() << " = " << state->id << endl;
	out << TABS(level + 1) << "goto _out" << endl;
}


/* Emit the goto to take for a given transition. */
std::ostream &GoIpGotoCodeGen::TRANS_GOTO( RedTransAp *trans, int level )
{
	if ( trans->action != 0 ) {
		/* Go to the transition which will go to the state. */
		out << TABS(level) << "goto tr" << trans->id;
	}
	else {
		/* Go directly to the target state. */
		out << TABS(level) << "goto st" << trans->targ->id;
	}
	return out;
}

int GoIpGotoCodeGen::TRANS_NR( RedTransAp *trans )
{
	if ( trans->action != 0 ) {
		/* Go to the transition which will go to the state. */
		return trans->id + redFsm->stateList.length();
	}
	else {
		/* Go directly to the target state. */
		return trans->targ->id;
	}
}

std::ostream &GoIpGotoCodeGen::EXIT_STATES()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st->outNeeded ) {
			testEofUsed = true;
			out << "	_test_eof" << st->id << ": " << vCS() << " = " <<
					st->id << "; goto _test_eof" << endl;
		}
	}
	return out;
}

std::ostream &GoIpGotoCodeGen::AGAIN_CASES( int level )
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		out << 
			TABS(level) << "case " << st->id << ":" << endl <<
			TABS(level + 1) << "goto st" << st->id << endl;
	}
	return out;
}

std::ostream &GoIpGotoCodeGen::FINISH_CASES( int level )
{
	bool anyWritten = false;

	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st->eofAction != 0 ) {
			if ( st->eofAction->eofRefs == 0 )
				st->eofAction->eofRefs = new IntSet;
			st->eofAction->eofRefs->insert( st->id );
		}
	}

	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st->eofTrans != 0 )
			out << TABS(level) << "case " << st->id << ":" << endl <<
			       TABS(level + 1) << "goto tr" << st->eofTrans->id << endl;
	}

	for ( GenActionTableMap::Iter act = redFsm->actionMap; act.lte(); act++ ) {
		if ( act->eofRefs != 0 ) {
			out << TABS(level) << "case ";
			for ( IntSet::Iter pst = *act->eofRefs; pst.lte(); pst++ ) {
				out << *pst;
				if ( !pst.last() )
					out << ", ";
			}
			out << ":" << endl;

			/* Remember that we wrote a trans so we know to write the
			 * line directive for going back to the output. */
			anyWritten = true;

			/* Write each action in the eof action list. */
			for ( GenActionTable::Iter item = act->key; item.lte(); item++ )
				ACTION( out, item->value, STATE_ERR_STATE, true, false );
		}
	}

	if ( anyWritten )
		genLineDirective( out );
	return out;
}

void GoIpGotoCodeGen::setLabelsNeeded( GenInlineList *inlineList )
{
	for ( GenInlineList::Iter item = *inlineList; item.lte(); item++ ) {
		switch ( item->type ) {
		case GenInlineItem::Goto: case GenInlineItem::Call: {
			/* Mark the target as needing a label. */
			item->targState->labelNeeded = true;
			break;
		}
		default: break;
		}

		if ( item->children != 0 )
			setLabelsNeeded( item->children );
	}
}

/* Set up labelNeeded flag for each state. */
void GoIpGotoCodeGen::setLabelsNeeded()
{
	/* If we use the _again label, then we the _again switch, which uses all
	 * labels. */
	if ( useAgainLabel() ) {
		for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ )
			st->labelNeeded = true;
	}
	else {
		/* Do not use all labels by default, init all labelNeeded vars to false. */
		for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ )
			st->labelNeeded = false;

		/* Walk all transitions and set only those that have targs. */
		for ( TransApSet::Iter trans = redFsm->transSet; trans.lte(); trans++ ) {
			/* If there is no action with a next statement, then the label will be
			 * needed. */
			if ( trans->action == 0 || !trans->action->anyNextStmt() )
				trans->targ->labelNeeded = true;

			/* Need labels for states that have goto or calls in action code
			 * invoked on characters (ie, not from out action code). */
			if ( trans->action != 0 ) {
				/* Loop the actions. */
				for ( GenActionTable::Iter act = trans->action->key; act.lte(); act++ ) {
					/* Get the action and walk it's tree. */
					setLabelsNeeded( act->value->inlineList );
				}
			}
		}
	}

	if ( !noEnd ) {
		for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
			if ( st != redFsm->errState )
				st->outNeeded = st->labelNeeded;
		}
	}
}

void GoIpGotoCodeGen::writeData()
{
	STATE_IDS();
}

void GoIpGotoCodeGen::writeExec()
{
	/* Must set labels immediately before writing because we may depend on the
	 * noend write option. */
	setLabelsNeeded();
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

	if ( useAgainLabel() ) {
		out <<
			"	goto _resume" << endl <<
			endl <<
			"_again:" << endl <<
			"	switch " << vCS() << " {" << endl;
			AGAIN_CASES(1) <<
			"	}" << endl <<
			endl;

		if ( !noEnd ) {
			testEofUsed = true;
			out <<
				"	if " << P() << "++; " << P() << " == " << PE() << " {" << endl <<
				"		goto _test_eof" << endl <<
				"	}" << endl;
		}
		else {
			out <<
				"	" << P() << "++" << endl;
		}
		out << "_resume:" << endl;
	}

	out <<
		"	switch " << vCS() << " {" << endl;
		STATE_GOTOS_SWITCH(1);
		out <<
		"	}" << endl;
		out << "	goto st_out" << endl;
		STATE_GOTOS(1);
		out << "	st_out:" << endl;
		EXIT_STATES() <<
		endl;

	if ( testEofUsed )
		out << "	_test_eof: {}" << endl;

	if ( redFsm->anyEofTrans() || redFsm->anyEofActions() ) {
		out <<
			"	if " << P() << " == " << vEOF() << " {" << endl <<
			"		switch " << vCS() << " {" << endl;
			FINISH_CASES(2);
			out <<
			"		}" << endl <<
			"	}" << endl <<
			endl;
	}

	if ( outLabelUsed )
		out << "	_out: {}" << endl;

	out <<
		"	}" << endl;
}
