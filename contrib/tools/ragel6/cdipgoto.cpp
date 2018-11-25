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
#include "cdipgoto.h"
#include "redfsm.h"
#include "gendata.h"
#include "bstmap.h"

bool IpGotoCodeGen::useAgainLabel()
{
	return redFsm->anyActionRets() || 
			redFsm->anyActionByValControl() || 
			redFsm->anyRegNextStmt();
}

void IpGotoCodeGen::EOF_CHECK( ostream &ret, int gotoDest )
{
	ret << 
		"       if ( " << P() << " == " << PE() << " )\n"
		"               goto _test_eof" << gotoDest << ";\n";

	testEofUsed = true;
}

void IpGotoCodeGen::GOTO( ostream &ret, int gotoDest, bool inFinish )
{
	ret << "{";

	if ( inFinish && !noEnd )
		EOF_CHECK( ret, gotoDest );

	ret << CTRL_FLOW() << "goto st" << gotoDest << ";";

	ret << "}";
}

void IpGotoCodeGen::CALL( ostream &ret, int callDest, int targState, bool inFinish )
{
	if ( prePushExpr != 0 ) {
		ret << "{";
		INLINE_LIST( ret, prePushExpr, 0, false, false );
	}

	ret << "{" << STACK() << "[" << TOP() << "++] = " << targState << ";";

	if ( inFinish && !noEnd )
		EOF_CHECK( ret, callDest );

	ret << CTRL_FLOW() << "goto st" << callDest << ";";

	ret << "}";

	if ( prePushExpr != 0 )
		ret << "}";
}

void IpGotoCodeGen::CALL_EXPR( ostream &ret, GenInlineItem *ilItem, int targState, bool inFinish )
{
	if ( prePushExpr != 0 ) {
		ret << "{";
		INLINE_LIST( ret, prePushExpr, 0, false, false );
	}

	ret << "{";

	ret << STACK() << "[" << TOP() << "++] = " << targState << "; " << vCS() << " = (";
	INLINE_LIST( ret, ilItem->children, 0, inFinish, false );
	ret << ");";
	
	if ( inFinish && !noEnd )
		FsmCodeGen::EOF_CHECK( ret );

	ret << CTRL_FLOW() << "goto _again;";

	ret << "}";

	if ( prePushExpr != 0 )
		ret << "}";
}

void IpGotoCodeGen::RET( ostream &ret, bool inFinish )
{
	ret << "{" << vCS() << " = " << STACK() << "[--" << TOP() << "];";

	if ( postPopExpr != 0 ) {
		ret << "{";
		INLINE_LIST( ret, postPopExpr, 0, false, false );
		ret << "}";
	}

	if ( inFinish && !noEnd )
		FsmCodeGen::EOF_CHECK( ret );

	ret << CTRL_FLOW() << "goto _again;";

	ret << "}";
}

void IpGotoCodeGen::GOTO_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish )
{
	ret << "{";

	ret << vCS() << " = (";
	INLINE_LIST( ret, ilItem->children, 0, inFinish, false );
	ret << ");";

	if ( inFinish && !noEnd )
		FsmCodeGen::EOF_CHECK( ret );

	ret << CTRL_FLOW() << "goto _again;";
	ret << "}";
}

void IpGotoCodeGen::NEXT( ostream &ret, int nextDest, bool inFinish )
{
	ret << vCS() << " = " << nextDest << ";";
}

void IpGotoCodeGen::NEXT_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish )
{
	ret << vCS() << " = (";
	INLINE_LIST( ret, ilItem->children, 0, inFinish, false );
	ret << ");";
}

void IpGotoCodeGen::CURS( ostream &ret, bool inFinish )
{
	ret << "(_ps)";
}

void IpGotoCodeGen::TARGS( ostream &ret, bool inFinish, int targState )
{
	ret << targState;
}

void IpGotoCodeGen::BREAK( ostream &ret, int targState, bool csForced )
{
	outLabelUsed = true;
	ret << "{" << P() << "++; ";
	if ( !csForced ) 
		ret << vCS() << " = " << targState << "; ";
	ret << CTRL_FLOW() << "goto _out;}";
}

bool IpGotoCodeGen::IN_TRANS_ACTIONS( RedStateAp *state )
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
			out << "tr" << trans->id << ":\n";

			/* If the action contains a next, then we must preload the current
			 * state since the action may or may not set it. */
			if ( trans->action->anyNextStmt() )
				out << "	" << vCS() << " = " << trans->targ->id << ";\n";

			/* Write each action in the list. */
			for ( GenActionTable::Iter item = trans->action->key; item.lte(); item++ ) {
				ACTION( out, item->value, trans->targ->id, false, 
						trans->action->anyNextStmt() );
			}

			/* If the action contains a next then we need to reload, otherwise
			 * jump directly to the target state. */
			if ( trans->action->anyNextStmt() )
				out << "\tgoto _again;\n";
			else
				out << "\tgoto st" << trans->targ->id << ";\n";
		}
	}

	return anyWritten;
}

/* Called from GotoCodeGen::STATE_GOTOS just before writing the gotos for each
 * state. */
void IpGotoCodeGen::GOTO_HEADER( RedStateAp *state )
{
	bool anyWritten = IN_TRANS_ACTIONS( state );

	if ( state->labelNeeded ) 
		out << "st" << state->id << ":\n";

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
				"	if ( ++" << P() << " == " << PE() << " )\n"
				"		goto _test_eof" << state->id << ";\n";
		}
		else {
			out << 
				"	" << P() << " += 1;\n";
		}
	}

	/* Give the state a switch case. */
	out << "case " << state->id << ":\n";

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
		out << "	_ps = " << state->id << ";\n";
}

void IpGotoCodeGen::STATE_GOTO_ERROR()
{
	/* In the error state we need to emit some stuff that usually goes into
	 * the header. */
	RedStateAp *state = redFsm->errState;
	bool anyWritten = IN_TRANS_ACTIONS( state );

	/* No case label needed since we don't switch on the error state. */
	if ( anyWritten )
		genLineDirective( out );

	if ( state->labelNeeded ) 
		out << "st" << state->id << ":\n";

	/* Break out here. */
	outLabelUsed = true;
	out << vCS() << " = " << state->id << ";\n";
	out << "	goto _out;\n";
}


/* Emit the goto to take for a given transition. */
std::ostream &IpGotoCodeGen::TRANS_GOTO( RedTransAp *trans, int level )
{
	if ( trans->action != 0 ) {
		/* Go to the transition which will go to the state. */
		out << TABS(level) << "goto tr" << trans->id << ";";
	}
	else {
		/* Go directly to the target state. */
		out << TABS(level) << "goto st" << trans->targ->id << ";";
	}
	return out;
}

std::ostream &IpGotoCodeGen::EXIT_STATES()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st->outNeeded ) {
			testEofUsed = true;
			out << "	_test_eof" << st->id << ": " << vCS() << " = " << 
					st->id << "; goto _test_eof;\n";
		}
	}
	return out;
}

std::ostream &IpGotoCodeGen::AGAIN_CASES()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		out << 
			"		case " << st->id << ": goto st" << st->id << ";\n";
	}
	return out;
}

std::ostream &IpGotoCodeGen::FINISH_CASES()
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
			out << "	case " << st->id << ": goto tr" << st->eofTrans->id << ";\n";
	}

	for ( GenActionTableMap::Iter act = redFsm->actionMap; act.lte(); act++ ) {
		if ( act->eofRefs != 0 ) {
			for ( IntSet::Iter pst = *act->eofRefs; pst.lte(); pst++ )
				out << "	case " << *pst << ": \n";

			/* Remember that we wrote a trans so we know to write the
			 * line directive for going back to the output. */
			anyWritten = true;

			/* Write each action in the eof action list. */
			for ( GenActionTable::Iter item = act->key; item.lte(); item++ )
				ACTION( out, item->value, STATE_ERR_STATE, true, false );
			out << "\tbreak;\n";
		}
	}

	if ( anyWritten )
		genLineDirective( out );
	return out;
}

void IpGotoCodeGen::setLabelsNeeded( GenInlineList *inlineList )
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
void IpGotoCodeGen::setLabelsNeeded()
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

		for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
			if ( st->eofAction != 0 ) {
				for ( GenActionTable::Iter item = st->eofAction->key; item.lte(); item++ )
					setLabelsNeeded( item->value->inlineList );
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

void IpGotoCodeGen::writeData()
{
	STATE_IDS();
}

void IpGotoCodeGen::writeExec()
{
	/* Must set labels immediately before writing because we may depend on the
	 * noend write option. */
	setLabelsNeeded();
	testEofUsed = false;
	outLabelUsed = false;

	out << "	{\n";

	if ( redFsm->anyRegCurStateRef() )
		out << "	int _ps = 0;\n";

	if ( redFsm->anyConditions() )
		out << "	" << WIDE_ALPH_TYPE() << " _widec;\n";

	if ( !noEnd ) {
		testEofUsed = true;
		out << 
			"	if ( " << P() << " == " << PE() << " )\n"
			"		goto _test_eof;\n";
	}

	if ( useAgainLabel() ) {
		out << 
			"	goto _resume;\n"
			"\n"
			"_again:\n"
			"	switch ( " << vCS() << " ) {\n";
			AGAIN_CASES() <<
			"	default: break;\n"
			"	}\n"
			"\n";

		if ( !noEnd ) {
			testEofUsed = true;
			out << 
				"	if ( ++" << P() << " == " << PE() << " )\n"
				"		goto _test_eof;\n";
		}
		else {
			out << 
				"	" << P() << " += 1;\n";
		}

		out << "_resume:\n";
	}

	out << 
		"	switch ( " << vCS() << " )\n	{\n";
		STATE_GOTOS();
		SWITCH_DEFAULT() <<
		"	}\n";
		EXIT_STATES() << 
		"\n";

	if ( testEofUsed ) 
		out << "	_test_eof: {}\n";

	if ( redFsm->anyEofTrans() || redFsm->anyEofActions() ) {
		out <<
			"	if ( " << P() << " == " << vEOF() << " )\n"
			"	{\n"
			"	switch ( " << vCS() << " ) {\n";
			FINISH_CASES();
			SWITCH_DEFAULT() <<
			"	}\n"
			"	}\n"
			"\n";
	}

	if ( outLabelUsed ) 
		out << "	_out: {}\n";

	out <<
		"	}\n";
}
