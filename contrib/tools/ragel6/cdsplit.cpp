/*
 *  Copyright 2006 Adrian Thurston <thurston@complang.org>
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
#include "cdsplit.h"
#include "gendata.h"
#include <assert.h>

using std::ostream;
using std::ios;
using std::endl;

/* Emit the goto to take for a given transition. */
std::ostream &SplitCodeGen::TRANS_GOTO( RedTransAp *trans, int level )
{
	if ( trans->targ->partition == currentPartition ) {
		if ( trans->action != 0 ) {
			/* Go to the transition which will go to the state. */
			out << TABS(level) << "goto tr" << trans->id << ";";
		}
		else {
			/* Go directly to the target state. */
			out << TABS(level) << "goto st" << trans->targ->id << ";";
		}
	}
	else {
		if ( trans->action != 0 ) {
			/* Go to the transition which will go to the state. */
			out << TABS(level) << "goto ptr" << trans->id << ";";
			trans->partitionBoundary = true;
		}
		else {
			/* Go directly to the target state. */
			out << TABS(level) << "goto pst" << trans->targ->id << ";";
			trans->targ->partitionBoundary = true;
		}
	}
	return out;
}

/* Called from before writing the gotos for each state. */
void SplitCodeGen::GOTO_HEADER( RedStateAp *state, bool stateInPartition )
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
				"		goto _out" << state->id << ";\n";
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

std::ostream &SplitCodeGen::STATE_GOTOS( int partition )
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st->partition == partition ) {
			if ( st == redFsm->errState )
				STATE_GOTO_ERROR();
			else {
				/* We call into the base of the goto which calls back into us
				 * using virtual functions. Set the current partition rather
				 * than coding parameter passing throughout. */
				currentPartition = partition;

				/* Writing code above state gotos. */
				GOTO_HEADER( st, st->partition == partition );

				if ( st->stateCondVect.length() > 0 ) {
					out << "	_widec = " << GET_KEY() << ";\n";
					emitCondBSearch( st, 1, 0, st->stateCondVect.length() - 1 );
				}

				/* Try singles. */
				if ( st->outSingle.length() > 0 )
					emitSingleSwitch( st );

				/* Default case is to binary search for the ranges, if that fails then */
				if ( st->outRange.length() > 0 )
					emitRangeBSearch( st, 1, 0, st->outRange.length() - 1 );

				/* Write the default transition. */
				TRANS_GOTO( st->defTrans, 1 ) << "\n";
			}
		}
	}
	return out;
}


std::ostream &SplitCodeGen::PART_TRANS( int partition )
{
	for ( TransApSet::Iter trans = redFsm->transSet; trans.lte(); trans++ ) {
		if ( trans->partitionBoundary ) {
			out << 
				"ptr" << trans->id << ":\n";

			if ( trans->action != 0 ) {
				/* If the action contains a next, then we must preload the current
				 * state since the action may or may not set it. */
				if ( trans->action->anyNextStmt() )
					out << "	" << vCS() << " = " << trans->targ->id << ";\n";

				/* Write each action in the list. */
				for ( GenActionTable::Iter item = trans->action->key; item.lte(); item++ ) {
					ACTION( out, item->value, trans->targ->id, false,
							trans->action->anyNextStmt() );
				}
			}

			out <<
				"	goto pst" << trans->targ->id << ";\n";
			trans->targ->partitionBoundary = true;
		}
	}

	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st->partitionBoundary ) {
			out << 
				"	pst" << st->id << ":\n" 
				"	" << vCS() << " = " << st->id << ";\n";

			if ( st->toStateAction != 0 ) {
				/* Remember that we wrote an action. Write every action in the list. */
				for ( GenActionTable::Iter item = st->toStateAction->key; item.lte(); item++ ) {
					ACTION( out, item->value, st->id, false,
							st->toStateAction->anyNextStmt() );
				}
				genLineDirective( out );
			}

			ptOutLabelUsed = true;
			out << "	goto _pt_out; \n";
		}
	}
	return out;
}

std::ostream &SplitCodeGen::EXIT_STATES( int partition )
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st->partition == partition && st->outNeeded ) {
			outLabelUsed = true;
			out << "	_out" << st->id << ": " << vCS() << " = " << 
					st->id << "; goto _out; \n";
		}
	}
	return out;
}


std::ostream &SplitCodeGen::PARTITION( int partition )
{
	outLabelUsed = false;
	ptOutLabelUsed = false;

	/* Initialize the partition boundaries, which get set during the writing
	 * of states. After the state writing we will */
	for ( TransApSet::Iter trans = redFsm->transSet; trans.lte(); trans++ )
		trans->partitionBoundary = false;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ )
		st->partitionBoundary = false;

	out << "	" << ALPH_TYPE() << " *p = *_pp, *pe = *_ppe;\n";

	if ( redFsm->anyRegCurStateRef() )
		out << "	int _ps = 0;\n";

	if ( redFsm->anyConditions() )
		out << "	" << WIDE_ALPH_TYPE() << " _widec;\n";

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
			outLabelUsed = true;
			out << 
				"	if ( ++" << P() << " == " << PE() << " )\n"
				"		goto _out;\n";

		}
		else {
			out << 
				"	" << P() << " += 1;\n";
		}

		out <<
			"_resume:\n";
	}

	out << 
		"	switch ( " << vCS() << " )\n	{\n";
		STATE_GOTOS( partition );
		SWITCH_DEFAULT() <<
		"	}\n";
		PART_TRANS( partition );
		EXIT_STATES( partition );

	if ( outLabelUsed ) {
		out <<
			"\n"
			"	_out:\n"
			"	*_pp = p;\n"
			"	*_ppe = pe;\n"
			"	return 0;\n";
	}

	if ( ptOutLabelUsed ) {
		out <<
			"\n"
			"	_pt_out:\n"
			"	*_pp = p;\n"
			"	*_ppe = pe;\n"
			"	return 1;\n";
	}

	return out;
}

std::ostream &SplitCodeGen::PART_MAP()
{
	int *partMap = new int[redFsm->stateList.length()];
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ )
		partMap[st->id] = st->partition;

	out << "\t";
	int totalItem = 0;
	for ( int i = 0; i < redFsm->stateList.length(); i++ ) {
		out << partMap[i];
		if ( i != redFsm->stateList.length() - 1 ) {
			out << ", ";
			if ( ++totalItem % IALL == 0 )
				out << "\n\t";
		}
	}

	delete[] partMap;
	return out;
}

void SplitCodeGen::writeData()
{
	out <<
		"static const int " << START() << " = " << START_STATE_ID() << ";\n"
		"\n";

	if ( !noFinal ) {
		out <<
			"static const int " << FIRST_FINAL() << " = " << FIRST_FINAL_STATE() << ";\n"
			"\n";
	}

	if ( !noError ) {
		out <<
			"static const int " << ERROR() << " = " << ERROR_STATE() << ";\n"
			"\n";
	}


	OPEN_ARRAY( ARRAY_TYPE(numSplitPartitions), PM() );
	PART_MAP();
	CLOSE_ARRAY() <<
	"\n";

	for ( int p = 0; p < redFsm->nParts; p++ ) {
		out << "int partition" << p << "( " << ALPH_TYPE() << " **_pp, " << ALPH_TYPE() << 
			" **_ppe, struct " << FSM_NAME() << " *fsm );\n";
	}
	out << "\n";
}

std::ostream &SplitCodeGen::ALL_PARTITIONS()
{
	/* compute the format string. */
	int width = 0, high = redFsm->nParts - 1;
	while ( high > 0 ) {
		width++;
		high /= 10;
	}
	assert( width <= 8 );
	char suffFormat[] = "_%6.6d.c";
	suffFormat[2] = suffFormat[4] = ( '0' + width );

	for ( int p = 0; p < redFsm->nParts; p++ ) {
		char suffix[10];
		sprintf( suffix, suffFormat, p );
		const char *fn = fileNameFromStem( sourceFileName, suffix );
		const char *include = fileNameFromStem( sourceFileName, ".h" );

		/* Create the filter on the output and open it. */
		output_filter *partFilter = new output_filter( fn );
		partFilter->open( fn, ios::out|ios::trunc );
		if ( !partFilter->is_open() ) {
			error() << "error opening " << fn << " for writing" << endl;
			exit(1);
		}

		/* Attach the new file to the output stream. */
		std::streambuf *prev_rdbuf = out.rdbuf( partFilter );

		out << 
			"#include \"" << include << "\"\n"
			"int partition" << p << "( " << ALPH_TYPE() << " **_pp, " << ALPH_TYPE() << 
					" **_ppe, struct " << FSM_NAME() << " *fsm )\n"
			"{\n";
			PARTITION( p ) <<
			"}\n\n";
		out.flush();

		/* Fix the output stream. */
		out.rdbuf( prev_rdbuf );
	}
	return out;
}


void SplitCodeGen::writeExec()
{
	/* Must set labels immediately before writing because we may depend on the
	 * noend write option. */
	setLabelsNeeded();
	out << 
		"	{\n"
		"	int _stat = 0;\n";

	if ( !noEnd ) {
		out <<
			"	if ( " << P() << " == " << PE() << " )\n"
			"		goto _out;\n";
	}

	out << "	goto _resume;\n";
	
	/* In this reentry, to-state actions have already been executed on the
	 * partition-switch exit from the last partition. */
	out << "_reenter:\n";

	if ( !noEnd ) {
		out <<
			"	if ( ++" << P() << " == " << PE() << " )\n"
			"		goto _out;\n";
	}
	else {
		out << 
			"	" << P() << " += 1;\n";
	}

	out << "_resume:\n";

	out << 
		"	switch ( " << PM() << "[" << vCS() << "] ) {\n";
	for ( int p = 0; p < redFsm->nParts; p++ ) {
		out <<
			"	case " << p << ":\n"
			"		_stat = partition" << p << "( &p, &pe, fsm );\n"
			"		break;\n";
	}
	out <<
		"	}\n"
		"	if ( _stat )\n"
		"		goto _reenter;\n";
	
	if ( !noEnd )
		out << "	_out: {}\n";

	out <<
		"	}\n";
	
	ALL_PARTITIONS();
}

void SplitCodeGen::setLabelsNeeded( RedStateAp *fromState, GenInlineList *inlineList )
{
	for ( GenInlineList::Iter item = *inlineList; item.lte(); item++ ) {
		switch ( item->type ) {
		case GenInlineItem::Goto: case GenInlineItem::Call: {
			/* In split code gen we only need labels for transitions across
			 * partitions. */
			if ( fromState->partition == item->targState->partition ){
				/* Mark the target as needing a label. */
				item->targState->labelNeeded = true;
			}
			break;
		}
		default: break;
		}

		if ( item->children != 0 )
			setLabelsNeeded( fromState, item->children );
	}
}

void SplitCodeGen::setLabelsNeeded( RedStateAp *fromState, RedTransAp *trans )
{
	/* In the split code gen we don't need labels for transitions across
	 * partitions. */
	if ( fromState->partition == trans->targ->partition ) {
		/* If there is no action with a next statement, then the label will be
		 * needed. */
		trans->labelNeeded = true;
		if ( trans->action == 0 || !trans->action->anyNextStmt() )
			trans->targ->labelNeeded = true;
	}

	/* Need labels for states that have goto or calls in action code
	 * invoked on characters (ie, not from out action code). */
	if ( trans->action != 0 ) {
		/* Loop the actions. */
		for ( GenActionTable::Iter act = trans->action->key; act.lte(); act++ ) {
			/* Get the action and walk it's tree. */
			setLabelsNeeded( fromState, act->value->inlineList );
		}
	}
}

/* Set up labelNeeded flag for each state. */
void SplitCodeGen::setLabelsNeeded()
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
		for ( TransApSet::Iter trans = redFsm->transSet; trans.lte(); trans++ )
			trans->labelNeeded = false;

		/* Walk all transitions and set only those that have targs. */
		for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
			for ( RedTransList::Iter tel = st->outRange; tel.lte(); tel++ )
				setLabelsNeeded( st, tel->value );

			for ( RedTransList::Iter tel = st->outSingle; tel.lte(); tel++ )
				setLabelsNeeded( st, tel->value );

			if ( st->defTrans != 0 )
				setLabelsNeeded( st, st->defTrans );
		}
	}

	if ( !noEnd ) {
		for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ )
			st->outNeeded = st->labelNeeded;
	}
	else {
		if ( redFsm->errState != 0 )
			redFsm->errState->outNeeded = true;

		for ( TransApSet::Iter trans = redFsm->transSet; trans.lte(); trans++ ) {
			/* Any state with a transition in that has a break will need an
			 * out label. */
			if ( trans->action != 0 && trans->action->anyBreakStmt() )
				trans->targ->outNeeded = true;
		}
	}
}

