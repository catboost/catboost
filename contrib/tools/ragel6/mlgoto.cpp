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
#include "mlgoto.h"
#include "redfsm.h"
#include "bstmap.h"
#include "gendata.h"

/* Emit the goto to take for a given transition. */
std::ostream &OCamlGotoCodeGen::TRANS_GOTO( RedTransAp *trans, int level )
{
	out << TABS(level) << "tr" << trans->id << " ()";
	return out;
}

std::ostream &OCamlGotoCodeGen::TO_STATE_ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numToStateRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << "\t| " << act->actionId << " ->\n";
			ACTION( out, act, 0, false );
			out << "\t()\n";
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &OCamlGotoCodeGen::FROM_STATE_ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numFromStateRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << "\t| " << act->actionId << " ->\n";
			ACTION( out, act, 0, false );
			out << "\t()\n";
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &OCamlGotoCodeGen::EOF_ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numEofRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << "\t| " << act->actionId << " ->\n";
			ACTION( out, act, 0, true );
			out << "\t()\n";
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &OCamlGotoCodeGen::ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numTransRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << "\t| " << act->actionId << " ->\n";
			ACTION( out, act, 0, false );
			out << "\t()\n";
		}
	}

	genLineDirective( out );
	return out;
}

void OCamlGotoCodeGen::GOTO_HEADER( RedStateAp *state )
{
	/* Label the state. */
	out << "| " << state->id << " ->\n";
}


void OCamlGotoCodeGen::emitSingleSwitch( RedStateAp *state )
{
	/* Load up the singles. */
	int numSingles = state->outSingle.length();
	RedTransEl *data = state->outSingle.data;

	if ( numSingles == 1 ) {
		/* If there is a single single key then write it out as an if. */
		out << "\tif " << GET_WIDE_KEY(state) << " = " << 
				KEY(data[0].lowKey) << " then\n\t\t"; 

		/* Virtual function for writing the target of the transition. */
		TRANS_GOTO(data[0].value, 0) << " else\n";
	}
	else if ( numSingles > 1 ) {
		/* Write out single keys in a switch if there is more than one. */
		out << "\tmatch " << GET_WIDE_KEY(state) << " with\n";

		/* Write out the single indicies. */
		for ( int j = 0; j < numSingles; j++ ) {
			out << "\t\t| " << ALPHA_KEY(data[j].lowKey) << " -> ";
			TRANS_GOTO(data[j].value, 0) << "\n";
		}

		out << "\t\t| _ ->\n";
	}
}

void OCamlGotoCodeGen::emitRangeBSearch( RedStateAp *state, int level, int low, int high, RedTransAp* def)
{
	/* Get the mid position, staying on the lower end of the range. */
	int mid = (low + high) >> 1;
	RedTransEl *data = state->outRange.data;

	/* Determine if we need to look higher or lower. */
	bool anyLower = mid > low;
	bool anyHigher = mid < high;

	/* Determine if the keys at mid are the limits of the alphabet. */
	bool limitLow = data[mid].lowKey == keyOps->minKey;
	bool limitHigh = data[mid].highKey == keyOps->maxKey;

	if ( anyLower && anyHigher ) {
		/* Can go lower and higher than mid. */
		out << TABS(level) << "if " << GET_WIDE_KEY(state) << " < " << 
				KEY(data[mid].lowKey) << " then begin\n";
		emitRangeBSearch( state, level+1, low, mid-1, def );
		out << TABS(level) << " end else if " << GET_WIDE_KEY(state) << " > " << 
				KEY(data[mid].highKey) << " then begin\n";
		emitRangeBSearch( state, level+1, mid+1, high, def );
		out << TABS(level) << " end else\n";
		TRANS_GOTO(data[mid].value, level+1) << "\n";
	}
	else if ( anyLower && !anyHigher ) {
		/* Can go lower than mid but not higher. */
		out << TABS(level) << "if " << GET_WIDE_KEY(state) << " < " << 
				KEY(data[mid].lowKey) << " then begin\n";
		emitRangeBSearch( state, level+1, low, mid-1, def );

		/* if the higher is the highest in the alphabet then there is no
		 * sense testing it. */
		if ( limitHigh ) {
			out << TABS(level) << " end else\n";
			TRANS_GOTO(data[mid].value, level+1) << "\n";
		}
		else {
			out << TABS(level) << " end else if " << GET_WIDE_KEY(state) << " <= " << 
					KEY(data[mid].highKey) << " then\n";
			TRANS_GOTO(data[mid].value, level+1) << "\n" << TABS(level) << "else\n";
      TRANS_GOTO(def, level+1) << "\n";
		}
	}
	else if ( !anyLower && anyHigher ) {
		/* Can go higher than mid but not lower. */
		out << TABS(level) << "if " << GET_WIDE_KEY(state) << " > " << 
				KEY(data[mid].highKey) << " then begin\n";
		emitRangeBSearch( state, level+1, mid+1, high, def );

		/* If the lower end is the lowest in the alphabet then there is no
		 * sense testing it. */
		if ( limitLow ) {
			out << TABS(level) << " end else\n";
			TRANS_GOTO(data[mid].value, level+1) << "\n";
		}
		else {
			out << TABS(level) << " end else if " << GET_WIDE_KEY(state) << " >= " << 
					KEY(data[mid].lowKey) << " then\n";
			TRANS_GOTO(data[mid].value, level+1) << "\n" << TABS(level) << "else\n";
      TRANS_GOTO(def, level+1) << "\n";
		}
	}
	else {
		/* Cannot go higher or lower than mid. It's mid or bust. What
		 * tests to do depends on limits of alphabet. */
		if ( !limitLow && !limitHigh ) {
			out << TABS(level) << "if " << KEY(data[mid].lowKey) << " <= " << 
					GET_WIDE_KEY(state) << " && " << GET_WIDE_KEY(state) << " <= " << 
					KEY(data[mid].highKey) << " then\n";
			TRANS_GOTO(data[mid].value, level+1) << "\n" << TABS(level) << "else\n";
      TRANS_GOTO(def, level+1) << "\n";
		}
		else if ( limitLow && !limitHigh ) {
			out << TABS(level) << "if " << GET_WIDE_KEY(state) << " <= " << 
					KEY(data[mid].highKey) << " then\n";
			TRANS_GOTO(data[mid].value, level+1) << "\n" << TABS(level) << "else\n";
      TRANS_GOTO(def, level+1) << "\n";
		}
		else if ( !limitLow && limitHigh ) {
			out << TABS(level) << "if " << KEY(data[mid].lowKey) << " <= " << 
					GET_WIDE_KEY(state) << " then\n";
			TRANS_GOTO(data[mid].value, level+1) << "\n" << TABS(level) << "else\n";
      TRANS_GOTO(def, level+1) << "\n";
		}
		else {
			/* Both high and low are at the limit. No tests to do. */
			TRANS_GOTO(data[mid].value, level+1) << "\n";
		}
	}
}

void OCamlGotoCodeGen::STATE_GOTO_ERROR()
{
	/* Label the state and bail immediately. */
	outLabelUsed = true;
	RedStateAp *state = redFsm->errState;
	out << "| " << state->id << " ->\n";
	out << "	do_out ()\n";
}

void OCamlGotoCodeGen::COND_TRANSLATE( GenStateCond *stateCond, int level )
{
	GenCondSpace *condSpace = stateCond->condSpace;
	out << TABS(level) << "_widec = " << CAST(WIDE_ALPH_TYPE()) << "(" <<
			KEY(condSpace->baseKey) << " + (" << GET_KEY() << 
			" - " << KEY(keyOps->minKey) << "));\n";

	for ( GenCondSet::Iter csi = condSpace->condSet; csi.lte(); csi++ ) {
		out << TABS(level) << "if ( ";
		CONDITION( out, *csi );
		Size condValOffset = ((1 << csi.pos()) * keyOps->alphSize());
		out << " ) _widec += " << condValOffset << ";\n";
	}
}

void OCamlGotoCodeGen::emitCondBSearch( RedStateAp *state, int level, int low, int high )
{
	/* Get the mid position, staying on the lower end of the range. */
	int mid = (low + high) >> 1;
	GenStateCond **data = state->stateCondVect.data;

	/* Determine if we need to look higher or lower. */
	bool anyLower = mid > low;
	bool anyHigher = mid < high;

	/* Determine if the keys at mid are the limits of the alphabet. */
	bool limitLow = data[mid]->lowKey == keyOps->minKey;
	bool limitHigh = data[mid]->highKey == keyOps->maxKey;

	if ( anyLower && anyHigher ) {
		/* Can go lower and higher than mid. */
		out << TABS(level) << "if ( " << GET_KEY() << " < " << 
				KEY(data[mid]->lowKey) << " ) {\n";
		emitCondBSearch( state, level+1, low, mid-1 );
		out << TABS(level) << "} else if ( " << GET_KEY() << " > " << 
				KEY(data[mid]->highKey) << " ) {\n";
		emitCondBSearch( state, level+1, mid+1, high );
		out << TABS(level) << "} else {\n";
		COND_TRANSLATE(data[mid], level+1);
		out << TABS(level) << "}\n";
	}
	else if ( anyLower && !anyHigher ) {
		/* Can go lower than mid but not higher. */
		out << TABS(level) << "if ( " << GET_KEY() << " < " << 
				KEY(data[mid]->lowKey) << " ) {\n";
		emitCondBSearch( state, level+1, low, mid-1 );

		/* if the higher is the highest in the alphabet then there is no
		 * sense testing it. */
		if ( limitHigh ) {
			out << TABS(level) << "} else {\n";
			COND_TRANSLATE(data[mid], level+1);
			out << TABS(level) << "}\n";
		}
		else {
			out << TABS(level) << "} else if ( " << GET_KEY() << " <= " << 
					KEY(data[mid]->highKey) << " ) {\n";
			COND_TRANSLATE(data[mid], level+1);
			out << TABS(level) << "}\n";
		}
	}
	else if ( !anyLower && anyHigher ) {
		/* Can go higher than mid but not lower. */
		out << TABS(level) << "if ( " << GET_KEY() << " > " << 
				KEY(data[mid]->highKey) << " ) {\n";
		emitCondBSearch( state, level+1, mid+1, high );

		/* If the lower end is the lowest in the alphabet then there is no
		 * sense testing it. */
		if ( limitLow ) {
			out << TABS(level) << "} else {\n";
			COND_TRANSLATE(data[mid], level+1);
			out << TABS(level) << "}\n";
		}
		else {
			out << TABS(level) << "} else if ( " << GET_KEY() << " >= " << 
					KEY(data[mid]->lowKey) << " ) {\n";
			COND_TRANSLATE(data[mid], level+1);
			out << TABS(level) << "}\n";
		}
	}
	else {
		/* Cannot go higher or lower than mid. It's mid or bust. What
		 * tests to do depends on limits of alphabet. */
		if ( !limitLow && !limitHigh ) {
			out << TABS(level) << "if ( " << KEY(data[mid]->lowKey) << " <= " << 
					GET_KEY() << " && " << GET_KEY() << " <= " << 
					KEY(data[mid]->highKey) << " ) {\n";
			COND_TRANSLATE(data[mid], level+1);
			out << TABS(level) << "}\n";
		}
		else if ( limitLow && !limitHigh ) {
			out << TABS(level) << "if ( " << GET_KEY() << " <= " << 
					KEY(data[mid]->highKey) << " ) {\n";
			COND_TRANSLATE(data[mid], level+1);
			out << TABS(level) << "}\n";
		}
		else if ( !limitLow && limitHigh ) {
			out << TABS(level) << "if ( " << KEY(data[mid]->lowKey) << " <= " << 
					GET_KEY() << " )\n {";
			COND_TRANSLATE(data[mid], level+1);
			out << TABS(level) << "}\n";
		}
		else {
			/* Both high and low are at the limit. No tests to do. */
			COND_TRANSLATE(data[mid], level);
		}
	}
}

std::ostream &OCamlGotoCodeGen::STATE_GOTOS()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st == redFsm->errState )
			STATE_GOTO_ERROR();
		else {
			/* Writing code above state gotos. */
			GOTO_HEADER( st );
      out << "\tbegin\n";

			if ( st->stateCondVect.length() > 0 ) {
				out << "	_widec = " << GET_KEY() << ";\n";
				emitCondBSearch( st, 1, 0, st->stateCondVect.length() - 1 );
			}

			/* Try singles. */
			if ( st->outSingle.length() > 0 )
				emitSingleSwitch( st );

			/* Default case is to binary search for the ranges, if that fails then */
			if ( st->outRange.length() > 0 )
				emitRangeBSearch( st, 1, 0, st->outRange.length() - 1, st->defTrans );
      else
  			/* Write the default transition. */
  			TRANS_GOTO( st->defTrans, 1 ) << "\n";

      out << "\tend\n";
		}
	}
	return out;
}

std::ostream &OCamlGotoCodeGen::TRANSITIONS()
{
	/* Emit any transitions that have functions and that go to 
	 * this state. */
	for ( TransApSet::Iter trans = redFsm->transSet; trans.lte(); trans++ ) {
		/* Write the label for the transition so it can be jumped to. */
		out << "	and tr" << trans->id << " () = ";

		/* Destination state. */
		if ( trans->action != 0 && trans->action->anyCurStateRef() )
			out << "_ps = " << vCS() << ";";
		out << vCS() << " <- " << trans->targ->id << "; ";

		if ( trans->action != 0 ) {
			/* Write out the transition func. */
			out << "f" << trans->action->actListId << " ()\n";
		}
		else {
			/* No code to execute, just loop around. */
			out << "do_again ()\n";
		}
	}
	return out;
}

std::ostream &OCamlGotoCodeGen::EXEC_FUNCS()
{
	/* Make labels that set acts and jump to execFuncs. Loop func indicies. */
	for ( GenActionTableMap::Iter redAct = redFsm->actionMap; redAct.lte(); redAct++ ) {
		if ( redAct->numTransRefs > 0 ) {
			out << "	and f" << redAct->actListId << " () = " <<
				"state.acts <- " << itoa( redAct->location+1 ) << "; "
				"execFuncs ()\n";
		}
	}

	out <<
		"\n"
		"and execFuncs () =\n"
		"	state.nacts <- " << AT( A(), POST_INCR( "state.acts") ) << ";\n"
		"	begin try while " << POST_DECR("state.nacts") << " > 0 do\n"
		"		match " << AT( A(), POST_INCR("state.acts") ) << " with\n";
		ACTION_SWITCH();
		SWITCH_DEFAULT() <<
		"	done with Goto_again -> () end;\n"
		"	do_again ()\n";
	return out;
}

unsigned int OCamlGotoCodeGen::TO_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->toStateAction != 0 )
		act = state->toStateAction->location+1;
	return act;
}

unsigned int OCamlGotoCodeGen::FROM_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->fromStateAction != 0 )
		act = state->fromStateAction->location+1;
	return act;
}

unsigned int OCamlGotoCodeGen::EOF_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->eofAction != 0 )
		act = state->eofAction->location+1;
	return act;
}

std::ostream &OCamlGotoCodeGen::TO_STATE_ACTIONS()
{
	/* Take one off for the psuedo start state. */
	int numStates = redFsm->stateList.length();
	unsigned int *vals = new unsigned int[numStates];
	memset( vals, 0, sizeof(unsigned int)*numStates );

	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ )
		vals[st->id] = TO_STATE_ACTION(st);

	out << "\t";
	for ( int st = 0; st < redFsm->nextStateId; st++ ) {
		/* Write any eof action. */
		out << vals[st];
		if ( st < numStates-1 ) {
			out << ARR_SEP();
			if ( (st+1) % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	delete[] vals;
	return out;
}

std::ostream &OCamlGotoCodeGen::FROM_STATE_ACTIONS()
{
	/* Take one off for the psuedo start state. */
	int numStates = redFsm->stateList.length();
	unsigned int *vals = new unsigned int[numStates];
	memset( vals, 0, sizeof(unsigned int)*numStates );

	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ )
		vals[st->id] = FROM_STATE_ACTION(st);

	out << "\t";
	for ( int st = 0; st < redFsm->nextStateId; st++ ) {
		/* Write any eof action. */
		out << vals[st];
		if ( st < numStates-1 ) {
			out << ARR_SEP();
			if ( (st+1) % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	delete[] vals;
	return out;
}

std::ostream &OCamlGotoCodeGen::EOF_ACTIONS()
{
	/* Take one off for the psuedo start state. */
	int numStates = redFsm->stateList.length();
	unsigned int *vals = new unsigned int[numStates];
	memset( vals, 0, sizeof(unsigned int)*numStates );

	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ )
		vals[st->id] = EOF_ACTION(st);

	out << "\t";
	for ( int st = 0; st < redFsm->nextStateId; st++ ) {
		/* Write any eof action. */
		out << vals[st];
		if ( st < numStates-1 ) {
			out << ARR_SEP();
			if ( (st+1) % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	delete[] vals;
	return out;
}

std::ostream &OCamlGotoCodeGen::FINISH_CASES()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* States that are final and have an out action need a case. */
		if ( st->eofAction != 0 ) {
			/* Write the case label. */
			out << "\t\t| " << st->id << " -> ";

			/* Write the goto func. */
			out << "f" << st->eofAction->actListId << " ()\n";
		}
	}
	
	return out;
}

void OCamlGotoCodeGen::GOTO( ostream &ret, int gotoDest, bool inFinish )
{
	ret << "begin " << vCS() << " <- " << gotoDest << "; " << 
			CTRL_FLOW() << "raise Goto_again end";
}

void OCamlGotoCodeGen::GOTO_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish )
{
	ret << "begin " << vCS() << " <- (";
	INLINE_LIST( ret, ilItem->children, 0, inFinish );
	ret << "); " << CTRL_FLOW() << "raise Goto_again end";
}

void OCamlGotoCodeGen::CURS( ostream &ret, bool inFinish )
{
	ret << "(_ps)";
}

void OCamlGotoCodeGen::TARGS( ostream &ret, bool inFinish, int targState )
{
	ret << "(" << vCS() << ")";
}

void OCamlGotoCodeGen::NEXT( ostream &ret, int nextDest, bool inFinish )
{
	ret << vCS() << " <- " << nextDest << ";";
}

void OCamlGotoCodeGen::NEXT_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish )
{
	ret << vCS() << " <- (";
	INLINE_LIST( ret, ilItem->children, 0, inFinish );
	ret << ");";
}

void OCamlGotoCodeGen::CALL( ostream &ret, int callDest, int targState, bool inFinish )
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

void OCamlGotoCodeGen::CALL_EXPR( ostream &ret, GenInlineItem *ilItem, int targState, bool inFinish )
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

void OCamlGotoCodeGen::RET( ostream &ret, bool inFinish )
{
	ret << "begin " << vCS() << " <- " << AT(STACK(), PRE_DECR(TOP()) ) << "; ";

	if ( postPopExpr != 0 ) {
		ret << "begin ";
		INLINE_LIST( ret, postPopExpr, 0, false );
		ret << "end ";
	}

	ret << CTRL_FLOW() <<  "raise Goto_again end";
}

void OCamlGotoCodeGen::BREAK( ostream &ret, int targState )
{
	outLabelUsed = true;
	ret << "begin " << P() << " <- " << P() << " + 1; " << CTRL_FLOW() << "raise Goto_out end";
}

void OCamlGotoCodeGen::writeData()
{
	if ( redFsm->anyActions() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActArrItem), A() );
		ACTIONS_ARRAY();
		CLOSE_ARRAY() <<
		"\n";
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

	STATE_IDS();

  out << "type " << TYPE_STATE() << " = { mutable acts : " << ARRAY_TYPE(redFsm->maxActionLoc) <<
         " ; mutable nacts : " << ARRAY_TYPE(redFsm->maxActArrItem) << "; }"
    << TOP_SEP();

  out << "exception Goto_again" << TOP_SEP();
}

void OCamlGotoCodeGen::writeExec()
{
	testEofUsed = false;
	outLabelUsed = false;

	out << "	begin\n";

//	if ( redFsm->anyRegCurStateRef() )
//		out << "	int _ps = 0;\n";

	if ( redFsm->anyToStateActions() || redFsm->anyRegActions() 
			|| redFsm->anyFromStateActions() )
	{
		out << "	let state = { acts = 0; nacts = 0; } in\n";
	}

//	if ( redFsm->anyConditions() )
//		out << "	" << WIDE_ALPH_TYPE() << " _widec;\n";

	out << "\n";
  out << "	let rec do_start () =\n";

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

	out <<
		"	begin match " << vCS() << " with\n";
		STATE_GOTOS();
		SWITCH_DEFAULT() <<
		"	end\n"
		"\n";
		TRANSITIONS() <<
		"\n";

	if ( redFsm->anyRegActions() )
		EXEC_FUNCS() << "\n";

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
			"	begin\n";

		if ( redFsm->anyEofTrans() ) {
			out <<
				"	match " << vCS() << " with\n";

			for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
				if ( st->eofTrans != 0 )
					out << "	| " << st->id << " -> tr" << st->eofTrans->id << " ()\n";
			}

			out << "\t| _ -> ();\n";
		}

		if ( redFsm->anyEofActions() ) {
			out <<
				"	let __acts = ref " << AT( EA(), vCS() ) << " in\n"
				"	let __nacts = ref " << AT( A(), "!__acts" ) << " in\n"
        " incr __acts;\n"
				"	begin try while !__nacts > 0 do\n"
        "   decr __nacts;\n"
				"		begin match " << AT( A(), POST_INCR("__acts.contents") ) << " with\n";
				EOF_ACTION_SWITCH();
				SWITCH_DEFAULT() <<
				"		end;\n"
				"	done with Goto_again -> do_again () end;\n";
		}

		out <<
			"	end\n"
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
