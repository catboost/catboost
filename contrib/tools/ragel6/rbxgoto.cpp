/*
 *  Copyright 2007 Victor Hugo Borja <vic@rubyforge.org>
 *            2006-2007 Adrian Thurston <thurston@complang.org>
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

#include <stdio.h>
#include <string>

#include "rbxgoto.h"
#include "ragel.h"
#include "redfsm.h"
#include "bstmap.h"
#include "gendata.h"

using std::ostream;
using std::string;

inline string label(string a, int i)
{
	return a + itoa(i);
}

ostream &RbxGotoCodeGen::rbxLabel(ostream &out, string label)
{
	out << "Rubinius.asm { @labels[:_" << FSM_NAME() << "_" << label << "].set! }\n";
	return out;
}

ostream &RbxGotoCodeGen::rbxGoto(ostream &out, string label)
{
	out << "Rubinius.asm { goto @labels[:_" << FSM_NAME() << "_" << label << "] }\n";
	return out;
}

/* Emit the goto to take for a given transition. */
std::ostream &RbxGotoCodeGen::TRANS_GOTO( RedTransAp *trans, int level )
{
	out << TABS(level);
	return rbxGoto(out, label("tr",trans->id));
}

std::ostream &RbxGotoCodeGen::TO_STATE_ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numToStateRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << "\twhen " << act->actionId << " then\n";
			ACTION( out, act, 0, false );
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &RbxGotoCodeGen::FROM_STATE_ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numFromStateRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << "\twhen " << act->actionId << " then\n";
			ACTION( out, act, 0, false );
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &RbxGotoCodeGen::EOF_ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numEofRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << "\twhen " << act->actionId << " then\n";
			ACTION( out, act, 0, true );
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &RbxGotoCodeGen::ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numTransRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << "\twhen " << act->actionId << " then\n";
			ACTION( out, act, 0, false );
		}
	}

	genLineDirective( out );
	return out;
}

void RbxGotoCodeGen::GOTO_HEADER( RedStateAp *state )
{
	/* Label the state. */
	out << "when " << state->id << " then\n";
}


void RbxGotoCodeGen::emitSingleSwitch( RedStateAp *state )
{
	/* Load up the singles. */
	int numSingles = state->outSingle.length();
	RedTransEl *data = state->outSingle.data;

	if ( numSingles == 1 ) {
		/* If there is a single single key then write it out as an if. */
		out << "\tif " << GET_WIDE_KEY(state) << " == " << 
			KEY(data[0].lowKey) << " \n\t\t"; 

		/* Virtual function for writing the target of the transition. */
		TRANS_GOTO(data[0].value, 0) << "\n";

		out << "end\n";
	}
	else if ( numSingles > 1 ) {
		/* Write out single keys in a switch if there is more than one. */
		out << "\tcase  " << GET_WIDE_KEY(state) << "\n";

		/* Write out the single indicies. */
		for ( int j = 0; j < numSingles; j++ ) {
			out << "\t\twhen " << KEY(data[j].lowKey) << " then\n";
			TRANS_GOTO(data[j].value, 0) << "\n";
		}
		
		/* Close off the transition switch. */
		out << "\tend\n";
	}
}

void RbxGotoCodeGen::emitRangeBSearch( RedStateAp *state, int level, int low, int high )
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
			KEY(data[mid].lowKey) << " \n";
		emitRangeBSearch( state, level+1, low, mid-1 );
		out << TABS(level) << "elsif " << GET_WIDE_KEY(state) << " > " << 
			KEY(data[mid].highKey) << " \n";
		emitRangeBSearch( state, level+1, mid+1, high );
		out << TABS(level) << "else\n";
		TRANS_GOTO(data[mid].value, level+1) << "\n";
		out << TABS(level) << "end\n";
	}
	else if ( anyLower && !anyHigher ) {
		/* Can go lower than mid but not higher. */
		out << TABS(level) << "if " << GET_WIDE_KEY(state) << " < " << 
			KEY(data[mid].lowKey) << " then\n";
		emitRangeBSearch( state, level+1, low, mid-1 );

		/* if the higher is the highest in the alphabet then there is no
		 * sense testing it. */
		if ( limitHigh ) {
			out << TABS(level) << "else\n";
			TRANS_GOTO(data[mid].value, level+1) << "\n";
		}
		else {
			out << TABS(level) << "elsif" << GET_WIDE_KEY(state) << " <= " << 
				KEY(data[mid].highKey) << " )\n";
			TRANS_GOTO(data[mid].value, level+1) << "\n";
		}
		out << TABS(level) << "end\n";
	}
	else if ( !anyLower && anyHigher ) {
		/* Can go higher than mid but not lower. */
		out << TABS(level) << "if " << GET_WIDE_KEY(state) << " > " << 
			KEY(data[mid].highKey) << " \n";
		emitRangeBSearch( state, level+1, mid+1, high );

		/* If the lower end is the lowest in the alphabet then there is no
		 * sense testing it. */
		if ( limitLow ) {
			out << TABS(level) << "else\n";
			TRANS_GOTO(data[mid].value, level+1) << "\n";
		}
		else {
			out << TABS(level) << "elsif " << GET_WIDE_KEY(state) << " >= " << 
				KEY(data[mid].lowKey) << " then\n";
			TRANS_GOTO(data[mid].value, level+1) << "\n";
		}
		out << TABS(level) << "end\n";
	}
	else {
		/* Cannot go higher or lower than mid. It's mid or bust. What
		 * tests to do depends on limits of alphabet. */
		if ( !limitLow && !limitHigh ) {
			out << TABS(level) << "if " << KEY(data[mid].lowKey) << " <= " << 
				GET_WIDE_KEY(state) << " && " << GET_WIDE_KEY(state) << " <= " << 
				KEY(data[mid].highKey) << " \n";
			TRANS_GOTO(data[mid].value, level+1) << "\n";
			out << TABS(level) << "end\n";
		}
		else if ( limitLow && !limitHigh ) {
			out << TABS(level) << "if " << GET_WIDE_KEY(state) << " <= " << 
				KEY(data[mid].highKey) << " \n";
			TRANS_GOTO(data[mid].value, level+1) << "\n";
			out << TABS(level) << "end\n";
		}
		else if ( !limitLow && limitHigh ) {
			out << TABS(level) << "if " << KEY(data[mid].lowKey) << " <= " << 
				GET_WIDE_KEY(state) << " \n";
			TRANS_GOTO(data[mid].value, level+1) << "\n";
			out << TABS(level) << "end\n";
		}
		else {
			/* Both high and low are at the limit. No tests to do. */
			TRANS_GOTO(data[mid].value, level+1) << "\n";
		}
	}
}

void RbxGotoCodeGen::STATE_GOTO_ERROR()
{
	/* Label the state and bail immediately. */
	outLabelUsed = true;
	RedStateAp *state = redFsm->errState;
	out << "when " << state->id << " then\n";
	rbxGoto(out << "	", "_out") << "\n";
}

void RbxGotoCodeGen::COND_TRANSLATE( GenStateCond *stateCond, int level )
{
	GenCondSpace *condSpace = stateCond->condSpace;
	out << TABS(level) << "_widec = " <<
		KEY(condSpace->baseKey) << " + (" << GET_KEY() << 
		" - " << KEY(keyOps->minKey) << ");\n";

	for ( GenCondSet::Iter csi = condSpace->condSet; csi.lte(); csi++ ) {
		out << TABS(level) << "if ";
		CONDITION( out, *csi );
		Size condValOffset = ((1 << csi.pos()) * keyOps->alphSize());
		out << "\n _widec += " << condValOffset << ";\n end";
	}
}

void RbxGotoCodeGen::emitCondBSearch( RedStateAp *state, int level, int low, int high )
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
		out << TABS(level) << "if " << GET_KEY() << " < " << 
			KEY(data[mid]->lowKey) << " \n";
		emitCondBSearch( state, level+1, low, mid-1 );
		out << TABS(level) << "elsif " << GET_KEY() << " > " << 
			KEY(data[mid]->highKey) << " \n";
		emitCondBSearch( state, level+1, mid+1, high );
		out << TABS(level) << "else\n";
		COND_TRANSLATE(data[mid], level+1);
		out << TABS(level) << "end\n";
	}
	else if ( anyLower && !anyHigher ) {
		/* Can go lower than mid but not higher. */
		out << TABS(level) << "if " << GET_KEY() << " < " << 
			KEY(data[mid]->lowKey) << " \n";
		emitCondBSearch( state, level+1, low, mid-1 );

		/* if the higher is the highest in the alphabet then there is no
		 * sense testing it. */
		if ( limitHigh ) {
			out << TABS(level) << "else\n";
			COND_TRANSLATE(data[mid], level+1);
		}
		else {
			out << TABS(level) << "elsif " << GET_KEY() << " <= " << 
				KEY(data[mid]->highKey) << " then\n";
			COND_TRANSLATE(data[mid], level+1);
		}
		out << TABS(level) << "end\n";

	}
	else if ( !anyLower && anyHigher ) {
		/* Can go higher than mid but not lower. */
		out << TABS(level) << "if " << GET_KEY() << " > " << 
			KEY(data[mid]->highKey) << " \n";
		emitCondBSearch( state, level+1, mid+1, high );

		/* If the lower end is the lowest in the alphabet then there is no
		 * sense testing it. */
		if ( limitLow ) {
			out << TABS(level) << "else\n";
			COND_TRANSLATE(data[mid], level+1);
		}
		else {
			out << TABS(level) << "elsif " << GET_KEY() << " >= " << 
				KEY(data[mid]->lowKey) << " then\n";
			COND_TRANSLATE(data[mid], level+1);
		}
		out << TABS(level) << "end\n";
	}
	else {
		/* Cannot go higher or lower than mid. It's mid or bust. What
		 * tests to do depends on limits of alphabet. */
		if ( !limitLow && !limitHigh ) {
			out << TABS(level) << "if " << KEY(data[mid]->lowKey) << " <= " << 
				GET_KEY() << " && " << GET_KEY() << " <= " << 
				KEY(data[mid]->highKey) << " then\n";
			COND_TRANSLATE(data[mid], level+1);
			out << TABS(level) << "end\n";
		}
		else if ( limitLow && !limitHigh ) {
			out << TABS(level) << "if " << GET_KEY() << " <= " << 
				KEY(data[mid]->highKey) << " then\n";
			COND_TRANSLATE(data[mid], level+1);
			out << TABS(level) << "end\n";
		}
		else if ( !limitLow && limitHigh ) {
			out << TABS(level) << "if " << KEY(data[mid]->lowKey) << " <= " << 
				GET_KEY() << " then\n";
			COND_TRANSLATE(data[mid], level+1);
			out << TABS(level) << "end\n";
		}
		else {
			/* Both high and low are at the limit. No tests to do. */
			COND_TRANSLATE(data[mid], level);
		}
	}
}

std::ostream &RbxGotoCodeGen::STATE_GOTOS()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st == redFsm->errState )
			STATE_GOTO_ERROR();
		else {
			/* Writing code above state gotos. */
			GOTO_HEADER( st );

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
	return out;
}

std::ostream &RbxGotoCodeGen::TRANSITIONS()
{
	/* Emit any transitions that have functions and that go to 
	 * this state. */
	for ( TransApSet::Iter trans = redFsm->transSet; trans.lte(); trans++ ) {
		/* Write the label for the transition so it can be jumped to. */
		rbxLabel(out << "	", label("tr", trans->id)) << "\n";

		/* Destination state. */
		if ( trans->action != 0 && trans->action->anyCurStateRef() )
			out << "_ps = " << vCS() << "'n";
		out << vCS() << " = " << trans->targ->id << "\n";

		if ( trans->action != 0 ) {
			/* Write out the transition func. */
			rbxGoto(out, label("f", trans->action->actListId)) << "\n";
		}
		else {
			/* No code to execute, just loop around. */
			rbxGoto(out, "_again") << "\n";
		}
	}
	return out;
}

std::ostream &RbxGotoCodeGen::EXEC_FUNCS()
{
	/* Make labels that set acts and jump to execFuncs. Loop func indicies. */
	for ( GenActionTableMap::Iter redAct = redFsm->actionMap; redAct.lte(); redAct++ ) {
		if ( redAct->numTransRefs > 0 ) {
			rbxLabel(out, label("f", redAct->actListId)) << "\n" <<
				"_acts = " << itoa( redAct->location+1 ) << "\n";
			rbxGoto(out, "execFuncs") << "\n";
		}
	}

	rbxLabel(out, "execFuncs") <<
		"\n"
		"	_nacts = " << A() << "[_acts]\n"
		"	_acts += 1\n"
		"	while ( _nacts > 0 ) \n"
		"		_nacts -= 1\n"
		"		_acts += 1\n"
		"		case ( "<< A() << "[_acts-1] ) \n";
	ACTION_SWITCH();
	out <<
		"		end\n"
		"	end \n";
	rbxGoto(out, "_again");
	return out;
}

int RbxGotoCodeGen::TO_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->toStateAction != 0 )
		act = state->toStateAction->location+1;
	return act;
}

int RbxGotoCodeGen::FROM_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->fromStateAction != 0 )
		act = state->fromStateAction->location+1;
	return act;
}

int RbxGotoCodeGen::EOF_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->eofAction != 0 )
		act = state->eofAction->location+1;
	return act;
}

std::ostream &RbxGotoCodeGen::TO_STATE_ACTIONS()
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
			out << ", ";
			if ( (st+1) % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	delete[] vals;
	return out;
}

std::ostream &RbxGotoCodeGen::FROM_STATE_ACTIONS()
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
			out << ", ";
			if ( (st+1) % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	delete[] vals;
	return out;
}

std::ostream &RbxGotoCodeGen::EOF_ACTIONS()
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
			out << ", ";
			if ( (st+1) % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	delete[] vals;
	return out;
}

std::ostream &RbxGotoCodeGen::FINISH_CASES()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* States that are final and have an out action need a case. */
		if ( st->eofAction != 0 ) {
			/* Write the case label. */
			out << "\t\twhen " << st->id << " then\n";

			/* Write the goto func. */
			rbxGoto(out, label("f", st->eofAction->actListId)) << "\n";
		}
	}
	
	return out;
}

void RbxGotoCodeGen::GOTO( ostream &ret, int gotoDest, bool inFinish )
{
	ret << "begin\n" << vCS() << " = " << gotoDest << " ";
	rbxGoto(ret, "_again") << 
		"\nend\n";
}

void RbxGotoCodeGen::GOTO_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish )
{
	ret << "begin\n" << vCS() << " = (";
	INLINE_LIST( ret, ilItem->children, 0, inFinish );
	ret << ")";
	rbxGoto(ret, "_again") << 
		"\nend\n";
}

void RbxGotoCodeGen::CURS( ostream &ret, bool inFinish )
{
	ret << "(_ps)";
}

void RbxGotoCodeGen::TARGS( ostream &ret, bool inFinish, int targState )
{
	ret << "(" << vCS() << ")";
}

void RbxGotoCodeGen::NEXT( ostream &ret, int nextDest, bool inFinish )
{
	ret << vCS() << " = " << nextDest << ";";
}

void RbxGotoCodeGen::NEXT_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish )
{
	ret << vCS() << " = (";
	INLINE_LIST( ret, ilItem->children, 0, inFinish );
	ret << ");";
}

void RbxGotoCodeGen::CALL( ostream &ret, int callDest, int targState, bool inFinish )
{
	if ( prePushExpr != 0 ) {
		ret << "{";
		INLINE_LIST( ret, prePushExpr, 0, false );
	}

	ret << "begin\n" 
	    << STACK() << "[" << TOP() << "++] = " << vCS() << "; " << vCS() << " = " << 
		callDest << "; ";
	rbxGoto(ret, "_again") << 
		"\nend\n";

	if ( prePushExpr != 0 )
		ret << "}";
}

void RbxGotoCodeGen::CALL_EXPR( ostream &ret, GenInlineItem *ilItem, int targState, bool inFinish )
{
	if ( prePushExpr != 0 ) {
		ret << "{";
		INLINE_LIST( ret, prePushExpr, 0, false );
	}

	ret << "begin\n" << STACK() << "[" << TOP() << "++] = " << vCS() << "; " << vCS() << " = (";
	INLINE_LIST( ret, ilItem->children, targState, inFinish );
	ret << "); ";
	rbxGoto(ret, "_again") << 
		"\nend\n";

	if ( prePushExpr != 0 )
		ret << "}";
}

void RbxGotoCodeGen::RET( ostream &ret, bool inFinish )
{
	ret << "begin\n" << vCS() << " = " << STACK() << "[--" << TOP() << "]; " ;

	if ( postPopExpr != 0 ) {
		ret << "{";
		INLINE_LIST( ret, postPopExpr, 0, false );
		ret << "}";
	}

	rbxGoto(ret, "_again") << 
		"\nend\n";
}

void RbxGotoCodeGen::BREAK( ostream &ret, int targState )
{
	outLabelUsed = true;

	out <<
		"	begin\n"
		"		" << P() << " += 1\n"
		"		"; rbxGoto(ret, "_out") << "\n" 
		"	end\n";
}

void RbxGotoCodeGen::writeData()
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
}

void RbxGotoCodeGen::writeExec()
{
	outLabelUsed = false;

	out << "	begin\n";

	out << "	Rubinius.asm { @labels = Hash.new { |h,k| h[k] = new_label } }\n";

	if ( redFsm->anyRegCurStateRef() )
		out << "	_ps = 0;\n";

	if ( redFsm->anyToStateActions() || redFsm->anyRegActions() 
	     || redFsm->anyFromStateActions() )
	{
                out <<  " _acts, _nacts = nil\n";
	}

	if ( redFsm->anyConditions() )
		out << "        _widec = nil\n";

	out << "\n";

	if ( !noEnd ) {
		outLabelUsed = true;
		out << 
			"	if ( " << P() << " == " << PE() << " )\n";
		rbxGoto(out << "		", "_out") << "\n" <<
			"	end\n";
	}

	if ( redFsm->errState != 0 ) {
		outLabelUsed = true;
		out << 
			"	if ( " << vCS() << " == " << redFsm->errState->id << " )\n";
		rbxGoto(out << "		", "_out") << "\n" <<
			"	end\n";
	}

	rbxLabel(out, "_resume") << "\n";

	if ( redFsm->anyFromStateActions() ) {
		out <<

			"	_acts = " << ARR_OFF( A(), FSA() + "[" + vCS() + "]" ) << ";\n"
			"	_nacts = " << " *_acts++;\n"
			"	while ( _nacts-- > 0 ) {\n"
			"		switch ( *_acts++ ) {\n";
		FROM_STATE_ACTION_SWITCH();
		out <<
			"		}\n"
			"	}\n"
			"\n";
	}

	out <<
		"	case ( " << vCS() << " )\n";
	STATE_GOTOS();
	out <<
		"	end # case\n"
		"\n";
	TRANSITIONS() <<
		"\n";

	if ( redFsm->anyRegActions() )
		EXEC_FUNCS() << "\n";


	rbxLabel(out, "_again") << "\n";

	if ( redFsm->anyToStateActions() ) {
		out <<
			"	_acts = " << ARR_OFF( A(), TSA() + "[" + vCS() + "]" ) << ";\n"
			"	_nacts = " << " *_acts++;\n"
			"	while ( _nacts-- > 0 ) {\n"
			"		switch ( *_acts++ ) {\n";
		TO_STATE_ACTION_SWITCH();
		out <<
			"		}\n"
			"	}\n"
			"\n";
	}

	if ( redFsm->errState != 0 ) {
		outLabelUsed = true;
		out << 
			"	if ( " << vCS() << " == " << redFsm->errState->id << " )\n";
		rbxGoto(out << "		", "_out") << "\n" <<
			"	end" << "\n";
	}

	if ( !noEnd ) {
		out <<  "	"  << P() << " += 1\n"
			"	if ( " << P() << " != " << PE() << " )\n";
		rbxGoto(out << "		", "_resume") << "\n" <<
			"	end" << "\n";
	}
	else {
		out << 
			"	" << P() << " += 1;\n";
		rbxGoto(out << "	", "_resume") << "\n";
	}

	if ( outLabelUsed )
		rbxLabel(out, "_out") << "\n";

	out << "	end\n";
}

void RbxGotoCodeGen::writeEOF()
{
	if ( redFsm->anyEofActions() ) {
		out << 
			"	{\n"
			"	 _acts = " << 
			ARR_OFF( A(), EA() + "[" + vCS() + "]" ) << ";\n"
			"	" << " _nacts = " << " *_acts++;\n"
			"	while ( _nacts-- > 0 ) {\n"
			"		switch ( *_acts++ ) {\n";
		EOF_ACTION_SWITCH();
		out <<
			"		}\n"
			"	}\n"
			"	}\n"
			"\n";
	}
}

/*
 * Local Variables:
 * mode: c++
 * indent-tabs-mode: 1
 * c-file-style: "bsd"
 * End:
 */
