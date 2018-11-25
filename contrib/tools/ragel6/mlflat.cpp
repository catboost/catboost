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

#include <sstream>
#include "ragel.h"
#include "mlflat.h"
#include "redfsm.h"
#include "gendata.h"

std::ostream &OCamlFlatCodeGen::TO_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->toStateAction != 0 )
		act = state->toStateAction->location+1;
	out << act;
	return out;
}

std::ostream &OCamlFlatCodeGen::FROM_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->fromStateAction != 0 )
		act = state->fromStateAction->location+1;
	out << act;
	return out;
}

std::ostream &OCamlFlatCodeGen::EOF_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->eofAction != 0 )
		act = state->eofAction->location+1;
	out << act;
	return out;
}

std::ostream &OCamlFlatCodeGen::TRANS_ACTION( RedTransAp *trans )
{
	/* If there are actions, emit them. Otherwise emit zero. */
	int act = 0;
	if ( trans->action != 0 )
		act = trans->action->location+1;
	out << act;
	return out;
}

std::ostream &OCamlFlatCodeGen::TO_STATE_ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numToStateRefs > 0 ) {
			/* Write the case label, the action and the case break */
			out << "\t| " << act->actionId << " ->\n";
			ACTION( out, act, 0, false );
			out << "\t()\n";
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &OCamlFlatCodeGen::FROM_STATE_ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numFromStateRefs > 0 ) {
			/* Write the case label, the action and the case break */
			out << "\t| " << act->actionId << " ->\n";
			ACTION( out, act, 0, false );
			out << "\t()\n";
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &OCamlFlatCodeGen::EOF_ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numEofRefs > 0 ) {
			/* Write the case label, the action and the case break */
			out << "\t| " << act->actionId << " ->\n";
			ACTION( out, act, 0, true );
			out << "\t()\n";
		}
	}

	genLineDirective( out );
	return out;
}


std::ostream &OCamlFlatCodeGen::ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numTransRefs > 0 ) {
			/* Write the case label, the action and the case break */
			out << "\t| " << act->actionId << " ->\n";
			ACTION( out, act, 0, false );
			out << "\t()\n";
		}
	}

	genLineDirective( out );
	return out;
}


std::ostream &OCamlFlatCodeGen::FLAT_INDEX_OFFSET()
{
	out << "\t";
	int totalStateNum = 0, curIndOffset = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write the index offset. */
		out << curIndOffset;
		if ( !st.last() ) {
			out << ARR_SEP();
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}

		/* Move the index offset ahead. */
		if ( st->transList != 0 )
			curIndOffset += keyOps->span( st->lowKey, st->highKey );

		if ( st->defTrans != 0 )
			curIndOffset += 1;
	}
	out << "\n";
	return out;
}

std::ostream &OCamlFlatCodeGen::KEY_SPANS()
{
	out << "\t";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write singles length. */
		unsigned long long span = 0;
		if ( st->transList != 0 )
			span = keyOps->span( st->lowKey, st->highKey );
		out << span;
		if ( !st.last() ) {
			out << ARR_SEP();
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}

std::ostream &OCamlFlatCodeGen::TO_STATE_ACTIONS()
{
	out << "\t";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write any eof action. */
		TO_STATE_ACTION(st);
		if ( !st.last() ) {
			out << ARR_SEP();
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}

std::ostream &OCamlFlatCodeGen::FROM_STATE_ACTIONS()
{
	out << "\t";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write any eof action. */
		FROM_STATE_ACTION(st);
		if ( !st.last() ) {
			out << ARR_SEP();
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}

std::ostream &OCamlFlatCodeGen::EOF_ACTIONS()
{
	out << "\t";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write any eof action. */
		EOF_ACTION(st);
		if ( !st.last() ) {
			out << ARR_SEP();
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}

std::ostream &OCamlFlatCodeGen::EOF_TRANS()
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
			out << ARR_SEP();
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}


std::ostream &OCamlFlatCodeGen::COND_KEYS()
{
	out << '\t';
	int totalTrans = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Emit just cond low key and cond high key. */
		out << ALPHA_KEY( st->condLowKey ) << ARR_SEP();
		out << ALPHA_KEY( st->condHighKey ) << ARR_SEP();
		if ( ++totalTrans % IALL == 0 )
			out << "\n\t";
	}

	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	out << /*"(char) " <<*/ 0 << "\n";
	return out;
}

std::ostream &OCamlFlatCodeGen::COND_KEY_SPANS()
{
	out << "\t";
	int totalStateNum = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write singles length. */
		unsigned long long span = 0;
		if ( st->condList != 0 )
			span = keyOps->span( st->condLowKey, st->condHighKey );
		out << span;
		if ( !st.last() ) {
			out << ARR_SEP();
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}

std::ostream &OCamlFlatCodeGen::CONDS()
{
	int totalTrans = 0;
	out << '\t';
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st->condList != 0 ) {
			/* Walk the singles. */
			unsigned long long span = keyOps->span( st->condLowKey, st->condHighKey );
			for ( unsigned long long pos = 0; pos < span; pos++ ) {
				if ( st->condList[pos] != 0 )
					out << st->condList[pos]->condSpaceId + 1 << ARR_SEP();
				else
					out << "0" << ARR_SEP();
				if ( ++totalTrans % IALL == 0 )
					out << "\n\t";
			}
		}
	}

	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	out << 0 << "\n";
	return out;
}

std::ostream &OCamlFlatCodeGen::COND_INDEX_OFFSET()
{
	out << "\t";
	int totalStateNum = 0, curIndOffset = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write the index offset. */
		out << curIndOffset;
		if ( !st.last() ) {
			out << ARR_SEP();
			if ( ++totalStateNum % IALL == 0 )
				out << "\n\t";
		}

		/* Move the index offset ahead. */
		if ( st->condList != 0 )
			curIndOffset += keyOps->span( st->condLowKey, st->condHighKey );
	}
	out << "\n";
	return out;
}


std::ostream &OCamlFlatCodeGen::KEYS()
{
	out << '\t';
	int totalTrans = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Emit just low key and high key. */
		out << ALPHA_KEY( st->lowKey ) << ARR_SEP();
		out << ALPHA_KEY( st->highKey ) << ARR_SEP();
		if ( ++totalTrans % IALL == 0 )
			out << "\n\t";
	}

	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	out << /*"(char) " <<*/ 0 << "\n";
	return out;
}

std::ostream &OCamlFlatCodeGen::INDICIES()
{
	int totalTrans = 0;
	out << '\t';
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st->transList != 0 ) {
			/* Walk the singles. */
			unsigned long long span = keyOps->span( st->lowKey, st->highKey );
			for ( unsigned long long pos = 0; pos < span; pos++ ) {
				out << st->transList[pos]->id << ARR_SEP();
				if ( ++totalTrans % IALL == 0 )
					out << "\n\t";
			}
		}

		/* The state's default index goes next. */
		if ( st->defTrans != 0 )
			out << st->defTrans->id << ARR_SEP();

		if ( ++totalTrans % IALL == 0 )
			out << "\n\t";
	}

	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	out << 0 << "\n";
	return out;
}

std::ostream &OCamlFlatCodeGen::TRANS_TARGS()
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
			out << ARR_SEP();
			if ( ++totalStates % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	delete[] transPtrs;
	return out;
}


std::ostream &OCamlFlatCodeGen::TRANS_ACTIONS()
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
			out << ARR_SEP();
			if ( ++totalAct % IALL == 0 )
				out << "\n\t";
		}
	}
	out << "\n";
	delete[] transPtrs;
	return out;
}

void OCamlFlatCodeGen::LOCATE_TRANS()
{
  std::ostringstream temp;
  temp << "inds + (\n"
		"		if slen > 0 && " << AT( K(), "keys" ) << " <= " << GET_WIDE_KEY() << " &&\n"
		"		" << GET_WIDE_KEY() << " <= " << AT( K(), "keys+1" ) << " then\n"
		"		" << GET_WIDE_KEY() << " - " << AT(K(), "keys" ) << " else slen)";
	out <<
		"	let keys = " << vCS() << " lsl 1 in\n"
		"	let inds = " << AT( IO(), vCS() ) << " in\n"
		"\n"
		"	let slen = " << AT( SP(), vCS() ) << " in\n"
		"	state.trans <- " << AT( I(), temp.str() ) << ";\n"
		"\n";
}

void OCamlFlatCodeGen::GOTO( ostream &ret, int gotoDest, bool inFinish )
{
	ret << "begin " << vCS() << " <- " << gotoDest << "; " << 
			CTRL_FLOW() << "raise Goto_again end";
}

void OCamlFlatCodeGen::GOTO_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish )
{
	ret << "begin " << vCS() << " <- (";
	INLINE_LIST( ret, ilItem->children, 0, inFinish );
	ret << "); " << CTRL_FLOW() << " raise Goto_again end";
}

void OCamlFlatCodeGen::CURS( ostream &ret, bool inFinish )
{
	ret << "(_ps)";
}

void OCamlFlatCodeGen::TARGS( ostream &ret, bool inFinish, int targState )
{
	ret << "(" << vCS() << ")";
}

void OCamlFlatCodeGen::NEXT( ostream &ret, int nextDest, bool inFinish )
{
	ret << vCS() << " <- " << nextDest << ";";
}

void OCamlFlatCodeGen::NEXT_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish )
{
	ret << vCS() << " <- (";
	INLINE_LIST( ret, ilItem->children, 0, inFinish );
	ret << ");";
}

void OCamlFlatCodeGen::CALL( ostream &ret, int callDest, int targState, bool inFinish )
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

void OCamlFlatCodeGen::CALL_EXPR( ostream &ret, GenInlineItem *ilItem, int targState, bool inFinish )
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

void OCamlFlatCodeGen::RET( ostream &ret, bool inFinish )
{
	ret << "begin " << vCS() << " <- " << AT(STACK(), PRE_DECR(TOP()) ) << "; ";

	if ( postPopExpr != 0 ) {
		ret << "begin ";
		INLINE_LIST( ret, postPopExpr, 0, false );
		ret << "end ";
	}

	ret << CTRL_FLOW() <<  "raise Goto_again end";
}

void OCamlFlatCodeGen::BREAK( ostream &ret, int targState )
{
	outLabelUsed = true;
	ret << "begin " << P() << " <- " << P() << " + 1; " << CTRL_FLOW() << "raise Goto_out end";
}

void OCamlFlatCodeGen::writeData()
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
		OPEN_ARRAY( WIDE_ALPH_TYPE(), CK() );
		COND_KEYS();
		CLOSE_ARRAY() <<
		"\n";

		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxCondSpan), CSP() );
		COND_KEY_SPANS();
		CLOSE_ARRAY() <<
		"\n";

		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxCond), C() );
		CONDS();
		CLOSE_ARRAY() <<
		"\n";

		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxCondIndexOffset), CO() );
		COND_INDEX_OFFSET();
		CLOSE_ARRAY() <<
		"\n";
	}

	OPEN_ARRAY( WIDE_ALPH_TYPE(), K() );
	KEYS();
	CLOSE_ARRAY() <<
	"\n";

	OPEN_ARRAY( ARRAY_TYPE(redFsm->maxSpan), SP() );
	KEY_SPANS();
	CLOSE_ARRAY() <<
	"\n";

	OPEN_ARRAY( ARRAY_TYPE(redFsm->maxFlatIndexOffset), IO() );
	FLAT_INDEX_OFFSET();
	CLOSE_ARRAY() <<
	"\n";

	OPEN_ARRAY( ARRAY_TYPE(redFsm->maxIndex), I() );
	INDICIES();
	CLOSE_ARRAY() <<
	"\n";

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

  out << "type " << TYPE_STATE() << " = { mutable trans : int; mutable acts : int; mutable nacts : int; }"
    << TOP_SEP();

  out << "exception Goto_match" << TOP_SEP();
  out << "exception Goto_again" << TOP_SEP();
  out << "exception Goto_eof_trans" << TOP_SEP();
}

void OCamlFlatCodeGen::COND_TRANSLATE()
{
	out << 
		"	_widec = " << GET_KEY() << ";\n";

	out <<
		"   _keys = " << vCS() << "<<1;\n"
		"   _conds = " << CO() << "[" << vCS() << "];\n"
//		"	_keys = " << ARR_OFF( CK(), "(" + vCS() + "<<1)" ) << ";\n"
//		"	_conds = " << ARR_OFF( C(), CO() + "[" + vCS() + "]" ) << ";\n"
		"\n"
		"	_slen = " << CSP() << "[" << vCS() << "];\n"
		"	if (_slen > 0 && " << CK() << "[_keys] <=" 
			<< GET_WIDE_KEY() << " &&\n"
		"		" << GET_WIDE_KEY() << " <= " << CK() << "[_keys+1])\n"
		"		_cond = " << C() << "[_conds+" << GET_WIDE_KEY() << " - " << 
			CK() << "[_keys]];\n"
		"	else\n"
		"		_cond = 0;"
		"\n";
	/*  XXX This version of the code doesn't work because Mono is weird.  Works
	 *  fine in Microsoft's csc, even though the bug report filed claimed it
	 *  didn't.
		"	_slen = " << CSP() << "[" << vCS() << "];\n"
		"	_cond = _slen > 0 && " << CK() << "[_keys] <=" 
			<< GET_WIDE_KEY() << " &&\n"
		"		" << GET_WIDE_KEY() << " <= " << CK() << "[_keys+1] ?\n"
		"		" << C() << "[_conds+" << GET_WIDE_KEY() << " - " << CK() 
			<< "[_keys]] : 0;\n"
		"\n";
		*/
	out <<
		"	switch ( _cond ) {\n";
	for ( CondSpaceList::Iter csi = condSpaceList; csi.lte(); csi++ ) {
		GenCondSpace *condSpace = csi;
		out << "	case " << condSpace->condSpaceId + 1 << ": {\n";
		out << TABS(2) << "_widec = " << CAST(WIDE_ALPH_TYPE()) << "(" <<
				KEY(condSpace->baseKey) << " + (" << GET_KEY() << 
				" - " << KEY(keyOps->minKey) << "));\n";

		for ( GenCondSet::Iter csi = condSpace->condSet; csi.lte(); csi++ ) {
			out << TABS(2) << "if ( ";
			CONDITION( out, *csi );
			Size condValOffset = ((1 << csi.pos()) * keyOps->alphSize());
			out << " ) _widec += " << condValOffset << ";\n";
		}

		out << "		}\n";
		out << "		break;\n";
	}

	SWITCH_DEFAULT();

	out <<
		"	}\n";
}

void OCamlFlatCodeGen::writeExec()
{
	testEofUsed = false;
	outLabelUsed = false;
	initVarTypes();

	out << 
		"	begin\n";
//		"	" << slenType << " _slen";

//	if ( redFsm->anyRegCurStateRef() )
//		out << ", _ps";

//	out << 
//		"	" << transType << " _trans";

//	if ( redFsm->anyConditions() )
//		out << ", _cond";
//	out << ";\n";

//	if ( redFsm->anyToStateActions() || 
//			redFsm->anyRegActions() || redFsm->anyFromStateActions() )
//	{
//		out << 
//			"	int _acts;\n"
//			"	int _nacts;\n"; 
//	}

//	out <<
//		"	" << "int _keys;\n"
//		"	" << indsType << " _inds;\n";
		/*
		"	" << PTR_CONST() << WIDE_ALPH_TYPE() << POINTER() << "_keys;\n"
		"	" << PTR_CONST() << ARRAY_TYPE(redFsm->maxIndex) << POINTER() << "_inds;\n";*/

	if ( redFsm->anyConditions() ) {
		out << 
			"	" << condsType << " _conds;\n"
			"	" << WIDE_ALPH_TYPE() << " _widec;\n";
	}

	out << "\n";

  out <<
    "	let state = { trans = 0; acts = 0; nacts = 0; } in\n"
    "	let rec do_start () =\n";

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

	if ( redFsm->anyConditions() )
		COND_TRANSLATE();

//  out << "\tbegin try\n";
	LOCATE_TRANS();
//  out << "\twith Goto_match -> () end;\n";

  out << "\tdo_eof_trans ()\n";

//	if ( redFsm->anyEofTrans() )
  out << "and do_eof_trans () =\n";

	if ( redFsm->anyRegCurStateRef() )
		out << "	let ps = " << vCS() << " in\n";

	out <<
		"	" << vCS() << " <- " << AT( TT() ,"state.trans" ) << ";\n"
		"\n";

	if ( redFsm->anyRegActions() ) {
		out <<
			"\tbegin try\n"
      "	match " << AT( TA(), "state.trans" ) << " with\n"
			"\t| 0 -> raise Goto_again\n"
      "\t| _ ->\n"
			"	state.acts <- " << AT( TA(), "state.trans" ) << ";\n"
			"	state.nacts <- " << AT( A(), POST_INCR("state.acts") ) << ";\n"
			"	while " << POST_DECR("state.nacts") << " > 0 do\n"
			"		begin match " << AT( A(), POST_INCR("state.acts") ) << " with\n";
			ACTION_SWITCH();
			SWITCH_DEFAULT() <<
			"		end;\n"
			"	done\n"
      "\twith Goto_again -> () end;\n";
	}
  out << "\tdo_again ()\n";

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
			"	begin try\n";

		if ( redFsm->anyEofTrans() ) {
			out <<
				"	if " << AT( ET(), vCS() ) << " > 0 then\n"
				"	begin\n"
        "   state.trans <- " << CAST(transType) << "(" << AT( ET(), vCS() ) << " - 1);\n"
				"		raise Goto_eof_trans;\n"
				"	end;\n";
		}

		if ( redFsm->anyEofActions() ) {
			out <<
				"	let __acts = ref " << AT( EA(), vCS() ) << " in\n"
				"	let __nacts = ref " << AT( A(), "!__acts" ) << " in\n"
        " incr __acts;\n"
				"	while !__nacts > 0 do\n"
        "   decr __nacts;\n"
				"		begin match " << AT( A(), POST_INCR("__acts.contents") ) << " with\n";
				EOF_ACTION_SWITCH();
				SWITCH_DEFAULT() <<
				"		end;\n"
				"	done\n";
		}

		out << 
			"	with Goto_again -> do_again ()\n"
			"	| Goto_eof_trans -> do_eof_trans () end\n"
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

void OCamlFlatCodeGen::initVarTypes()
{
	slenType = ARRAY_TYPE(MAX(redFsm->maxSpan, redFsm->maxCondSpan));
	transType = ARRAY_TYPE(redFsm->maxIndex+1);
	indsType = ARRAY_TYPE(redFsm->maxFlatIndexOffset);
	condsType = ARRAY_TYPE(redFsm->maxCondIndexOffset);
}
