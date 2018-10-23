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
#include "mlftable.h"
#include "redfsm.h"
#include "gendata.h"

/* Determine if we should use indicies or not. */
void OCamlFTabCodeGen::calcIndexSize()
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
		sizeWithInds += arrayTypeSize(redFsm->maxActListId) * redFsm->transSet.length();

	/* Calculate the cost of not using indicies. */
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		int totalIndex = st->outSingle.length() + st->outRange.length() + 
				(st->defTrans == 0 ? 0 : 1);
		sizeWithoutInds += arrayTypeSize(redFsm->maxState) * totalIndex;
		if ( redFsm->anyActions() )
			sizeWithoutInds += arrayTypeSize(redFsm->maxActListId) * totalIndex;
	}

	/* If using indicies reduces the size, use them. */
	useIndicies = sizeWithInds < sizeWithoutInds;
}

std::ostream &OCamlFTabCodeGen::TO_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->toStateAction != 0 )
		act = state->toStateAction->actListId+1;
	out << act;
	return out;
}

std::ostream &OCamlFTabCodeGen::FROM_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->fromStateAction != 0 )
		act = state->fromStateAction->actListId+1;
	out << act;
	return out;
}

std::ostream &OCamlFTabCodeGen::EOF_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->eofAction != 0 )
		act = state->eofAction->actListId+1;
	out << act;
	return out;
}


/* Write out the function for a transition. */
std::ostream &OCamlFTabCodeGen::TRANS_ACTION( RedTransAp *trans )
{
	int action = 0;
	if ( trans->action != 0 )
		action = trans->action->actListId+1;
	out << action;
	return out;
}

/* Write out the function switch. This switch is keyed on the values
 * of the func index. */
std::ostream &OCamlFTabCodeGen::TO_STATE_ACTION_SWITCH()
{
	/* Loop the actions. */
	for ( GenActionTableMap::Iter redAct = redFsm->actionMap; redAct.lte(); redAct++ ) {
		if ( redAct->numToStateRefs > 0 ) {
			/* Write the entry label. */
			out << "\t| " << redAct->actListId+1 << " ->\n";

			/* Write each action in the list of action items. */
			for ( GenActionTable::Iter item = redAct->key; item.lte(); item++ )
				ACTION( out, item->value, 0, false );

			out << "\t()\n";
		}
	}

	genLineDirective( out );
	return out;
}

/* Write out the function switch. This switch is keyed on the values
 * of the func index. */
std::ostream &OCamlFTabCodeGen::FROM_STATE_ACTION_SWITCH()
{
	/* Loop the actions. */
	for ( GenActionTableMap::Iter redAct = redFsm->actionMap; redAct.lte(); redAct++ ) {
		if ( redAct->numFromStateRefs > 0 ) {
			/* Write the entry label. */
			out << "\t| " << redAct->actListId+1 << " ->\n";

			/* Write each action in the list of action items. */
			for ( GenActionTable::Iter item = redAct->key; item.lte(); item++ )
				ACTION( out, item->value, 0, false );

			out << "\t()\n";
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &OCamlFTabCodeGen::EOF_ACTION_SWITCH()
{
	/* Loop the actions. */
	for ( GenActionTableMap::Iter redAct = redFsm->actionMap; redAct.lte(); redAct++ ) {
		if ( redAct->numEofRefs > 0 ) {
			/* Write the entry label. */
			out << "\t| " << redAct->actListId+1 << " ->\n";

			/* Write each action in the list of action items. */
			for ( GenActionTable::Iter item = redAct->key; item.lte(); item++ )
				ACTION( out, item->value, 0, true );

			out << "\t()\n";
		}
	}

	genLineDirective( out );
	return out;
}

/* Write out the function switch. This switch is keyed on the values
 * of the func index. */
std::ostream &OCamlFTabCodeGen::ACTION_SWITCH()
{
	/* Loop the actions. */
	for ( GenActionTableMap::Iter redAct = redFsm->actionMap; redAct.lte(); redAct++ ) {
		if ( redAct->numTransRefs > 0 ) {
			/* Write the entry label. */
			out << "\t| " << redAct->actListId+1 << " ->\n";

			/* Write each action in the list of action items. */
			for ( GenActionTable::Iter item = redAct->key; item.lte(); item++ )
				ACTION( out, item->value, 0, false );

			out << "\t()\n";
		}
	}

	genLineDirective( out );
	return out;
}

void OCamlFTabCodeGen::writeData()
{
	if ( redFsm->anyConditions() ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxCondOffset), CO() );
		COND_OFFSETS();
		CLOSE_ARRAY() <<
		"\n";

		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxCondLen), CL() );
		COND_LENS();
		CLOSE_ARRAY() <<
		"\n";

		OPEN_ARRAY( WIDE_ALPH_TYPE(), CK() );
		COND_KEYS();
		CLOSE_ARRAY() <<
		"\n";

		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxCondSpaceId), C() );
		COND_SPACES();
		CLOSE_ARRAY() <<
		"\n";
	}

	OPEN_ARRAY( ARRAY_TYPE(redFsm->maxKeyOffset), KO() );
	KEY_OFFSETS();
	CLOSE_ARRAY() <<
	"\n";

	OPEN_ARRAY( WIDE_ALPH_TYPE(), K() );
	KEYS();
	CLOSE_ARRAY() <<
	"\n";

	OPEN_ARRAY( ARRAY_TYPE(redFsm->maxSingleLen), SL() );
	SINGLE_LENS();
	CLOSE_ARRAY() <<
	"\n";

	OPEN_ARRAY( ARRAY_TYPE(redFsm->maxRangeLen), RL() );
	RANGE_LENS();
	CLOSE_ARRAY() <<
	"\n";

	OPEN_ARRAY( ARRAY_TYPE(redFsm->maxIndexOffset), IO() );
	INDEX_OFFSETS();
	CLOSE_ARRAY() <<
	"\n";

	if ( useIndicies ) {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxIndex), I() );
		INDICIES();
		CLOSE_ARRAY() <<
		"\n";

		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxState), TT() );
		TRANS_TARGS_WI();
		CLOSE_ARRAY() <<
		"\n";

		if ( redFsm->anyActions() ) {
			OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActListId), TA() );
			TRANS_ACTIONS_WI();
			CLOSE_ARRAY() <<
			"\n";
		}
	}
	else {
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxState), TT() );
		TRANS_TARGS();
		CLOSE_ARRAY() <<
		"\n";

		if ( redFsm->anyActions() ) {
			OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActListId), TA() );
			TRANS_ACTIONS();
			CLOSE_ARRAY() <<
			"\n";
		}
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
		OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActListId), EA() );
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

  out << "type " << TYPE_STATE() << " = { mutable keys : int; mutable trans : int; }"
    << TOP_SEP();

  out << "exception Goto_match" << TOP_SEP();
  out << "exception Goto_again" << TOP_SEP();
  out << "exception Goto_eof_trans" << TOP_SEP();
}

void OCamlFTabCodeGen::writeExec()
{
	testEofUsed = false;
	outLabelUsed = false;
	initVarTypes();

	out << 
		"	begin\n";

//	if ( redFsm->anyRegCurStateRef() )
//		out << klenType ", _ps";

  out <<
    "	let state = { keys = 0; trans = 0; } in\n"
    "	let rec do_start () =\n";

//	if ( redFsm->anyConditions() )
//		out << "	" << WIDE_ALPH_TYPE() << " _widec;\n";

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
			"	begin match " << AT( FSA(), vCS() ) << " with\n";
			FROM_STATE_ACTION_SWITCH();
			SWITCH_DEFAULT() <<
			"	end;\n"
			"\n";
	}

	if ( redFsm->anyConditions() )
		COND_TRANSLATE();

  out << "\tbegin try\n";
	LOCATE_TRANS();
  out << "\twith Goto_match -> () end;\n";

  out << 
    "\tdo_match ()\n";

	out << "and do_match () =\n";

	if ( useIndicies )
		out << "	state.trans <- " << CAST(transType) << AT( I(), "state.trans" ) << ";\n";

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
			"	begin try if " << AT( TA() , "state.trans" ) << " = 0 then\n"
			"		raise Goto_again;\n"
			"\n"
			"	match " << AT( TA(), "state.trans" ) << " with\n";
			ACTION_SWITCH();
			SWITCH_DEFAULT() <<
			"	with Goto_again -> () end;\n"
			"\n";
	}
  out << "\tdo_again ()\n";

//	if ( redFsm->anyRegActions() || redFsm->anyActionGotos() || 
//			redFsm->anyActionCalls() || redFsm->anyActionRets() )
  out << "\tand do_again () =\n";

	if ( redFsm->anyToStateActions() ) {
		out <<
			"	begin match " << AT( TSA(), vCS() ) << " with\n";
			TO_STATE_ACTION_SWITCH();
			SWITCH_DEFAULT() <<
			"	end;\n"
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
				"	begin match " << AT( EA(), vCS() ) << " with\n";
				EOF_ACTION_SWITCH();
				SWITCH_DEFAULT() <<
				"	end\n";
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

