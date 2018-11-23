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
#include "mlfgoto.h"
#include "redfsm.h"
#include "gendata.h"
#include "bstmap.h"

std::ostream &OCamlFGotoCodeGen::EXEC_ACTIONS()
{
	/* Loop the actions. */
	for ( GenActionTableMap::Iter redAct = redFsm->actionMap; redAct.lte(); redAct++ ) {
		if ( redAct->numTransRefs > 0 ) {
			/* 	We are at the start of a glob, write the case. */
			out << "and f" << redAct->actListId << " () =\n";
			out << "\tbegin try\n";

			/* Write each action in the list of action items. */
			for ( GenActionTable::Iter item = redAct->key; item.lte(); item++ )
				ACTION( out, item->value, 0, false );

			out << "\twith Goto_again -> () end;\n";
			out << "\tdo_again ()\n";
		}
	}
	return out;
}

/* Write out the function switch. This switch is keyed on the values
 * of the func index. */
std::ostream &OCamlFGotoCodeGen::TO_STATE_ACTION_SWITCH()
{
	/* Loop the actions. */
	for ( GenActionTableMap::Iter redAct = redFsm->actionMap; redAct.lte(); redAct++ ) {
		if ( redAct->numToStateRefs > 0 ) {
			/* Write the entry label. */
			out << "\t|" << redAct->actListId+1 << " ->\n";

			/* Write each action in the list of action items. */
			for ( GenActionTable::Iter item = redAct->key; item.lte(); item++ )
				ACTION( out, item->value, 0, false );

//			out << "\tbreak;\n";
		}
	}

	genLineDirective( out );
	return out;
}

/* Write out the function switch. This switch is keyed on the values
 * of the func index. */
std::ostream &OCamlFGotoCodeGen::FROM_STATE_ACTION_SWITCH()
{
	/* Loop the actions. */
	for ( GenActionTableMap::Iter redAct = redFsm->actionMap; redAct.lte(); redAct++ ) {
		if ( redAct->numFromStateRefs > 0 ) {
			/* Write the entry label. */
			out << "\t| " << redAct->actListId+1 << " ->\n";

			/* Write each action in the list of action items. */
			for ( GenActionTable::Iter item = redAct->key; item.lte(); item++ )
				ACTION( out, item->value, 0, false );

//			out << "\tbreak;\n";
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &OCamlFGotoCodeGen::EOF_ACTION_SWITCH()
{
	/* Loop the actions. */
	for ( GenActionTableMap::Iter redAct = redFsm->actionMap; redAct.lte(); redAct++ ) {
		if ( redAct->numEofRefs > 0 ) {
			/* Write the entry label. */
			out << "\t| " << redAct->actListId+1 << " ->\n";

			/* Write each action in the list of action items. */
			for ( GenActionTable::Iter item = redAct->key; item.lte(); item++ )
				ACTION( out, item->value, 0, true );

//			out << "\tbreak;\n";
		}
	}

	genLineDirective( out );
	return out;
}


std::ostream &OCamlFGotoCodeGen::FINISH_CASES()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* States that are final and have an out action need a case. */
		if ( st->eofAction != 0 ) {
			/* Write the case label. */
			out << "\t\t| " << st->id << " -> ";

			/* Jump to the func. */
			out << "f" << st->eofAction->actListId << " ()\n";
		}
	}

	return out;
}

unsigned int OCamlFGotoCodeGen::TO_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->toStateAction != 0 )
		act = state->toStateAction->actListId+1;
	return act;
}

unsigned int OCamlFGotoCodeGen::FROM_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->fromStateAction != 0 )
		act = state->fromStateAction->actListId+1;
	return act;
}

unsigned int OCamlFGotoCodeGen::EOF_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->eofAction != 0 )
		act = state->eofAction->actListId+1;
	return act;
}

void OCamlFGotoCodeGen::writeData()
{
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

  out << "exception Goto_again" << TOP_SEP();
}

void OCamlFGotoCodeGen::writeExec()
{
	testEofUsed = false;
	outLabelUsed = false;

	out << "	begin\n";

	if ( redFsm->anyRegCurStateRef() )
		out << "	let _ps = ref 0 in\n";

	if ( redFsm->anyConditions() )
		out << "	let _widec : " << WIDE_ALPH_TYPE() << " = ref 0 in\n";

	out << "\n";
  out << "\tlet rec do_start () =\n";
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
			"	begin match " << AT(FSA(),vCS()) << " with\n";
			FROM_STATE_ACTION_SWITCH();
			SWITCH_DEFAULT() <<
			"	end;\n"
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
		EXEC_ACTIONS() << "\n";

	out << "\tand do_again () =\n";

	if ( redFsm->anyToStateActions() ) {
		out <<
			" begin match " << AT(TSA(), vCS()) << " with\n";
			TO_STATE_ACTION_SWITCH();
			SWITCH_DEFAULT() <<
			"	end;\n"
			"\n";
	}

	if ( redFsm->errState != 0 ) {
		outLabelUsed = true;
		out << 
			"	match " << vCS() << " with\n" <<
      "	| " << redFsm->errState->id << " -> do_out ()\n"
      "	| _ ->\n";
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

			SWITCH_DEFAULT() << ";\n"; // fall through
		}

		if ( redFsm->anyEofActions() ) {
			out <<
				"	try match " << AT(EA(), vCS()) << " with\n";
				EOF_ACTION_SWITCH();
				SWITCH_DEFAULT() <<
				"	\n"
				"	with Goto_again -> do_again () \n";
		}

		out <<
			"	end\n"
			"\n";
	}
  else
    out << "\t()\n";

	if ( outLabelUsed )
		out << "\tand do_out () = ()\n";

  out << "\tin do_start ()\n";
	out << "	end;\n";
}
