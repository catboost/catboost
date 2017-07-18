/*
 *  2007 Victor Hugo Borja <vic@rubyforge.org>
 *  Copyright 2001-2007 Adrian Thurston <thurston@complang.org>
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

#include <iomanip>
#include <sstream>
#include "redfsm.h"
#include "gendata.h"
#include "ragel.h"
#include "rubyftable.h"

using std::ostream;
using std::ostringstream;
using std::string;
using std::cerr;
using std::endl;

void RubyFTabCodeGen::GOTO( ostream &out, int gotoDest, bool inFinish )
{
	out << 
		"	begin\n"
		"		" << vCS() << " = " << gotoDest << "\n"
		"		_goto_level = _again\n"
		"		next\n"
		"	end\n";
}

void RubyFTabCodeGen::GOTO_EXPR( ostream &out, GenInlineItem *ilItem, bool inFinish )
{
	out << 
		"	begin\n"
		"		" << vCS() << " = (";
	INLINE_LIST( out, ilItem->children, 0, inFinish );
	out << ")\n";
	out <<
		"		_goto_level = _again\n"
		"		next\n"
		"	end\n";
}

void RubyFTabCodeGen::CALL( ostream &out, int callDest, int targState, bool inFinish )
{
	if ( prePushExpr != 0 ) {
		out << "begin\n";
		INLINE_LIST( out, prePushExpr, 0, false );
	}

	out <<
		"	begin\n"
		"		" << STACK() << "[" << TOP() << "] = " << vCS() << "\n"
		"		" << TOP() << "+= 1\n"
		"		" << vCS() << " = " << callDest << "\n"
		"		_goto_level = _again\n"
		"		next\n"
		"	end\n";

	if ( prePushExpr != 0 )
		out << "end\n";
}

void RubyFTabCodeGen::CALL_EXPR(ostream &out, GenInlineItem *ilItem, 
		int targState, bool inFinish )
{
	if ( prePushExpr != 0 ) {
		out << "begin\n";
		INLINE_LIST( out, prePushExpr, 0, false );
	}

	out <<
		"	begin\n"
		"		" << STACK() << "[" << TOP() << "] = " << vCS() << "\n"
		"		" << TOP() << " += 1\n"
		"		" << vCS() << " = (";
	INLINE_LIST( out, ilItem->children, targState, inFinish );
	out << ")\n";

	out << 
		"		_goto_level = _again\n"
		"		next\n"
		"	end\n";

	if ( prePushExpr != 0 )
		out << "end\n";
}

void RubyFTabCodeGen::RET( ostream &out, bool inFinish )
{
	out <<
		"	begin\n"
		"		" << TOP() << " -= 1\n"
		"		" << vCS() << " = " << STACK() << "[" << TOP() << "]\n";

	if ( postPopExpr != 0 ) {
		out << "begin\n";
		INLINE_LIST( out, postPopExpr, 0, false );
		out << "end\n";
	}

	out <<
		"		_goto_level = _again\n"
		"		next\n"
		"	end\n";
}

void RubyFTabCodeGen::BREAK( ostream &out, int targState )
{
	out << 
		"	begin\n"
		"		" << P() << " += 1\n"
		"		_goto_level = _out\n"
		"		next\n"
		"	end\n";
}


std::ostream &RubyFTabCodeGen::TO_STATE_ACTION_SWITCH()
{
	/* Loop the actions. */
	for ( GenActionTableMap::Iter redAct = redFsm->actionMap; redAct.lte(); redAct++ ) {
		if ( redAct->numToStateRefs > 0 ) {
			/* Write the entry label. */
			out << "\twhen " << redAct->actListId+1 << " then\n";

			/* Write each action in the list of action items. */
			for ( GenActionTable::Iter item = redAct->key; item.lte(); item++ )
				ACTION( out, item->value, 0, false );

		}
	}

	genLineDirective( out );
	return out;
}

/* Write out the function switch. This switch is keyed on the values
 * of the func index. */
std::ostream &RubyFTabCodeGen::FROM_STATE_ACTION_SWITCH()
{
	/* Loop the actions. */
	for ( GenActionTableMap::Iter redAct = redFsm->actionMap; redAct.lte(); redAct++ ) {
		if ( redAct->numFromStateRefs > 0 ) {
			/* Write the entry label. */
			out << "\twhen " << redAct->actListId+1 << " then\n";

			/* Write each action in the list of action items. */
			for ( GenActionTable::Iter item = redAct->key; item.lte(); item++ )
				ACTION( out, item->value, 0, false );

		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &RubyFTabCodeGen::EOF_ACTION_SWITCH()
{
	/* Loop the actions. */
	for ( GenActionTableMap::Iter redAct = redFsm->actionMap; redAct.lte(); redAct++ ) {
		if ( redAct->numEofRefs > 0 ) {
			/* Write the entry label. */
			out << "\twhen " << redAct->actListId+1 << " then\n";

			/* Write each action in the list of action items. */
			for ( GenActionTable::Iter item = redAct->key; item.lte(); item++ )
				ACTION( out, item->value, 0, true );

		}
	}

	genLineDirective( out );
	return out;
}

/* Write out the function switch. This switch is keyed on the values
 * of the func index. */
std::ostream &RubyFTabCodeGen::ACTION_SWITCH()
{
	/* Loop the actions. */
	for ( GenActionTableMap::Iter redAct = redFsm->actionMap; redAct.lte(); redAct++ ) {
		if ( redAct->numTransRefs > 0 ) {
			/* Write the entry label. */
			out << "\twhen " << redAct->actListId+1 << " then\n";

			/* Write each action in the list of action items. */
			for ( GenActionTable::Iter item = redAct->key; item.lte(); item++ )
				ACTION( out, item->value, 0, false );

		}
	}

	genLineDirective( out );
	return out;
}


int RubyFTabCodeGen::TO_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->toStateAction != 0 )
		act = state->toStateAction->actListId+1;
	return act;
}

int RubyFTabCodeGen::FROM_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->fromStateAction != 0 )
		act = state->fromStateAction->actListId+1;
	return act;
}

int RubyFTabCodeGen::EOF_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->eofAction != 0 )
		act = state->eofAction->actListId+1;
        return act;
}


/* Write out the function for a transition. */
int RubyFTabCodeGen::TRANS_ACTION( RedTransAp *trans )
{
	int action = 0;
	if ( trans->action != 0 )
		action = trans->action->actListId+1;
	return action;
}

void RubyFTabCodeGen::writeData()
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
}

void RubyFTabCodeGen::writeExec()
{
	out << 
		"begin\n"
		"	testEof = false\n"
		"	_klen, _trans, _keys";

	if ( redFsm->anyRegCurStateRef() )
		out << ", _ps";

	if ( redFsm->anyConditions() )
		out << ", _widec";

	out << " = nil\n";

	out << 
		"	_goto_level = 0\n"
		"	_resume = 10\n"
		"	_eof_trans = 15\n"
		"	_again = 20\n"
		"	_test_eof = 30\n"
		"	_out = 40\n";

	out << 
		"	while true\n"
		"	if _goto_level <= 0\n";

	if ( !noEnd ) {
		out << 
			"	if " << P() << " == " << PE() << "\n"
			"		_goto_level = _test_eof\n"
			"		next\n"
			"	end\n";
	}

	if ( redFsm->errState != 0 ) {
		out << 
			"	if " << vCS() << " == " << redFsm->errState->id << "\n"
			"		_goto_level = _out\n"
			"		next\n"
			"	end\n";
	}

	/* The resume label. */
	out << 
		"	end\n"
		"	if _goto_level <= _resume\n";
	
	if ( redFsm->anyFromStateActions() ) {
		out <<
			"	case " << FSA() << "[" << vCS() << "] \n";
			FROM_STATE_ACTION_SWITCH() <<
			"	end # from state action switch \n"
			"\n";
	}

	if ( redFsm->anyConditions() )
		COND_TRANSLATE();

	LOCATE_TRANS();

	if ( useIndicies )
		out << "	_trans = " << I() << "[_trans];\n";

	if ( redFsm->anyEofTrans() ) {
		out << 
			"	end\n"
			"	if _goto_level <= _eof_trans\n";
	}

	if ( redFsm->anyRegCurStateRef() )
		out << "	_ps = " << vCS() << ";\n";

	out <<
		"	" << vCS() << " = " << TT() << "[_trans];\n"
		"\n";

	if ( redFsm->anyRegActions() ) {
		out << 
			"	if " << TA() << "[_trans] != 0\n"
			"\n"
			"		case " << TA() << "[_trans] \n";
			ACTION_SWITCH() <<
			"		end # action switch \n"
			"	end\n"
			"\n";
	}
        
	/* The again label. */
	out <<
		"	end\n"
		"	if _goto_level <= _again\n";

	if ( redFsm->anyToStateActions() ) {
		out <<
			"	case " << TSA() << "[" << vCS() << "] \n";
			TO_STATE_ACTION_SWITCH() <<
			"	end\n"
			"\n";
	}

	if ( redFsm->errState != 0 ) {
		out << 
			"	if " << vCS() << " == " << redFsm->errState->id << "\n"
			"		_goto_level = _out\n"
			"		next\n"
			"	end\n";
	}

	out << "	" << P() << " += 1\n";

	if ( !noEnd ) {
		out << 
			"	if " << P() << " != " << PE() << "\n"
			"		_goto_level = _resume\n"
			"		next\n"
			"	end\n";
	}
	else {
		out <<
			"	_goto_level = _resume\n"
			"	next\n";
	}

	/* The test eof label. */
	out <<
		"	end\n"
		"	if _goto_level <= _test_eof\n";

	if ( redFsm->anyEofTrans() || redFsm->anyEofActions() ) {
		out <<
			"	if " << P() << " == " << vEOF() << "\n";

		if ( redFsm->anyEofTrans() ) {
			out <<
				"	if " << ET() << "[" << vCS() << "] > 0\n"
				"		_trans = " << ET() << "[" << vCS() << "] - 1;\n"
				"		_goto_level = _eof_trans\n"
				"		next;\n"
				"	end\n";
		}

		if ( redFsm->anyEofActions() ) {
			out <<
				"	begin\n"
				"		case ( " << EA() << "[" << vCS() << "] )\n";
				EOF_ACTION_SWITCH() <<
				"		end\n"
				"	end\n";
		}

		out << 
			"	end\n"
			"\n";
	}

	out << 
		"	end\n"
		"	if _goto_level <= _out\n"
		"		break\n"
		"	end\n"
		"end\n";

	/* Wrapping the execute block. */
	out << "	end\n";
}


void RubyFTabCodeGen::calcIndexSize()
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

/*
 * Local Variables:
 * mode: c++
 * indent-tabs-mode: 1
 * c-file-style: "bsd"
 * End:
 */
