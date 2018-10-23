/*
 *  Copyright 2006-2007 Adrian Thurston <thurston@complang.org>
 *            2007 Colin Fleming <colin.fleming@caverock.com>
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
#include "javacodegen.h"
#include "redfsm.h"
#include "gendata.h"
#include <iomanip>
#include <sstream>

/* Integer array line length. */
#define IALL 12

/* Static array initialization item count 
 * (should be multiple of IALL). */
#define SAIIC 8184

#define _resume    1
#define _again     2
#define _eof_trans 3
#define _test_eof  4
#define _out       5

using std::setw;
using std::ios;
using std::ostringstream;
using std::string;
using std::cerr;

using std::istream;
using std::ifstream;
using std::ostream;
using std::ios;
using std::cin;
using std::cout;
using std::cerr;
using std::endl;
using std::setiosflags;

void javaLineDirective( ostream &out, const char *fileName, int line )
{
	/* Write the preprocessor line info for to the input file. */
	out << "// line " << line  << " \"";
	for ( const char *pc = fileName; *pc != 0; pc++ ) {
		if ( *pc == '\\' )
			out << "\\\\";
		else
			out << *pc;
	}
	out << "\"\n";
}

void JavaTabCodeGen::genLineDirective( ostream &out )
{
	std::streambuf *sbuf = out.rdbuf();
	output_filter *filter = static_cast<output_filter*>(sbuf);
	javaLineDirective( out, filter->fileName, filter->line + 1 );
}

void JavaTabCodeGen::GOTO( ostream &ret, int gotoDest, bool inFinish )
{
	ret << "{" << vCS() << " = " << gotoDest << "; _goto_targ = " << _again << "; " << 
			CTRL_FLOW() << "continue _goto;}";
}

void JavaTabCodeGen::GOTO_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish )
{
	ret << "{" << vCS() << " = (";
	INLINE_LIST( ret, ilItem->children, 0, inFinish );
	ret << "); _goto_targ = " << _again << "; " << CTRL_FLOW() << "continue _goto;}";
}

void JavaTabCodeGen::CALL( ostream &ret, int callDest, int targState, bool inFinish )
{
	if ( prePushExpr != 0 ) {
		ret << "{";
		INLINE_LIST( ret, prePushExpr, 0, false );
	}

	ret << "{" << STACK() << "[" << TOP() << "++] = " << vCS() << "; " << vCS() << " = " << 
			callDest << "; _goto_targ = " << _again << "; " << CTRL_FLOW() << "continue _goto;}";

	if ( prePushExpr != 0 )
		ret << "}";
}

void JavaTabCodeGen::CALL_EXPR( ostream &ret, GenInlineItem *ilItem, int targState, bool inFinish )
{
	if ( prePushExpr != 0 ) {
		ret << "{";
		INLINE_LIST( ret, prePushExpr, 0, false );
	}

	ret << "{" << STACK() << "[" << TOP() << "++] = " << vCS() << "; " << vCS() << " = (";
	INLINE_LIST( ret, ilItem->children, targState, inFinish );
	ret << "); _goto_targ = " << _again << "; " << CTRL_FLOW() << "continue _goto;}";

	if ( prePushExpr != 0 )
		ret << "}";
}

void JavaTabCodeGen::RET( ostream &ret, bool inFinish )
{
	ret << "{" << vCS() << " = " << STACK() << "[--" << TOP() << "];";

	if ( postPopExpr != 0 ) {
		ret << "{";
		INLINE_LIST( ret, postPopExpr, 0, false );
		ret << "}";
	}

	ret << "_goto_targ = " << _again << "; " << CTRL_FLOW() << "continue _goto;}";
}

void JavaTabCodeGen::BREAK( ostream &ret, int targState )
{
	ret << "{ " << P() << " += 1; _goto_targ = " << _out << "; " << 
			CTRL_FLOW() << " continue _goto;}";
}

void JavaTabCodeGen::NEXT( ostream &ret, int nextDest, bool inFinish )
{
	ret << vCS() << " = " << nextDest << ";";
}

void JavaTabCodeGen::NEXT_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish )
{
	ret << vCS() << " = (";
	INLINE_LIST( ret, ilItem->children, 0, inFinish );
	ret << ");";
}

void JavaTabCodeGen::EXEC( ostream &ret, GenInlineItem *item, int targState, int inFinish )
{
	/* The parser gives fexec two children. The double brackets are for D
	 * code. If the inline list is a single word it will get interpreted as a
	 * C-style cast by the D compiler. */
	ret << "{" << P() << " = ((";
	INLINE_LIST( ret, item->children, targState, inFinish );
	ret << "))-1;}";
}

/* Write out an inline tree structure. Walks the list and possibly calls out
 * to virtual functions than handle language specific items in the tree. */
void JavaTabCodeGen::INLINE_LIST( ostream &ret, GenInlineList *inlineList, 
		int targState, bool inFinish )
{
	for ( GenInlineList::Iter item = *inlineList; item.lte(); item++ ) {
		switch ( item->type ) {
		case GenInlineItem::Text:
			ret << item->data;
			break;
		case GenInlineItem::Goto:
			GOTO( ret, item->targState->id, inFinish );
			break;
		case GenInlineItem::Call:
			CALL( ret, item->targState->id, targState, inFinish );
			break;
		case GenInlineItem::Next:
			NEXT( ret, item->targState->id, inFinish );
			break;
		case GenInlineItem::Ret:
			RET( ret, inFinish );
			break;
		case GenInlineItem::PChar:
			ret << P();
			break;
		case GenInlineItem::Char:
			ret << GET_KEY();
			break;
		case GenInlineItem::Hold:
			ret << P() << "--;";
			break;
		case GenInlineItem::Exec:
			EXEC( ret, item, targState, inFinish );
			break;
		case GenInlineItem::Curs:
			ret << "(_ps)";
			break;
		case GenInlineItem::Targs:
			ret << "(" << vCS() << ")";
			break;
		case GenInlineItem::Entry:
			ret << item->targState->id;
			break;
		case GenInlineItem::GotoExpr:
			GOTO_EXPR( ret, item, inFinish );
			break;
		case GenInlineItem::CallExpr:
			CALL_EXPR( ret, item, targState, inFinish );
			break;
		case GenInlineItem::NextExpr:
			NEXT_EXPR( ret, item, inFinish );
			break;
		case GenInlineItem::LmSwitch:
			LM_SWITCH( ret, item, targState, inFinish );
			break;
		case GenInlineItem::LmSetActId:
			SET_ACT( ret, item );
			break;
		case GenInlineItem::LmSetTokEnd:
			SET_TOKEND( ret, item );
			break;
		case GenInlineItem::LmGetTokEnd:
			GET_TOKEND( ret, item );
			break;
		case GenInlineItem::LmInitTokStart:
			INIT_TOKSTART( ret, item );
			break;
		case GenInlineItem::LmInitAct:
			INIT_ACT( ret, item );
			break;
		case GenInlineItem::LmSetTokStart:
			SET_TOKSTART( ret, item );
			break;
		case GenInlineItem::SubAction:
			SUB_ACTION( ret, item, targState, inFinish );
			break;
		case GenInlineItem::Break:
			BREAK( ret, targState );
			break;
		}
	}
}

string JavaTabCodeGen::DATA_PREFIX()
{
	if ( !noPrefix )
		return FSM_NAME() + "_";
	return "";
}

/* Emit the alphabet data type. */
string JavaTabCodeGen::ALPH_TYPE()
{
	string ret = keyOps->alphType->data1;
	if ( keyOps->alphType->data2 != 0 ) {
		ret += " ";
		ret += + keyOps->alphType->data2;
	}
	return ret;
}

/* Emit the alphabet data type. */
string JavaTabCodeGen::WIDE_ALPH_TYPE()
{
	string ret;
	if ( redFsm->maxKey <= keyOps->maxKey )
		ret = ALPH_TYPE();
	else {
		long long maxKeyVal = redFsm->maxKey.getLongLong();
		HostType *wideType = keyOps->typeSubsumes( keyOps->isSigned, maxKeyVal );
		assert( wideType != 0 );

		ret = wideType->data1;
		if ( wideType->data2 != 0 ) {
			ret += " ";
			ret += wideType->data2;
		}
	}
	return ret;
}



void JavaTabCodeGen::COND_TRANSLATE()
{
	out << 
		"	_widec = " << GET_KEY() << ";\n"
		"	_keys = " << CO() << "[" << vCS() << "]*2\n;"
		"	_klen = " << CL() << "[" << vCS() << "];\n"
		"	if ( _klen > 0 ) {\n"
		"		int _lower = _keys\n;"
		"		int _mid;\n"
		"		int _upper = _keys + (_klen<<1) - 2;\n"
		"		while (true) {\n"
		"			if ( _upper < _lower )\n"
		"				break;\n"
		"\n"
		"			_mid = _lower + (((_upper-_lower) >> 1) & ~1);\n"
		"			if ( " << GET_WIDE_KEY() << " < " << CK() << "[_mid] )\n"
		"				_upper = _mid - 2;\n"
		"			else if ( " << GET_WIDE_KEY() << " > " << CK() << "[_mid+1] )\n"
		"				_lower = _mid + 2;\n"
		"			else {\n"
		"				switch ( " << C() << "[" << CO() << "[" << vCS() << "]"
							" + ((_mid - _keys)>>1)] ) {\n"
		;

	for ( CondSpaceList::Iter csi = condSpaceList; csi.lte(); csi++ ) {
		GenCondSpace *condSpace = csi;
		out << "	case " << condSpace->condSpaceId << ": {\n";
		out << TABS(2) << "_widec = " << KEY(condSpace->baseKey) << 
				" + (" << GET_KEY() << " - " << KEY(keyOps->minKey) << ");\n";

		for ( GenCondSet::Iter csi = condSpace->condSet; csi.lte(); csi++ ) {
			out << TABS(2) << "if ( ";
			CONDITION( out, *csi );
			Size condValOffset = ((1 << csi.pos()) * keyOps->alphSize());
			out << " ) _widec += " << condValOffset << ";\n";
		}

		out << 
			"		break;\n"
			"	}\n";
	}

	out << 
		"				}\n"
		"				break;\n"
		"			}\n"
		"		}\n"
		"	}\n"
		"\n";
}


void JavaTabCodeGen::LOCATE_TRANS()
{
	out <<
		"	_match: do {\n"
		"	_keys = " << KO() << "[" << vCS() << "]" << ";\n"
		"	_trans = " << IO() << "[" << vCS() << "];\n"
		"	_klen = " << SL() << "[" << vCS() << "];\n"
		"	if ( _klen > 0 ) {\n"
		"		int _lower = _keys;\n"
		"		int _mid;\n"
		"		int _upper = _keys + _klen - 1;\n"
		"		while (true) {\n"
		"			if ( _upper < _lower )\n"
		"				break;\n"
		"\n"
		"			_mid = _lower + ((_upper-_lower) >> 1);\n"
		"			if ( " << GET_WIDE_KEY() << " < " << K() << "[_mid] )\n"
		"				_upper = _mid - 1;\n"
		"			else if ( " << GET_WIDE_KEY() << " > " << K() << "[_mid] )\n"
		"				_lower = _mid + 1;\n"
		"			else {\n"
		"				_trans += (_mid - _keys);\n"
		"				break _match;\n"
		"			}\n"
		"		}\n"
		"		_keys += _klen;\n"
		"		_trans += _klen;\n"
		"	}\n"
		"\n"
		"	_klen = " << RL() << "[" << vCS() << "];\n"
		"	if ( _klen > 0 ) {\n"
		"		int _lower = _keys;\n"
		"		int _mid;\n"
		"		int _upper = _keys + (_klen<<1) - 2;\n"
		"		while (true) {\n"
		"			if ( _upper < _lower )\n"
		"				break;\n"
		"\n"
		"			_mid = _lower + (((_upper-_lower) >> 1) & ~1);\n"
		"			if ( " << GET_WIDE_KEY() << " < " << K() << "[_mid] )\n"
		"				_upper = _mid - 2;\n"
		"			else if ( " << GET_WIDE_KEY() << " > " << K() << "[_mid+1] )\n"
		"				_lower = _mid + 2;\n"
		"			else {\n"
		"				_trans += ((_mid - _keys)>>1);\n"
		"				break _match;\n"
		"			}\n"
		"		}\n"
		"		_trans += _klen;\n"
		"	}\n"
		"	} while (false);\n"
		"\n";
}

/* Determine if we should use indicies or not. */
void JavaTabCodeGen::calcIndexSize()
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
		sizeWithInds += arrayTypeSize(redFsm->maxActionLoc) * redFsm->transSet.length();

	/* Calculate the cost of not using indicies. */
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		int totalIndex = st->outSingle.length() + st->outRange.length() + 
				(st->defTrans == 0 ? 0 : 1);
		sizeWithoutInds += arrayTypeSize(redFsm->maxState) * totalIndex;
		if ( redFsm->anyActions() )
			sizeWithoutInds += arrayTypeSize(redFsm->maxActionLoc) * totalIndex;
	}

	/* If using indicies reduces the size, use them. */
	useIndicies = sizeWithInds < sizeWithoutInds;
}

int JavaTabCodeGen::TO_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->toStateAction != 0 )
		act = state->toStateAction->location+1;
	return act;
}

int JavaTabCodeGen::FROM_STATE_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->fromStateAction != 0 )
		act = state->fromStateAction->location+1;
	return act;
}

int JavaTabCodeGen::EOF_ACTION( RedStateAp *state )
{
	int act = 0;
	if ( state->eofAction != 0 )
		act = state->eofAction->location+1;
	return act;
}


int JavaTabCodeGen::TRANS_ACTION( RedTransAp *trans )
{
	/* If there are actions, emit them. Otherwise emit zero. */
	int act = 0;
	if ( trans->action != 0 )
		act = trans->action->location+1;
	return act;
}

std::ostream &JavaTabCodeGen::TO_STATE_ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numToStateRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << "\tcase " << act->actionId << ":\n";
			ACTION( out, act, 0, false );
			out << "\tbreak;\n";
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &JavaTabCodeGen::FROM_STATE_ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numFromStateRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << "\tcase " << act->actionId << ":\n";
			ACTION( out, act, 0, false );
			out << "\tbreak;\n";
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &JavaTabCodeGen::EOF_ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numEofRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << "\tcase " << act->actionId << ":\n";
			ACTION( out, act, 0, true );
			out << "\tbreak;\n";
		}
	}

	genLineDirective( out );
	return out;
}


std::ostream &JavaTabCodeGen::ACTION_SWITCH()
{
	/* Walk the list of functions, printing the cases. */
	for ( GenActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Write out referenced actions. */
		if ( act->numTransRefs > 0 ) {
			/* Write the case label, the action and the case break. */
			out << "\tcase " << act->actionId << ":\n";
			ACTION( out, act, 0, false );
			out << "\tbreak;\n";
		}
	}

	genLineDirective( out );
	return out;
}

std::ostream &JavaTabCodeGen::COND_OFFSETS()
{
	int curKeyOffset = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write the key offset. */
		ARRAY_ITEM( INT(curKeyOffset), st.last() );

		/* Move the key offset ahead. */
		curKeyOffset += st->stateCondList.length();
	}
	return out;
}

std::ostream &JavaTabCodeGen::KEY_OFFSETS()
{
	int curKeyOffset = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write the key offset. */
		ARRAY_ITEM( INT(curKeyOffset), st.last() );

		/* Move the key offset ahead. */
		curKeyOffset += st->outSingle.length() + st->outRange.length()*2;
	}
	return out;
}


std::ostream &JavaTabCodeGen::INDEX_OFFSETS()
{
	int curIndOffset = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write the index offset. */
		ARRAY_ITEM( INT(curIndOffset), st.last() );

		/* Move the index offset ahead. */
		curIndOffset += st->outSingle.length() + st->outRange.length();
		if ( st->defTrans != 0 )
			curIndOffset += 1;
	}
	return out;
}

std::ostream &JavaTabCodeGen::COND_LENS()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write singles length. */
		ARRAY_ITEM( INT(st->stateCondList.length()), st.last() );
	}
	return out;
}


std::ostream &JavaTabCodeGen::SINGLE_LENS()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write singles length. */
		ARRAY_ITEM( INT(st->outSingle.length()), st.last() );
	}
	return out;
}

std::ostream &JavaTabCodeGen::RANGE_LENS()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Emit length of range index. */
		ARRAY_ITEM( INT(st->outRange.length()), st.last() );
	}
	return out;
}

std::ostream &JavaTabCodeGen::TO_STATE_ACTIONS()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write any eof action. */
		ARRAY_ITEM( INT(TO_STATE_ACTION(st)), st.last() );
	}
	return out;
}

std::ostream &JavaTabCodeGen::FROM_STATE_ACTIONS()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write any eof action. */
		ARRAY_ITEM( INT(FROM_STATE_ACTION(st)), st.last() );
	}
	return out;
}

std::ostream &JavaTabCodeGen::EOF_ACTIONS()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write any eof action. */
		ARRAY_ITEM( INT(EOF_ACTION(st)), st.last() );
	}
	return out;
}

std::ostream &JavaTabCodeGen::EOF_TRANS()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Write any eof action. */
		long trans = 0;
		if ( st->eofTrans != 0 ) {
			assert( st->eofTrans->pos >= 0 );
			trans = st->eofTrans->pos+1;
		}

		/* Write any eof action. */
		ARRAY_ITEM( INT(trans), st.last() );
	}
	return out;
}


std::ostream &JavaTabCodeGen::COND_KEYS()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Loop the state's transitions. */
		for ( GenStateCondList::Iter sc = st->stateCondList; sc.lte(); sc++ ) {
			/* Lower key. */
			ARRAY_ITEM( KEY( sc->lowKey ), false );
			ARRAY_ITEM( KEY( sc->highKey ), false );
		}
	}

	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	ARRAY_ITEM( INT(0), true );
	return out;
}

std::ostream &JavaTabCodeGen::COND_SPACES()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Loop the state's transitions. */
		for ( GenStateCondList::Iter sc = st->stateCondList; sc.lte(); sc++ ) {
			/* Cond Space id. */
			ARRAY_ITEM( KEY( sc->condSpace->condSpaceId ), false );
		}
	}

	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	ARRAY_ITEM( INT(0), true );
	return out;
}

std::ostream &JavaTabCodeGen::KEYS()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Loop the singles. */
		for ( RedTransList::Iter stel = st->outSingle; stel.lte(); stel++ ) {
			ARRAY_ITEM( KEY( stel->lowKey ), false );
		}

		/* Loop the state's transitions. */
		for ( RedTransList::Iter rtel = st->outRange; rtel.lte(); rtel++ ) {
			/* Lower key. */
			ARRAY_ITEM( KEY( rtel->lowKey ), false );

			/* Upper key. */
			ARRAY_ITEM( KEY( rtel->highKey ), false );
		}
	}

	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	ARRAY_ITEM( INT(0), true );
	return out;
}

std::ostream &JavaTabCodeGen::INDICIES()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Walk the singles. */
		for ( RedTransList::Iter stel = st->outSingle; stel.lte(); stel++ ) {
			ARRAY_ITEM( KEY( stel->value->id ), false );
		}

		/* Walk the ranges. */
		for ( RedTransList::Iter rtel = st->outRange; rtel.lte(); rtel++ ) {
			ARRAY_ITEM( KEY( rtel->value->id ), false );
		}

		/* The state's default index goes next. */
		if ( st->defTrans != 0 ) {
			ARRAY_ITEM( KEY( st->defTrans->id ), false );
		}
	}

	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	ARRAY_ITEM( INT(0), true );
	return out;
}

std::ostream &JavaTabCodeGen::TRANS_TARGS()
{
	int totalTrans = 0;
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Walk the singles. */
		for ( RedTransList::Iter stel = st->outSingle; stel.lte(); stel++ ) {
			RedTransAp *trans = stel->value;
			ARRAY_ITEM( KEY( trans->targ->id ), false );
			totalTrans++;
		}

		/* Walk the ranges. */
		for ( RedTransList::Iter rtel = st->outRange; rtel.lte(); rtel++ ) {
			RedTransAp *trans = rtel->value;
			ARRAY_ITEM( KEY( trans->targ->id ), false );
			totalTrans++;
		}

		/* The state's default target state. */
		if ( st->defTrans != 0 ) {
			RedTransAp *trans = st->defTrans;
			ARRAY_ITEM( KEY( trans->targ->id ), false );
			totalTrans++;
		}
	}

	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st->eofTrans != 0 ) {
			RedTransAp *trans = st->eofTrans;
			trans->pos = totalTrans++;
			ARRAY_ITEM( KEY( trans->targ->id ), false );
		}
	}

	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	ARRAY_ITEM( INT(0), true );
	return out;
}


std::ostream &JavaTabCodeGen::TRANS_ACTIONS()
{
	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		/* Walk the singles. */
		for ( RedTransList::Iter stel = st->outSingle; stel.lte(); stel++ ) {
			RedTransAp *trans = stel->value;
			ARRAY_ITEM( INT(TRANS_ACTION( trans )), false );
		}

		/* Walk the ranges. */
		for ( RedTransList::Iter rtel = st->outRange; rtel.lte(); rtel++ ) {
			RedTransAp *trans = rtel->value;
			ARRAY_ITEM( INT(TRANS_ACTION( trans )), false );
		}

		/* The state's default index goes next. */
		if ( st->defTrans != 0 ) {
			RedTransAp *trans = st->defTrans;
			ARRAY_ITEM( INT(TRANS_ACTION( trans )), false );
		}
	}

	for ( RedStateList::Iter st = redFsm->stateList; st.lte(); st++ ) {
		if ( st->eofTrans != 0 ) {
			RedTransAp *trans = st->eofTrans;
			ARRAY_ITEM( INT(TRANS_ACTION( trans )), false );
		}
	}

	/* Output one last number so we don't have to figure out when the last
	 * entry is and avoid writing a comma. */
	ARRAY_ITEM( INT(0), true );
	return out;
}

std::ostream &JavaTabCodeGen::TRANS_TARGS_WI()
{
	/* Transitions must be written ordered by their id. */
	RedTransAp **transPtrs = new RedTransAp*[redFsm->transSet.length()];
	for ( TransApSet::Iter trans = redFsm->transSet; trans.lte(); trans++ )
		transPtrs[trans->id] = trans;

	/* Keep a count of the num of items in the array written. */
	for ( int t = 0; t < redFsm->transSet.length(); t++ ) {
		/* Save the position. Needed for eofTargs. */
		RedTransAp *trans = transPtrs[t];
		trans->pos = t;

		/* Write out the target state. */
		ARRAY_ITEM( INT(trans->targ->id), ( t >= redFsm->transSet.length()-1 ) );
	}
	delete[] transPtrs;
	return out;
}


std::ostream &JavaTabCodeGen::TRANS_ACTIONS_WI()
{
	/* Transitions must be written ordered by their id. */
	RedTransAp **transPtrs = new RedTransAp*[redFsm->transSet.length()];
	for ( TransApSet::Iter trans = redFsm->transSet; trans.lte(); trans++ )
		transPtrs[trans->id] = trans;

	/* Keep a count of the num of items in the array written. */
	for ( int t = 0; t < redFsm->transSet.length(); t++ ) {
		/* Write the function for the transition. */
		RedTransAp *trans = transPtrs[t];
		ARRAY_ITEM( INT(TRANS_ACTION( trans )), ( t >= redFsm->transSet.length()-1 ) );
	}
	delete[] transPtrs;
	return out;
}

void JavaTabCodeGen::writeExports()
{
	if ( exportList.length() > 0 ) {
		for ( ExportList::Iter ex = exportList; ex.lte(); ex++ ) {
			STATIC_VAR( ALPH_TYPE(), DATA_PREFIX() + "ex_" + ex->name ) 
					<< " = " << KEY(ex->key) << ";\n";
		}
		out << "\n";
	}
}

void JavaTabCodeGen::writeStart()
{
	out << START_STATE_ID();
}

void JavaTabCodeGen::writeFirstFinal()
{
	out << FIRST_FINAL_STATE();
}

void JavaTabCodeGen::writeError()
{
	out << ERROR_STATE();
}

void JavaTabCodeGen::writeData()
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
			OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActionLoc), TA() );
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
			OPEN_ARRAY( ARRAY_TYPE(redFsm->maxActionLoc), TA() );
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

	if ( redFsm->startState != 0 )
		STATIC_VAR( "int", START() ) << " = " << START_STATE_ID() << ";\n";

	if ( !noFinal )
		STATIC_VAR( "int" , FIRST_FINAL() ) << " = " << FIRST_FINAL_STATE() << ";\n";

	if ( !noError )
		STATIC_VAR( "int", ERROR() ) << " = " << ERROR_STATE() << ";\n";
	
	out << "\n";

	if ( !noEntry && entryPointNames.length() > 0 ) {
		for ( EntryNameVect::Iter en = entryPointNames; en.lte(); en++ ) {
			STATIC_VAR( "int", DATA_PREFIX() + "en_" + *en ) << 
					" = " << entryPointIds[en.pos()] << ";\n";
		}
		out << "\n";
	}
}

void JavaTabCodeGen::writeExec()
{
	out <<
		"	{\n"
		"	int _klen";

	if ( redFsm->anyRegCurStateRef() )
		out << ", _ps";

	out << 
		";\n"
		"	int _trans = 0;\n";

	if ( redFsm->anyConditions() )
		out << "	int _widec;\n";

	if ( redFsm->anyToStateActions() || redFsm->anyRegActions() || 
			redFsm->anyFromStateActions() )
	{
		out << 
			"	int _acts;\n"
			"	int _nacts;\n";
	}

	out <<
		"	int _keys;\n"
		"	int _goto_targ = 0;\n"
		"\n";
	
	out <<
		"	_goto: while (true) {\n"
		"	switch ( _goto_targ ) {\n"
		"	case 0:\n";

	if ( !noEnd ) {
		out << 
			"	if ( " << P() << " == " << PE() << " ) {\n"
			"		_goto_targ = " << _test_eof << ";\n"
			"		continue _goto;\n"
			"	}\n";
	}

	if ( redFsm->errState != 0 ) {
		out << 
			"	if ( " << vCS() << " == " << redFsm->errState->id << " ) {\n"
			"		_goto_targ = " << _out << ";\n"
			"		continue _goto;\n"
			"	}\n";
	}

	out << "case " << _resume << ":\n"; 

	if ( redFsm->anyFromStateActions() ) {
		out <<
			"	_acts = " << FSA() << "[" << vCS() << "]" << ";\n"
			"	_nacts = " << CAST("int") << " " << A() << "[_acts++];\n"
			"	while ( _nacts-- > 0 ) {\n"
			"		switch ( " << A() << "[_acts++] ) {\n";
			FROM_STATE_ACTION_SWITCH() <<
			"		}\n"
			"	}\n"
			"\n";
	}

	if ( redFsm->anyConditions() )
		COND_TRANSLATE();

	LOCATE_TRANS();

	if ( useIndicies )
		out << "	_trans = " << I() << "[_trans];\n";
	
	if ( redFsm->anyEofTrans() )
		out << "case " << _eof_trans << ":\n";

	if ( redFsm->anyRegCurStateRef() )
		out << "	_ps = " << vCS() << ";\n";

	out <<
		"	" << vCS() << " = " << TT() << "[_trans];\n"
		"\n";

	if ( redFsm->anyRegActions() ) {
		out <<
			"	if ( " << TA() << "[_trans] != 0 ) {\n"
			"		_acts = " <<  TA() << "[_trans]" << ";\n"
			"		_nacts = " << CAST("int") << " " <<  A() << "[_acts++];\n"
			"		while ( _nacts-- > 0 )\n	{\n"
			"			switch ( " << A() << "[_acts++] )\n"
			"			{\n";
			ACTION_SWITCH() <<
			"			}\n"
			"		}\n"
			"	}\n"
			"\n";
	}

	out << "case " << _again << ":\n";

	if ( redFsm->anyToStateActions() ) {
		out <<
			"	_acts = " << TSA() << "[" << vCS() << "]" << ";\n"
			"	_nacts = " << CAST("int") << " " << A() << "[_acts++];\n"
			"	while ( _nacts-- > 0 ) {\n"
			"		switch ( " << A() << "[_acts++] ) {\n";
			TO_STATE_ACTION_SWITCH() <<
			"		}\n"
			"	}\n"
			"\n";
	}

	if ( redFsm->errState != 0 ) {
		out << 
			"	if ( " << vCS() << " == " << redFsm->errState->id << " ) {\n"
			"		_goto_targ = " << _out << ";\n"
			"		continue _goto;\n"
			"	}\n";
	}

	if ( !noEnd ) {
		out << 
			"	if ( ++" << P() << " != " << PE() << " ) {\n"
			"		_goto_targ = " << _resume << ";\n"
			"		continue _goto;\n"
			"	}\n";
	}
	else {
		out << 
			"	" << P() << " += 1;\n"
			"	_goto_targ = " << _resume << ";\n"
			"	continue _goto;\n";
	}

	out << "case " << _test_eof << ":\n"; 

	if ( redFsm->anyEofTrans() || redFsm->anyEofActions() ) {
		out <<
			"	if ( " << P() << " == " << vEOF() << " )\n"
			"	{\n";

		if ( redFsm->anyEofTrans() ) {
			out <<
				"	if ( " << ET() << "[" << vCS() << "] > 0 ) {\n"
				"		_trans = " << ET() << "[" << vCS() << "] - 1;\n"
				"		_goto_targ = " << _eof_trans << ";\n"
				"		continue _goto;\n"
				"	}\n";
		}

		if ( redFsm->anyEofActions() ) {
			out <<
				"	int __acts = " << EA() << "[" << vCS() << "]" << ";\n"
				"	int __nacts = " << CAST("int") << " " << A() << "[__acts++];\n"
				"	while ( __nacts-- > 0 ) {\n"
				"		switch ( " << A() << "[__acts++] ) {\n";
				EOF_ACTION_SWITCH() <<
				"		}\n"
				"	}\n";
		}

		out <<
			"	}\n"
			"\n";
	}

	out << "case " << _out << ":\n"; 

	/* The switch and goto loop. */
	out << "	}\n";
	out << "	break; }\n";

	/* The execute block. */
	out << "	}\n";
}

std::ostream &JavaTabCodeGen::OPEN_ARRAY( string type, string name )
{
	array_type = type;
	array_name = name;
	item_count = 0;
	div_count = 1;

	out <<  "private static " << type << "[] init_" << name << "_0()\n"
		"{\n\t"
		"return new " << type << " [] {\n\t";
	return out;
}

std::ostream &JavaTabCodeGen::ARRAY_ITEM( string item, bool last )
{
	item_count++;

	out << setw(5) << setiosflags(ios::right) << item;
	
	if ( !last ) {
		if ( item_count % SAIIC == 0 ) {
			out << "\n\t};\n};\n"
				"private static "<< array_type << "[] init_" << 
				array_name << "_" << div_count << "()\n"
				"{\n\t"
				"return new " << array_type << " [] {\n\t";
			div_count++;
		} else if (item_count % IALL == 0) { 
			out << ",\n\t";
		} else {
			out << ",";
		}
	}
	return out;
}

std::ostream &JavaTabCodeGen::CLOSE_ARRAY()
{
	out << "\n\t};\n}\n\n";

	if (item_count < SAIIC) {
		out << "private static final " << array_type << " " << array_name << 
			"[] = init_" << array_name << "_0();\n\n";
	} else {
		out << "private static final " << array_type << " [] combine_" << array_name
			<< "() {\n\t"
			<< array_type << " [] combined = new " << array_type << 
			" [ " << item_count << " ];\n\t";
		int block = 0;
		int full_blocks = item_count / SAIIC;
		for (;block < full_blocks; ++block) {
			out << "System.arraycopy ( init_" << array_name << "_" << block << 
				"(), 0, combined, " << SAIIC * block << ", " << SAIIC << " );\n\t";
		}
		if ( (item_count % SAIIC) > 0 ) {
			out << "System.arraycopy ( init_" << array_name << "_" << block << 
				"(), 0, combined, " << SAIIC * block << ", " << 
				(item_count % SAIIC) << " );\n\t";
		}
		out << "return combined;\n}\n";
		out << "private static final " << array_type << " [] " << array_name << 
			" = combine_" << array_name << "();";
	}
	return out;
}


std::ostream &JavaTabCodeGen::STATIC_VAR( string type, string name )
{
	out << "static final " << type << " " << name;
	return out;
}

string JavaTabCodeGen::ARR_OFF( string ptr, string offset )
{
	return ptr + " + " + offset;
}

string JavaTabCodeGen::CAST( string type )
{
	return "(" + type + ")";
}

string JavaTabCodeGen::NULL_ITEM()
{
	/* In java we use integers instead of pointers. */
	return "-1";
}

string JavaTabCodeGen::GET_KEY()
{
	ostringstream ret;
	if ( getKeyExpr != 0 ) { 
		/* Emit the user supplied method of retrieving the key. */
		ret << "(";
		INLINE_LIST( ret, getKeyExpr, 0, false );
		ret << ")";
	}
	else {
		/* Expression for retrieving the key, use simple dereference. */
		ret << DATA() << "[" << P() << "]";
	}
	return ret.str();
}

string JavaTabCodeGen::CTRL_FLOW()
{
	return "if (true) ";
}

unsigned int JavaTabCodeGen::arrayTypeSize( unsigned long maxVal )
{
	long long maxValLL = (long long) maxVal;
	HostType *arrayType = keyOps->typeSubsumes( maxValLL );
	assert( arrayType != 0 );
	return arrayType->size;
}

string JavaTabCodeGen::ARRAY_TYPE( unsigned long maxVal )
{
	long long maxValLL = (long long) maxVal;
	HostType *arrayType = keyOps->typeSubsumes( maxValLL );
	assert( arrayType != 0 );

	string ret = arrayType->data1;
	if ( arrayType->data2 != 0 ) {
		ret += " ";
		ret += arrayType->data2;
	}
	return ret;
}


/* Write out the fsm name. */
string JavaTabCodeGen::FSM_NAME()
{
	return fsmName;
}

/* Emit the offset of the start state as a decimal integer. */
string JavaTabCodeGen::START_STATE_ID()
{
	ostringstream ret;
	ret << redFsm->startState->id;
	return ret.str();
};

/* Write out the array of actions. */
std::ostream &JavaTabCodeGen::ACTIONS_ARRAY()
{
	ARRAY_ITEM( INT(0), false );
	for ( GenActionTableMap::Iter act = redFsm->actionMap; act.lte(); act++ ) {
		/* Write out the length, which will never be the last character. */
		ARRAY_ITEM( INT(act->key.length()), false );

		for ( GenActionTable::Iter item = act->key; item.lte(); item++ )
			ARRAY_ITEM( INT(item->value->actionId), (act.last() && item.last()) );
	}
	return out;
}


string JavaTabCodeGen::ACCESS()
{
	ostringstream ret;
	if ( accessExpr != 0 )
		INLINE_LIST( ret, accessExpr, 0, false );
	return ret.str();
}

string JavaTabCodeGen::P()
{ 
	ostringstream ret;
	if ( pExpr == 0 )
		ret << "p";
	else {
		ret << "(";
		INLINE_LIST( ret, pExpr, 0, false );
		ret << ")";
	}
	return ret.str();
}

string JavaTabCodeGen::PE()
{
	ostringstream ret;
	if ( peExpr == 0 )
		ret << "pe";
	else {
		ret << "(";
		INLINE_LIST( ret, peExpr, 0, false );
		ret << ")";
	}
	return ret.str();
}

string JavaTabCodeGen::vEOF()
{
	ostringstream ret;
	if ( eofExpr == 0 )
		ret << "eof";
	else {
		ret << "(";
		INLINE_LIST( ret, eofExpr, 0, false );
		ret << ")";
	}
	return ret.str();
}

string JavaTabCodeGen::vCS()
{
	ostringstream ret;
	if ( csExpr == 0 )
		ret << ACCESS() << "cs";
	else {
		/* Emit the user supplied method of retrieving the key. */
		ret << "(";
		INLINE_LIST( ret, csExpr, 0, false );
		ret << ")";
	}
	return ret.str();
}

string JavaTabCodeGen::TOP()
{
	ostringstream ret;
	if ( topExpr == 0 )
		ret << ACCESS() + "top";
	else {
		ret << "(";
		INLINE_LIST( ret, topExpr, 0, false );
		ret << ")";
	}
	return ret.str();
}

string JavaTabCodeGen::STACK()
{
	ostringstream ret;
	if ( stackExpr == 0 )
		ret << ACCESS() + "stack";
	else {
		ret << "(";
		INLINE_LIST( ret, stackExpr, 0, false );
		ret << ")";
	}
	return ret.str();
}

string JavaTabCodeGen::ACT()
{
	ostringstream ret;
	if ( actExpr == 0 )
		ret << ACCESS() + "act";
	else {
		ret << "(";
		INLINE_LIST( ret, actExpr, 0, false );
		ret << ")";
	}
	return ret.str();
}

string JavaTabCodeGen::TOKSTART()
{
	ostringstream ret;
	if ( tokstartExpr == 0 )
		ret << ACCESS() + "ts";
	else {
		ret << "(";
		INLINE_LIST( ret, tokstartExpr, 0, false );
		ret << ")";
	}
	return ret.str();
}

string JavaTabCodeGen::TOKEND()
{
	ostringstream ret;
	if ( tokendExpr == 0 )
		ret << ACCESS() + "te";
	else {
		ret << "(";
		INLINE_LIST( ret, tokendExpr, 0, false );
		ret << ")";
	}
	return ret.str();
}

string JavaTabCodeGen::DATA()
{
	ostringstream ret;
	if ( dataExpr == 0 )
		ret << ACCESS() + "data";
	else {
		ret << "(";
		INLINE_LIST( ret, dataExpr, 0, false );
		ret << ")";
	}
	return ret.str();
}


string JavaTabCodeGen::GET_WIDE_KEY()
{
	if ( redFsm->anyConditions() ) 
		return "_widec";
	else
		return GET_KEY();
}

string JavaTabCodeGen::GET_WIDE_KEY( RedStateAp *state )
{
	if ( state->stateCondList.length() > 0 )
		return "_widec";
	else
		return GET_KEY();
}

/* Write out level number of tabs. Makes the nested binary search nice
 * looking. */
string JavaTabCodeGen::TABS( int level )
{
	string result;
	while ( level-- > 0 )
		result += "\t";
	return result;
}

string JavaTabCodeGen::KEY( Key key )
{
	ostringstream ret;
	if ( keyOps->isSigned || !hostLang->explicitUnsigned )
		ret << key.getVal();
	else
		ret << (unsigned long) key.getVal();
	return ret.str();
}

string JavaTabCodeGen::INT( int i )
{
	ostringstream ret;
	ret << i;
	return ret.str();
}

void JavaTabCodeGen::LM_SWITCH( ostream &ret, GenInlineItem *item, 
		int targState, int inFinish )
{
	ret << 
		"	switch( " << ACT() << " ) {\n";

	for ( GenInlineList::Iter lma = *item->children; lma.lte(); lma++ ) {
		/* Write the case label, the action and the case break. */
		if ( lma->lmId < 0 )
			ret << "	default:\n";
		else
			ret << "	case " << lma->lmId << ":\n";

		/* Write the block and close it off. */
		ret << "	{";
		INLINE_LIST( ret, lma->children, targState, inFinish );
		ret << "}\n";

		ret << "	break;\n";
	}

	ret << 
		"	}\n"
		"\t";
}

void JavaTabCodeGen::SET_ACT( ostream &ret, GenInlineItem *item )
{
	ret << ACT() << " = " << item->lmId << ";";
}

void JavaTabCodeGen::SET_TOKEND( ostream &ret, GenInlineItem *item )
{
	/* The tokend action sets tokend. */
	ret << TOKEND() << " = " << P();
	if ( item->offset != 0 ) 
		out << "+" << item->offset;
	out << ";";
}

void JavaTabCodeGen::GET_TOKEND( ostream &ret, GenInlineItem *item )
{
	ret << TOKEND();
}

void JavaTabCodeGen::INIT_TOKSTART( ostream &ret, GenInlineItem *item )
{
	ret << TOKSTART() << " = " << NULL_ITEM() << ";";
}

void JavaTabCodeGen::INIT_ACT( ostream &ret, GenInlineItem *item )
{
	ret << ACT() << " = 0;";
}

void JavaTabCodeGen::SET_TOKSTART( ostream &ret, GenInlineItem *item )
{
	ret << TOKSTART() << " = " << P() << ";";
}

void JavaTabCodeGen::SUB_ACTION( ostream &ret, GenInlineItem *item, 
		int targState, bool inFinish )
{
	if ( item->children->length() > 0 ) {
		/* Write the block and close it off. */
		ret << "{";
		INLINE_LIST( ret, item->children, targState, inFinish );
		ret << "}";
	}
}

void JavaTabCodeGen::ACTION( ostream &ret, GenAction *action, int targState, bool inFinish )
{
	/* Write the preprocessor line info for going into the source file. */
	javaLineDirective( ret, action->loc.fileName, action->loc.line );

	/* Write the block and close it off. */
	ret << "\t{";
	INLINE_LIST( ret, action->inlineList, targState, inFinish );
	ret << "}\n";
}

void JavaTabCodeGen::CONDITION( ostream &ret, GenAction *condition )
{
	ret << "\n";
	javaLineDirective( ret, condition->loc.fileName, condition->loc.line );
	INLINE_LIST( ret, condition->inlineList, 0, false );
}

string JavaTabCodeGen::ERROR_STATE()
{
	ostringstream ret;
	if ( redFsm->errState != 0 )
		ret << redFsm->errState->id;
	else
		ret << "-1";
	return ret.str();
}

string JavaTabCodeGen::FIRST_FINAL_STATE()
{
	ostringstream ret;
	if ( redFsm->firstFinState != 0 )
		ret << redFsm->firstFinState->id;
	else
		ret << redFsm->nextStateId;
	return ret.str();
}

void JavaTabCodeGen::writeInit()
{
	out << "	{\n";

	if ( !noCS )
		out << "\t" << vCS() << " = " << START() << ";\n";
	
	/* If there are any calls, then the stack top needs initialization. */
	if ( redFsm->anyActionCalls() || redFsm->anyActionRets() )
		out << "\t" << TOP() << " = 0;\n";

	if ( hasLongestMatch ) {
		out << 
			"	" << TOKSTART() << " = " << NULL_ITEM() << ";\n"
			"	" << TOKEND() << " = " << NULL_ITEM() << ";\n"
			"	" << ACT() << " = 0;\n";
	}
	out << "	}\n";
}

void JavaTabCodeGen::finishRagelDef()
{
	/* The frontend will do this for us, but it may be a good idea to force it
	 * if the intermediate file is edited. */
	redFsm->sortByStateId();

	/* Choose default transitions and the single transition. */
	redFsm->chooseDefaultSpan();
		
	/* Maybe do flat expand, otherwise choose single. */
	redFsm->chooseSingle();

	/* If any errors have occured in the input file then don't write anything. */
	if ( gblErrorCount > 0 )
		return;
	
	/* Anlayze Machine will find the final action reference counts, among
	 * other things. We will use these in reporting the usage
	 * of fsm directives in action code. */
	analyzeMachine();

	/* Determine if we should use indicies. */
	calcIndexSize();
}

ostream &JavaTabCodeGen::source_warning( const InputLoc &loc )
{
	cerr << sourceFileName << ":" << loc.line << ":" << loc.col << ": warning: ";
	return cerr;
}

ostream &JavaTabCodeGen::source_error( const InputLoc &loc )
{
	gblErrorCount += 1;
	assert( sourceFileName != 0 );
	cerr << sourceFileName << ":" << loc.line << ":" << loc.col << ": ";
	return cerr;
}

#undef _resume
#undef _again
#undef _eof_trans
#undef _test_eof
#undef _out
