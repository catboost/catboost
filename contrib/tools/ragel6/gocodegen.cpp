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

#include "gocodegen.h"
#include "ragel.h"
#include "redfsm.h"
#include "gendata.h"
#include <sstream>
#include <string>
#include <assert.h>


using std::ostream;
using std::ostringstream;
using std::string;
using std::cerr;
using std::endl;
using std::istream;
using std::ifstream;
using std::ostream;
using std::ios;
using std::cin;
using std::cout;
using std::cerr;
using std::endl;

/*
 * Go Specific
 */

void goLineDirective( ostream &out, const char *fileName, int line )
{
	out << "//line " << fileName << ":" << line << endl;
}

void GoCodeGen::genLineDirective( ostream &out )
{
	std::streambuf *sbuf = out.rdbuf();
	output_filter *filter = static_cast<output_filter*>(sbuf);
	goLineDirective( out, filter->fileName, filter->line + 1 );
}

unsigned int GoCodeGen::arrayTypeSize( unsigned long maxVal )
{
	long long maxValLL = (long long) maxVal;
	HostType *arrayType = keyOps->typeSubsumes( maxValLL );
	assert( arrayType != 0 );
	return arrayType->size;
}

string GoCodeGen::ARRAY_TYPE( unsigned long maxVal )
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
string GoCodeGen::FSM_NAME()
{
	return fsmName;
}

/* Emit the offset of the start state as a decimal integer. */
string GoCodeGen::START_STATE_ID()
{
	ostringstream ret;
	ret << redFsm->startState->id;
	return ret.str();
};

/* Write out the array of actions. */
std::ostream &GoCodeGen::ACTIONS_ARRAY()
{
	out << "	0, ";
	int totalActions = 1;
	for ( GenActionTableMap::Iter act = redFsm->actionMap; act.lte(); act++ ) {
		/* Write out the length, which will never be the last character. */
		out << act->key.length() << ", ";
		if ( totalActions++ % IALL == 0 )
			out << endl << "	";

		for ( GenActionTable::Iter item = act->key; item.lte(); item++ ) {
			out << item->value->actionId << ", ";
			if ( ! (act.last() && item.last()) ) {
				if ( totalActions++ % IALL == 0 )
					out << endl << "	";
			}
		}
	}
	out << endl;
	return out;
}


string GoCodeGen::ACCESS()
{
	ostringstream ret;
	if ( accessExpr != 0 )
		INLINE_LIST( ret, accessExpr, 0, false, false );
	return ret.str();
}


string GoCodeGen::P()
{
	ostringstream ret;
	if ( pExpr == 0 )
		ret << "p";
	else {
		ret << "(";
		INLINE_LIST( ret, pExpr, 0, false, false );
		ret << ")";
	}
	return ret.str();
}

string GoCodeGen::PE()
{
	ostringstream ret;
	if ( peExpr == 0 )
		ret << "pe";
	else {
		ret << "(";
		INLINE_LIST( ret, peExpr, 0, false, false );
		ret << ")";
	}
	return ret.str();
}

string GoCodeGen::vEOF()
{
	ostringstream ret;
	if ( eofExpr == 0 )
		ret << "eof";
	else {
		ret << "(";
		INLINE_LIST( ret, eofExpr, 0, false, false );
		ret << ")";
	}
	return ret.str();
}

string GoCodeGen::vCS()
{
	ostringstream ret;
	if ( csExpr == 0 )
		ret << ACCESS() << "cs";
	else {
		/* Emit the user supplied method of retrieving the key. */
		ret << "(";
		INLINE_LIST( ret, csExpr, 0, false, false );
		ret << ")";
	}
	return ret.str();
}

string GoCodeGen::TOP()
{
	ostringstream ret;
	if ( topExpr == 0 )
		ret << ACCESS() + "top";
	else {
		ret << "(";
		INLINE_LIST( ret, topExpr, 0, false, false );
		ret << ")";
	}
	return ret.str();
}

string GoCodeGen::STACK()
{
	ostringstream ret;
	if ( stackExpr == 0 )
		ret << ACCESS() + "stack";
	else {
		ret << "(";
		INLINE_LIST( ret, stackExpr, 0, false, false );
		ret << ")";
	}
	return ret.str();
}

string GoCodeGen::ACT()
{
	ostringstream ret;
	if ( actExpr == 0 )
		ret << ACCESS() + "act";
	else {
		ret << "(";
		INLINE_LIST( ret, actExpr, 0, false, false );
		ret << ")";
	}
	return ret.str();
}

string GoCodeGen::TOKSTART()
{
	ostringstream ret;
	if ( tokstartExpr == 0 )
		ret << ACCESS() + "ts";
	else {
		ret << "(";
		INLINE_LIST( ret, tokstartExpr, 0, false, false );
		ret << ")";
	}
	return ret.str();
}

string GoCodeGen::TOKEND()
{
	ostringstream ret;
	if ( tokendExpr == 0 )
		ret << ACCESS() + "te";
	else {
		ret << "(";
		INLINE_LIST( ret, tokendExpr, 0, false, false );
		ret << ")";
	}
	return ret.str();
}

string GoCodeGen::GET_WIDE_KEY()
{
	if ( redFsm->anyConditions() )
		return "_widec";
	else
		return GET_KEY();
}

string GoCodeGen::GET_WIDE_KEY( RedStateAp *state )
{
	if ( state->stateCondList.length() > 0 )
		return "_widec";
	else
		return GET_KEY();
}

string GoCodeGen::GET_KEY()
{
	ostringstream ret;
	if ( getKeyExpr != 0 ) {
		/* Emit the user supplied method of retrieving the key. */
		ret << "(";
		INLINE_LIST( ret, getKeyExpr, 0, false, false );
		ret << ")";
	}
	else {
		/* Expression for retrieving the key, use simple dereference. */
		ret << DATA() << "[" << P() << "]";
	}
	return ret.str();
}

/* Write out level number of tabs. Makes the nested binary search nice
 * looking. */
string GoCodeGen::TABS( int level )
{
	string result;
	while ( level-- > 0 )
		result += "\t";
	return result;
}

/* Write out a key from the fsm code gen. Depends on wether or not the key is
 * signed. */
string GoCodeGen::KEY( Key key )
{
	ostringstream ret;
	if ( keyOps->isSigned || !hostLang->explicitUnsigned )
		ret << key.getVal();
	else
		ret << (unsigned long) key.getVal() << 'u';
	return ret.str();
}

bool GoCodeGen::isAlphTypeSigned()
{
	return keyOps->isSigned;
}

bool GoCodeGen::isWideAlphTypeSigned()
{
	string ret;
	if ( redFsm->maxKey <= keyOps->maxKey )
		return isAlphTypeSigned();
	else {
		long long maxKeyVal = redFsm->maxKey.getLongLong();
		HostType *wideType = keyOps->typeSubsumes( keyOps->isSigned, maxKeyVal );
		return wideType->isSigned;
	}
}

string GoCodeGen::WIDE_KEY( RedStateAp *state, Key key )
{
	if ( state->stateCondList.length() > 0 ) {
		ostringstream ret;
		if ( isWideAlphTypeSigned() )
			ret << key.getVal();
		else
			ret << (unsigned long) key.getVal() << 'u';
		return ret.str();
	}
	else {
		return KEY( key );
	}
}



void GoCodeGen::EXEC( ostream &ret, GenInlineItem *item, int targState, int inFinish )
{
	/* The parser gives fexec two children. The double brackets are for D
	 * code. If the inline list is a single word it will get interpreted as a
	 * C-style cast by the D compiler. */
	ret << P() << " = (";
	INLINE_LIST( ret, item->children, targState, inFinish, false );
	ret << ") - 1" << endl;
}

void GoCodeGen::LM_SWITCH( ostream &ret, GenInlineItem *item,
		int targState, int inFinish, bool csForced )
{
	ret <<
		"	switch " << ACT() << " {" << endl;

	for ( GenInlineList::Iter lma = *item->children; lma.lte(); lma++ ) {
		/* Write the case label, the action and the case break. */
		if ( lma->lmId < 0 ) {
			ret << "	default:" << endl;
		}
		else
			ret << "	case " << lma->lmId << ":" << endl;

		/* Write the block and close it off. */
		ret << "	{";
		INLINE_LIST( ret, lma->children, targState, inFinish, csForced );
		ret << "}" << endl;
	}

	ret <<
		"	}" << endl <<
		"	";
}

void GoCodeGen::SET_ACT( ostream &ret, GenInlineItem *item )
{
	ret << ACT() << " = " << item->lmId << ";";
}

void GoCodeGen::SET_TOKEND( ostream &ret, GenInlineItem *item )
{
	/* The tokend action sets tokend. */
	ret << TOKEND() << " = " << P();
	if ( item->offset != 0 )
		out << "+" << item->offset;
	out << endl;
}

void GoCodeGen::GET_TOKEND( ostream &ret, GenInlineItem *item )
{
	ret << TOKEND();
}

void GoCodeGen::INIT_TOKSTART( ostream &ret, GenInlineItem *item )
{
	ret << TOKSTART() << " = " << NULL_ITEM() << endl;
}

void GoCodeGen::INIT_ACT( ostream &ret, GenInlineItem *item )
{
	ret << ACT() << " = 0" << endl;
}

void GoCodeGen::SET_TOKSTART( ostream &ret, GenInlineItem *item )
{
	ret << TOKSTART() << " = " << P() << endl;
}

void GoCodeGen::SUB_ACTION( ostream &ret, GenInlineItem *item,
		int targState, bool inFinish, bool csForced )
{
	if ( item->children->length() > 0 ) {
		/* Write the block and close it off. */
		ret << "{";
		INLINE_LIST( ret, item->children, targState, inFinish, csForced );
		ret << "}";
	}
}


/* Write out an inline tree structure. Walks the list and possibly calls out
 * to virtual functions than handle language specific items in the tree. */
void GoCodeGen::INLINE_LIST( ostream &ret, GenInlineList *inlineList,
		int targState, bool inFinish, bool csForced )
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
			ret << P() << "--" << endl;
			break;
		case GenInlineItem::Exec:
			EXEC( ret, item, targState, inFinish );
			break;
		case GenInlineItem::Curs:
			CURS( ret, inFinish );
			break;
		case GenInlineItem::Targs:
			TARGS( ret, inFinish, targState );
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
			LM_SWITCH( ret, item, targState, inFinish, csForced );
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
			SUB_ACTION( ret, item, targState, inFinish, csForced );
			break;
		case GenInlineItem::Break:
			BREAK( ret, targState, csForced );
			break;
		}
	}
}
/* Write out paths in line directives. Escapes any special characters. */
string GoCodeGen::LDIR_PATH( char *path )
{
	ostringstream ret;
	for ( char *pc = path; *pc != 0; pc++ ) {
		if ( *pc == '\\' )
			ret << "\\\\";
		else
			ret << *pc;
	}
	return ret.str();
}

void GoCodeGen::ACTION( ostream &ret, GenAction *action, int targState,
		bool inFinish, bool csForced )
{
	/* Write the preprocessor line info for going into the source file. */
	goLineDirective( ret, action->loc.fileName, action->loc.line );

	/* Write the block and close it off. */
	INLINE_LIST( ret, action->inlineList, targState, inFinish, csForced );
	ret << endl;
}

void GoCodeGen::CONDITION( ostream &ret, GenAction *condition )
{
	INLINE_LIST( ret, condition->inlineList, 0, false, false );
}

string GoCodeGen::ERROR_STATE()
{
	ostringstream ret;
	if ( redFsm->errState != 0 )
		ret << redFsm->errState->id;
	else
		ret << "-1";
	return ret.str();
}

string GoCodeGen::FIRST_FINAL_STATE()
{
	ostringstream ret;
	if ( redFsm->firstFinState != 0 )
		ret << redFsm->firstFinState->id;
	else
		ret << redFsm->nextStateId;
	return ret.str();
}

void GoCodeGen::writeInit()
{
	out << "	{" << endl;

	if ( !noCS )
		out << "	" << vCS() << " = " << START() << endl;

	/* If there are any calls, then the stack top needs initialization. */
	if ( redFsm->anyActionCalls() || redFsm->anyActionRets() )
		out << "	" << TOP() << " = 0" << endl;

	if ( hasLongestMatch ) {
		out <<
			"	" << TOKSTART() << " = " << NULL_ITEM() << endl <<
			"	" << TOKEND() << " = " << NULL_ITEM() << endl <<
			"	" << ACT() << " = 0" << endl;
	}
	out << "	}" << endl;
}

string GoCodeGen::DATA()
{
	ostringstream ret;
	if ( dataExpr == 0 )
		ret << ACCESS() + "data";
	else {
		ret << "(";
		INLINE_LIST( ret, dataExpr, 0, false, false );
		ret << ")";
	}
	return ret.str();
}

string GoCodeGen::DATA_PREFIX()
{
	if ( !noPrefix )
		return FSM_NAME() + "_";
	return "";
}

/* Emit the alphabet data type. */
string GoCodeGen::ALPH_TYPE()
{
	string ret = keyOps->alphType->data1;
	if ( keyOps->alphType->data2 != 0 ) {
		ret += " ";
		ret += + keyOps->alphType->data2;
	}
	return ret;
}

/* Emit the alphabet data type. */
string GoCodeGen::WIDE_ALPH_TYPE()
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

void GoCodeGen::STATE_IDS()
{
	if ( redFsm->startState != 0 )
		CONST( "int", START() ) << " = " << START_STATE_ID() << endl;

	if ( !noFinal )
		CONST( "int" , FIRST_FINAL() ) << " = " << FIRST_FINAL_STATE() << endl;

	if ( !noError )
		CONST( "int", ERROR() ) << " = " << ERROR_STATE() << endl;

	out << endl;

	if ( !noEntry && entryPointNames.length() > 0 ) {
		for ( EntryNameVect::Iter en = entryPointNames; en.lte(); en++ ) {
			CONST( "int", DATA_PREFIX() + "en_" + *en ) <<
					" = " << entryPointIds[en.pos()] << endl;
		}
		out << endl;
	}
}

void GoCodeGen::writeStart()
{
	out << START_STATE_ID();
}

void GoCodeGen::writeFirstFinal()
{
	out << FIRST_FINAL_STATE();
}

void GoCodeGen::writeError()
{
	out << ERROR_STATE();
}

void GoCodeGen::finishRagelDef()
{
	if ( codeStyle == GenGoto || codeStyle == GenFGoto ||
			codeStyle == GenIpGoto || codeStyle == GenSplit )
	{
		/* For directly executable machines there is no required state
		 * ordering. Choose a depth-first ordering to increase the
		 * potential for fall-throughs. */
		redFsm->depthFirstOrdering();
	}
	else {
		/* The frontend will do this for us, but it may be a good idea to
		 * force it if the intermediate file is edited. */
		redFsm->sortByStateId();
	}

	/* Choose default transitions and the single transition. */
	redFsm->chooseDefaultSpan();

	/* Maybe do flat expand, otherwise choose single. */
	if ( codeStyle == GenFlat || codeStyle == GenFFlat )
		redFsm->makeFlat();
	else
		redFsm->chooseSingle();

	/* If any errors have occured in the input file then don't write anything. */
	if ( gblErrorCount > 0 )
		return;

	if ( codeStyle == GenSplit )
		redFsm->partitionFsm( numSplitPartitions );

	if ( codeStyle == GenIpGoto || codeStyle == GenSplit )
		redFsm->setInTrans();

	/* Anlayze Machine will find the final action reference counts, among
	 * other things. We will use these in reporting the usage
	 * of fsm directives in action code. */
	analyzeMachine();

	/* Determine if we should use indicies. */
	calcIndexSize();
}

ostream &GoCodeGen::source_warning( const InputLoc &loc )
{
	cerr << sourceFileName << ":" << loc.line << ":" << loc.col << ": warning: ";
	return cerr;
}

ostream &GoCodeGen::source_error( const InputLoc &loc )
{
	gblErrorCount += 1;
	assert( sourceFileName != 0 );
	cerr << sourceFileName << ":" << loc.line << ":" << loc.col << ": ";
	return cerr;
}


/*
 * Go implementation.
 *
 */

std::ostream &GoCodeGen::OPEN_ARRAY( string type, string name )
{
	out << "var " << name << " []" << type << " = []" << type << "{" << endl;
	return out;
}

std::ostream &GoCodeGen::CLOSE_ARRAY()
{
	return out << "}" << endl;
}

std::ostream &GoCodeGen::STATIC_VAR( string type, string name )
{
	out << "var " << name << " " << type;
	return out;
}

std::ostream &GoCodeGen::CONST( string type, string name )
{
	out << "const " << name << " " << type;
	return out;
}

string GoCodeGen::UINT( )
{
	return "uint";
}

string GoCodeGen::INT()
{
	return "int";
}

string GoCodeGen::CAST( string type, string expr )
{
	return type + "(" + expr + ")";
}

string GoCodeGen::NULL_ITEM()
{
	return "0";
}

void GoCodeGen::writeExports()
{
	if ( exportList.length() > 0 ) {
		for ( ExportList::Iter ex = exportList; ex.lte(); ex++ ) {
			out << "const " << DATA_PREFIX() << "ex_" << ex->name << " = " <<
					KEY(ex->key) << endl;
		}
		out << endl;
	}
}
