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
#include "rubycodegen.h"
#include "pcheck.h"
#include "vector.h"
#include "version.h"
#include "common.h"

#include "ragel.h"
#include "rubytable.h"
#include "rubyftable.h"
#include "rubyflat.h"
#include "rubyfflat.h"
#include "rbxgoto.h"

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

/* Target ruby impl */

/* Target language and output style. */
extern CodeStyle codeStyle;

extern int numSplitPartitions;
extern bool noLineDirectives;

/*
 * Callbacks invoked by the XML data parser.
 */


void rubyLineDirective( ostream &out, const char *fileName, int line )
{
	if ( noLineDirectives )
		return;

	/* Write a comment containing line info. */
	out << "# line " << line  << " \"";
	for ( const char *pc = fileName; *pc != 0; pc++ ) {
		if ( *pc == '\\' )
			out << "\\\\";
		else
			out << *pc;
	}
	out << "\"\n";
}

void RubyCodeGen::genLineDirective( ostream &out )
{
	std::streambuf *sbuf = out.rdbuf();
	output_filter *filter = static_cast<output_filter*>(sbuf);
	rubyLineDirective( out, filter->fileName, filter->line + 1 );
}

string RubyCodeGen::DATA_PREFIX()
{
	if ( !noPrefix )
		return FSM_NAME() + "_";
	return "";
}

std::ostream &RubyCodeGen::STATIC_VAR( string type, string name )
{
	out << 
		"class << self\n"
		"	attr_accessor :" << name << "\n"
		"end\n"
		"self." << name;
	return out;
}


std::ostream &RubyCodeGen::OPEN_ARRAY( string type, string name )
{
	out << 
		"class << self\n"
		"	attr_accessor :" << name << "\n"
		"	private :" << name << ", :" << name << "=\n"
		"end\n"
		"self." << name << " = [\n";
	return out;
}

std::ostream &RubyCodeGen::CLOSE_ARRAY()
{
	out << "]\n";
	return out;
}


string RubyCodeGen::ARR_OFF( string ptr, string offset )
{
	return ptr + "[" + offset + "]";
}

string RubyCodeGen::NULL_ITEM()
{
	return "nil";
}


string RubyCodeGen::P()
{ 
	ostringstream ret;
	if ( pExpr == 0 )
		ret << "p";
	else {
		//ret << "(";
		INLINE_LIST( ret, pExpr, 0, false );
		//ret << ")";
	}
	return ret.str();
}

string RubyCodeGen::PE()
{
	ostringstream ret;
	if ( peExpr == 0 )
		ret << "pe";
	else {
		//ret << "(";
		INLINE_LIST( ret, peExpr, 0, false );
		//ret << ")";
	}
	return ret.str();
}

string RubyCodeGen::vEOF()
{
	ostringstream ret;
	if ( eofExpr == 0 )
		ret << "eof";
	else {
		//ret << "(";
		INLINE_LIST( ret, eofExpr, 0, false );
		//ret << ")";
	}
	return ret.str();
}

string RubyCodeGen::vCS()
{
	ostringstream ret;
	if ( csExpr == 0 )
		ret << ACCESS() << "cs";
	else {
		//ret << "(";
		INLINE_LIST( ret, csExpr, 0, false );
		//ret << ")";
	}
	return ret.str();
}

string RubyCodeGen::TOP()
{
	ostringstream ret;
	if ( topExpr == 0 )
		ret << ACCESS() + "top";
	else {
		//ret << "(";
		INLINE_LIST( ret, topExpr, 0, false );
		//ret << ")";
	}
	return ret.str();
}

string RubyCodeGen::STACK()
{
	ostringstream ret;
	if ( stackExpr == 0 )
		ret << ACCESS() + "stack";
	else {
		//ret << "(";
		INLINE_LIST( ret, stackExpr, 0, false );
		//ret << ")";
	}
	return ret.str();
}

string RubyCodeGen::ACT()
{
	ostringstream ret;
	if ( actExpr == 0 )
		ret << ACCESS() + "act";
	else {
		//ret << "(";
		INLINE_LIST( ret, actExpr, 0, false );
		//ret << ")";
	}
	return ret.str();
}

string RubyCodeGen::TOKSTART()
{
	ostringstream ret;
	if ( tokstartExpr == 0 )
		ret << ACCESS() + "ts";
	else {
		//ret << "(";
		INLINE_LIST( ret, tokstartExpr, 0, false );
		//ret << ")";
	}
	return ret.str();
}

string RubyCodeGen::TOKEND()
{
	ostringstream ret;
	if ( tokendExpr == 0 )
		ret << ACCESS() + "te";
	else {
		//ret << "(";
		INLINE_LIST( ret, tokendExpr, 0, false );
		//ret << ")";
	}
	return ret.str();
}

string RubyCodeGen::DATA()
{
	ostringstream ret;
	if ( dataExpr == 0 )
		ret << ACCESS() + "data";
	else {
		//ret << "(";
		INLINE_LIST( ret, dataExpr, 0, false );
		//ret << ")";
	}
	return ret.str();
}

/* Write out the fsm name. */
string RubyCodeGen::FSM_NAME()
{
	return fsmName;
}


void RubyCodeGen::ACTION( ostream &ret, GenAction *action, int targState, bool inFinish )
{
	/* Write the preprocessor line info for going into the source file. */
	rubyLineDirective( ret, action->loc.fileName, action->loc.line );

	/* Write the block and close it off. */
	ret << "		begin\n";
	INLINE_LIST( ret, action->inlineList, targState, inFinish );
	ret << "		end\n";
}



string RubyCodeGen::GET_WIDE_KEY()
{
	if ( redFsm->anyConditions() ) 
		return "_widec";
	else
		return GET_KEY();
}

string RubyCodeGen::GET_WIDE_KEY( RedStateAp *state )
{
	if ( state->stateCondList.length() > 0 )
		return "_widec";
	else
		return GET_KEY();
}

string RubyCodeGen::GET_KEY()
{
	ostringstream ret;
	if ( getKeyExpr != 0 ) { 
		/* Emit the user supplied method of retrieving the key. */
		ret << "(";
		INLINE_LIST( ret, getKeyExpr, 0, false );
		ret << ")";
	}
	else {
		/* Expression for retrieving the key, use dereference and read ordinal,
		 * for compatibility with Ruby 1.9. */
		ret << DATA() << "[" << P() << "].ord";
	}
	return ret.str();
}

string RubyCodeGen::KEY( Key key )
{
	ostringstream ret;
	if ( keyOps->isSigned || !hostLang->explicitUnsigned )
		ret << key.getVal();
	else
		ret << (unsigned long) key.getVal();
	return ret.str();
}


/* Write out level number of tabs. Makes the nested binary search nice
 * looking. */
string RubyCodeGen::TABS( int level )
{
	string result;
	while ( level-- > 0 )
		result += "\t";
	return result;
}

string RubyCodeGen::INT( int i )
{
	ostringstream ret;
	ret << i;
	return ret.str();
}

void RubyCodeGen::CONDITION( ostream &ret, GenAction *condition )
{
	ret << "\n";
	rubyLineDirective( ret, condition->loc.fileName, condition->loc.line );
	INLINE_LIST( ret, condition->inlineList, 0, false );
}

/* Emit the alphabet data type. */
string RubyCodeGen::ALPH_TYPE()
{
	string ret = keyOps->alphType->data1;
	if ( keyOps->alphType->data2 != 0 ) {
		ret += " ";
		ret += + keyOps->alphType->data2;
	}
	return ret;
}

/* Emit the alphabet data type. */
string RubyCodeGen::WIDE_ALPH_TYPE()
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


string RubyCodeGen::ARRAY_TYPE( unsigned long maxVal )
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

/* Write out the array of actions. */
std::ostream &RubyCodeGen::ACTIONS_ARRAY()
{
	START_ARRAY_LINE();
	int totalActions = 0;
	ARRAY_ITEM( INT(0), ++totalActions, false );
	for ( GenActionTableMap::Iter act = redFsm->actionMap; act.lte(); act++ ) {
		/* Write out the length, which will never be the last character. */
		ARRAY_ITEM( INT(act->key.length()), ++totalActions, false );

		for ( GenActionTable::Iter item = act->key; item.lte(); item++ ) {
			ARRAY_ITEM( INT(item->value->actionId), ++totalActions, (act.last() && item.last()) );
		}
	}
	END_ARRAY_LINE();
	return out;
}

void RubyCodeGen::STATE_IDS()
{
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

std::ostream &RubyCodeGen::START_ARRAY_LINE()
{
	out << "\t";
	return out;
}

std::ostream &RubyCodeGen::ARRAY_ITEM( string item, int count, bool last )
{
	out << item;
	if ( !last )
	{
		out << ", ";
		if ( count % IALL == 0 )
		{
			END_ARRAY_LINE();
			START_ARRAY_LINE();
		}
	}
	return out;
}

std::ostream &RubyCodeGen::END_ARRAY_LINE()
{
	out << "\n";
	return out;
}

/* Emit the offset of the start state as a decimal integer. */
string RubyCodeGen::START_STATE_ID()
{
	ostringstream ret;
	ret << redFsm->startState->id;
	return ret.str();
};

string RubyCodeGen::ERROR_STATE()
{
	ostringstream ret;
	if ( redFsm->errState != 0 )
		ret << redFsm->errState->id;
	else
		ret << "-1";
	return ret.str();
}

string RubyCodeGen::FIRST_FINAL_STATE()
{
	ostringstream ret;
	if ( redFsm->firstFinState != 0 )
		ret << redFsm->firstFinState->id;
	else
		ret << redFsm->nextStateId;
	return ret.str();
}

string RubyCodeGen::ACCESS()
{
	ostringstream ret;
	if ( accessExpr != 0 )
		INLINE_LIST( ret, accessExpr, 0, false );
	return ret.str();
}

/* Write out an inline tree structure. Walks the list and possibly calls out
 * to virtual functions than handle language specific items in the tree. */
void RubyCodeGen::INLINE_LIST( ostream &ret, GenInlineList *inlineList, 
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
			ret << P() << " = " << P() << " - 1;";
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


void RubyCodeGen::EXEC( ostream &ret, GenInlineItem *item, int targState, int inFinish )
{
	/* The parser gives fexec two children. The double brackets are for D
	 * code. If the inline list is a single word it will get interpreted as a
	 * C-style cast by the D compiler. */
	ret << " begin " << P() << " = ((";
	INLINE_LIST( ret, item->children, targState, inFinish );
	ret << "))-1; end\n";
}

void RubyCodeGen::LM_SWITCH( ostream &ret, GenInlineItem *item, 
		int targState, int inFinish )
{
	ret << 
		"	case " << ACT() << "\n";

	for ( GenInlineList::Iter lma = *item->children; lma.lte(); lma++ ) {
		/* Write the case label, the action and the case break. */
		if ( lma->lmId < 0 )
			ret << "	else\n";
		else
			ret << "	when " << lma->lmId << " then\n";


		/* Write the block and close it off. */
		ret << "	begin";
		INLINE_LIST( ret, lma->children, targState, inFinish );
		ret << "end\n";
	}

	ret << "end \n\t";
}

void RubyCodeGen::SET_ACT( ostream &ret, GenInlineItem *item )
{
	ret << ACT() << " = " << item->lmId << ";";
}

void RubyCodeGen::INIT_TOKSTART( ostream &ret, GenInlineItem *item )
{
	ret << TOKSTART() << " = " << NULL_ITEM() << ";";
}

void RubyCodeGen::INIT_ACT( ostream &ret, GenInlineItem *item )
{
	ret << ACT() << " = 0\n";
}

void RubyCodeGen::SET_TOKSTART( ostream &ret, GenInlineItem *item )
{
	ret << TOKSTART() << " = " << P() << "\n";
}

void RubyCodeGen::SET_TOKEND( ostream &ret, GenInlineItem *item )
{
	/* The tokend action sets tokend. */
	ret << TOKEND() << " = " << P();
	if ( item->offset != 0 ) 
		out << "+" << item->offset;
	out << "\n";
}

void RubyCodeGen::GET_TOKEND( ostream &ret, GenInlineItem *item )
{
	ret << TOKEND();
}

void RubyCodeGen::SUB_ACTION( ostream &ret, GenInlineItem *item, 
		int targState, bool inFinish )
{
	if ( item->children->length() > 0 ) {
		/* Write the block and close it off. */
		ret << " begin ";
		INLINE_LIST( ret, item->children, targState, inFinish );
		ret << " end\n";
	}
}

int RubyCodeGen::TRANS_ACTION( RedTransAp *trans )
{
	/* If there are actions, emit them. Otherwise emit zero. */
	int act = 0;
	if ( trans->action != 0 )
		act = trans->action->location+1;
	return act;
}

ostream &RubyCodeGen::source_warning( const InputLoc &loc )
{
	cerr << sourceFileName << ":" << loc.line << ":" << loc.col << ": warning: ";
	return cerr;
}

ostream &RubyCodeGen::source_error( const InputLoc &loc )
{
	gblErrorCount += 1;
	assert( sourceFileName != 0 );
	cerr << sourceFileName << ":" << loc.line << ":" << loc.col << ": ";
	return cerr;
}

void RubyCodeGen::finishRagelDef()
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


/* Determine if we should use indicies or not. */
void RubyCodeGen::calcIndexSize()
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

unsigned int RubyCodeGen::arrayTypeSize( unsigned long maxVal )
{
	long long maxValLL = (long long) maxVal;
	HostType *arrayType = keyOps->typeSubsumes( maxValLL );
	assert( arrayType != 0 );
	return arrayType->size;
}


void RubyCodeGen::writeInit()
{
	out << "begin\n";
	
	out << "	" << P() << " ||= 0\n";

	if ( !noEnd ) 
		out << "	" << PE() << " ||= " << DATA() << ".length\n";

	if ( !noCS )
		out << "	" << vCS() << " = " << START() << "\n";

	/* If there are any calls, then the stack top needs initialization. */
	if ( redFsm->anyActionCalls() || redFsm->anyActionRets() )
		out << "	" << TOP() << " = 0\n";

	if ( hasLongestMatch ) {
		out <<
			"	" << TOKSTART() << " = " << NULL_ITEM() << "\n"
			"	" << TOKEND() << " = " << NULL_ITEM() << "\n"
			"	" << ACT() << " = 0\n";
	}

	out << "end\n";
}

void RubyCodeGen::writeExports()
{
	if ( exportList.length() > 0 ) {
		for ( ExportList::Iter ex = exportList; ex.lte(); ex++ ) {
			STATIC_VAR( ALPH_TYPE(), DATA_PREFIX() + "ex_" + ex->name ) 
					<< " = " << KEY(ex->key) << "\n";
		}
		out << "\n";
	}
}

void RubyCodeGen::writeStart()
{
	out << START_STATE_ID();
}

void RubyCodeGen::writeFirstFinal()
{
	out << FIRST_FINAL_STATE();
}

void RubyCodeGen::writeError()
{
	out << ERROR_STATE();
}


/*
 * Local Variables:
 * mode: c++
 * indent-tabs-mode: 1
 * c-file-style: "bsd"
 * End:
 */
