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
#include "cscodegen.h"
#include "redfsm.h"
#include "gendata.h"
#include <sstream>
#include <iomanip>
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

void csharpLineDirective( ostream &out, const char *fileName, int line )
{
	if ( noLineDirectives )
		out << "/* ";

	/* Write the preprocessor line info for to the input file. */
	out << "#line " << line  << " \"";
	for ( const char *pc = fileName; *pc != 0; pc++ ) {
		if ( *pc == '\\' )
			out << "\\\\";
		else
			out << *pc;
	}
	out << '"';

	if ( noLineDirectives )
		out << " */";

	out << '\n';
}

void CSharpFsmCodeGen::genLineDirective( ostream &out )
{
	std::streambuf *sbuf = out.rdbuf();
	output_filter *filter = static_cast<output_filter*>(sbuf);
	csharpLineDirective( out, filter->fileName, filter->line + 1 );
}


/* Init code gen with in parameters. */
CSharpFsmCodeGen::CSharpFsmCodeGen( ostream &out )
:
	CodeGenData(out)
{
}

unsigned int CSharpFsmCodeGen::arrayTypeSize( unsigned long maxVal )
{
	long long maxValLL = (long long) maxVal;
	HostType *arrayType = keyOps->typeSubsumes( maxValLL );
	assert( arrayType != 0 );
	return arrayType->size;
}

string CSharpFsmCodeGen::ARRAY_TYPE( unsigned long maxVal )
{
	return ARRAY_TYPE( maxVal, false );
}

string CSharpFsmCodeGen::ARRAY_TYPE( unsigned long maxVal, bool forceSigned )
{
	long long maxValLL = (long long) maxVal;
	HostType *arrayType;
	if (forceSigned)
		arrayType = keyOps->typeSubsumes(true, maxValLL);
	else
		arrayType = keyOps->typeSubsumes( maxValLL );
	assert( arrayType != 0 );

	string ret = arrayType->data1;
	if ( arrayType->data2 != 0 ) {
		ret += " ";
		ret += arrayType->data2;
	}
	return ret;
}

/* Write out the fsm name. */
string CSharpFsmCodeGen::FSM_NAME()
{
	return fsmName;
}

/* Emit the offset of the start state as a decimal integer. */
string CSharpFsmCodeGen::START_STATE_ID()
{
	ostringstream ret;
	ret << redFsm->startState->id;
	return ret.str();
};

/* Write out the array of actions. */
std::ostream &CSharpFsmCodeGen::ACTIONS_ARRAY()
{
	out << "\t0, ";
	int totalActions = 1;
	for ( GenActionTableMap::Iter act = redFsm->actionMap; act.lte(); act++ ) {
		/* Write out the length, which will never be the last character. */
		out << act->key.length() << ", ";
		/* Put in a line break every 8 */
		if ( totalActions++ % 8 == 7 )
			out << "\n\t";

		for ( GenActionTable::Iter item = act->key; item.lte(); item++ ) {
			out << item->value->actionId;
			if ( ! (act.last() && item.last()) )
				out << ", ";

			/* Put in a line break every 8 */
			if ( totalActions++ % 8 == 7 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}


string CSharpFsmCodeGen::ACCESS()
{
	ostringstream ret;
	if ( accessExpr != 0 )
		INLINE_LIST( ret, accessExpr, 0, false );
	return ret.str();
}


string CSharpFsmCodeGen::P()
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

string CSharpFsmCodeGen::PE()
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

string CSharpFsmCodeGen::vEOF()
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

string CSharpFsmCodeGen::vCS()
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

string CSharpFsmCodeGen::TOP()
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

string CSharpFsmCodeGen::STACK()
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

string CSharpFsmCodeGen::ACT()
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

string CSharpFsmCodeGen::TOKSTART()
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

string CSharpFsmCodeGen::TOKEND()
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

string CSharpFsmCodeGen::GET_WIDE_KEY()
{
	if ( redFsm->anyConditions() ) 
		return "_widec";
	else
		return GET_KEY();
}

string CSharpFsmCodeGen::GET_WIDE_KEY( RedStateAp *state )
{
	if ( state->stateCondList.length() > 0 )
		return "_widec";
	else
		return GET_KEY();
}

string CSharpFsmCodeGen::GET_KEY()
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
		ret << "(*" << P() << ")";
	}
	return ret.str();
}

/* Write out level number of tabs. Makes the nested binary search nice
 * looking. */
string CSharpFsmCodeGen::TABS( int level )
{
	string result;
	while ( level-- > 0 )
		result += "\t";
	return result;
}

/* Write out a key from the fsm code gen. Depends on wether or not the key is
 * signed. */
string CSharpFsmCodeGen::KEY( Key key )
{
	ostringstream ret;
	if ( keyOps->isSigned || !hostLang->explicitUnsigned )
		ret << key.getVal();
	else
		ret << (unsigned long) key.getVal() << 'u';
	return ret.str();
}

string CSharpFsmCodeGen::ALPHA_KEY( Key key )
{
	ostringstream ret;
	if (key.getVal() > 0xFFFF) {
		ret << key.getVal();
	} else {
		if ( keyOps->alphType->isChar )
			ret << "'\\u" << std::hex << std::setw(4) << std::setfill('0') << key.getVal() << "'";
		else
			ret << key.getVal();
	}
	//ret << "(char) " << key.getVal();
	return ret.str();
}

void CSharpFsmCodeGen::EXEC( ostream &ret, GenInlineItem *item, int targState, int inFinish )
{
	/* The parser gives fexec two children. The double brackets are for D
	 * code. If the inline list is a single word it will get interpreted as a
	 * C-style cast by the D compiler. */
	ret << "{" << P() << " = ((";
	INLINE_LIST( ret, item->children, targState, inFinish );
	ret << "))-1;}";
}

void CSharpFsmCodeGen::LM_SWITCH( ostream &ret, GenInlineItem *item, 
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

void CSharpFsmCodeGen::SET_ACT( ostream &ret, GenInlineItem *item )
{
	ret << ACT() << " = " << item->lmId << ";";
}

void CSharpFsmCodeGen::SET_TOKEND( ostream &ret, GenInlineItem *item )
{
	/* The tokend action sets tokend. */
	ret << TOKEND() << " = " << P();
	if ( item->offset != 0 ) 
		out << "+" << item->offset;
	out << ";";
}

void CSharpFsmCodeGen::GET_TOKEND( ostream &ret, GenInlineItem *item )
{
	ret << TOKEND();
}

void CSharpFsmCodeGen::INIT_TOKSTART( ostream &ret, GenInlineItem *item )
{
	ret << TOKSTART() << " = " << NULL_ITEM() << ";";
}

void CSharpFsmCodeGen::INIT_ACT( ostream &ret, GenInlineItem *item )
{
	ret << ACT() << " = 0;";
}

void CSharpFsmCodeGen::SET_TOKSTART( ostream &ret, GenInlineItem *item )
{
	ret << TOKSTART() << " = " << P() << ";";
}

void CSharpFsmCodeGen::SUB_ACTION( ostream &ret, GenInlineItem *item, 
		int targState, bool inFinish )
{
	if ( item->children->length() > 0 ) {
		/* Write the block and close it off. */
		ret << "{";
		INLINE_LIST( ret, item->children, targState, inFinish );
		ret << "}";
	}
}


/* Write out an inline tree structure. Walks the list and possibly calls out
 * to virtual functions than handle language specific items in the tree. */
void CSharpFsmCodeGen::INLINE_LIST( ostream &ret, GenInlineList *inlineList, 
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
/* Write out paths in line directives. Escapes any special characters. */
string CSharpFsmCodeGen::LDIR_PATH( char *path )
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

void CSharpFsmCodeGen::ACTION( ostream &ret, GenAction *action, int targState, bool inFinish )
{
	/* Write the preprocessor line info for going into the source file. */
	csharpLineDirective( ret, action->loc.fileName, action->loc.line );

	/* Write the block and close it off. */
	ret << "\t{";
	INLINE_LIST( ret, action->inlineList, targState, inFinish );
	ret << "}\n";
}

void CSharpFsmCodeGen::CONDITION( ostream &ret, GenAction *condition )
{
	ret << "\n";
	csharpLineDirective( ret, condition->loc.fileName, condition->loc.line );
	INLINE_LIST( ret, condition->inlineList, 0, false );
}

string CSharpFsmCodeGen::ERROR_STATE()
{
	ostringstream ret;
	if ( redFsm->errState != 0 )
		ret << redFsm->errState->id;
	else
		ret << "-1";
	return ret.str();
}

string CSharpFsmCodeGen::FIRST_FINAL_STATE()
{
	ostringstream ret;
	if ( redFsm->firstFinState != 0 )
		ret << redFsm->firstFinState->id;
	else
		ret << redFsm->nextStateId;
	return ret.str();
}

void CSharpFsmCodeGen::writeInit()
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

string CSharpFsmCodeGen::DATA_PREFIX()
{
	if ( !noPrefix )
		return FSM_NAME() + "_";
	return "";
}

/* Emit the alphabet data type. */
string CSharpFsmCodeGen::ALPH_TYPE()
{
	string ret = keyOps->alphType->data1;
	if ( keyOps->alphType->data2 != 0 ) {
		ret += " ";
		ret += + keyOps->alphType->data2;
	}
	return ret;
}

/* Emit the alphabet data type. */
string CSharpFsmCodeGen::WIDE_ALPH_TYPE()
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

void CSharpFsmCodeGen::STATE_IDS()
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


void CSharpFsmCodeGen::writeStart()
{
	out << START_STATE_ID();
}

void CSharpFsmCodeGen::writeFirstFinal()
{
	out << FIRST_FINAL_STATE();
}

void CSharpFsmCodeGen::writeError()
{
	out << ERROR_STATE();
}

/*
 * C# Specific
 */
string CSharpCodeGen::GET_KEY()
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
		if ( dataExpr == 0 )
			ret << "data";
		else
			INLINE_LIST( ret, dataExpr, 0, false );

		ret << "[" << P() << "]";
	}
	return ret.str();
}
string CSharpCodeGen::NULL_ITEM()
{
	return "-1";
}

string CSharpCodeGen::POINTER()
{
	// XXX C# has no pointers
	// multiple items seperated by commas can also be pointer types.
	return " ";
}

string CSharpCodeGen::PTR_CONST()
{
	return "";
}

std::ostream &CSharpCodeGen::OPEN_ARRAY( string type, string name )
{
	out << "static readonly " << type << "[] " << name << " =  ";
	/*
	if (type == "char")
		out << "Encoding.ASCII.Get";
	else */
		out << "new " << type << " [] {\n";
	return out;
}

std::ostream &CSharpCodeGen::CLOSE_ARRAY()
{
	return out << "};\n";
}

std::ostream &CSharpCodeGen::STATIC_VAR( string type, string name )
{
	out << "const " << type << " " << name;
	return out;
}

string CSharpCodeGen::ARR_OFF( string ptr, string offset )
{
	// XXX C# can't do pointer arithmetic
	return "&" + ptr + "[" + offset + "]";
}

string CSharpCodeGen::CAST( string type )
{
	return "(" + type + ")";
}

string CSharpCodeGen::UINT( )
{
	return "uint";
}

std::ostream &CSharpCodeGen::SWITCH_DEFAULT()
{
	out << "		default: break;\n";
	return out;
}

string CSharpCodeGen::CTRL_FLOW()
{
	return "if (true) ";
}

void CSharpCodeGen::writeExports()
{
	if ( exportList.length() > 0 ) {
		for ( ExportList::Iter ex = exportList; ex.lte(); ex++ ) {
			out << "const " << ALPH_TYPE() << " " << DATA_PREFIX() << 
					"ex_" << ex->name << " = " << KEY(ex->key) << ";\n";
		}
		out << "\n";
	}
}

/*
 * End C#-specific code.
 */

void CSharpFsmCodeGen::finishRagelDef()
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

ostream &CSharpFsmCodeGen::source_warning( const InputLoc &loc )
{
	cerr << sourceFileName << ":" << loc.line << ":" << loc.col << ": warning: ";
	return cerr;
}

ostream &CSharpFsmCodeGen::source_error( const InputLoc &loc )
{
	gblErrorCount += 1;
	assert( sourceFileName != 0 );
	cerr << sourceFileName << ":" << loc.line << ":" << loc.col << ": ";
	return cerr;
}

