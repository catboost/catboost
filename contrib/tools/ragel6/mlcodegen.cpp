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
#include "mlcodegen.h"
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

void ocamlLineDirective( ostream &out, const char *fileName, int line )
{
	if ( noLineDirectives )
		return;

	/* Write the line info for to the input file. */
	out << "# " << line << " \"";
	for ( const char *pc = fileName; *pc != 0; pc++ ) {
		if ( *pc == '\\' || *pc == '"' )
			out << "\\";
  	out << *pc;
	}
	out << "\"\n";
}

void OCamlCodeGen::genLineDirective( ostream &out )
{
	std::streambuf *sbuf = out.rdbuf();
	output_filter *filter = static_cast<output_filter*>(sbuf);
	ocamlLineDirective( out, filter->fileName, filter->line + 1 );
}


/* Init code gen with in parameters. */
OCamlCodeGen::OCamlCodeGen( ostream &out )
:
	CodeGenData(out)
{
}

unsigned int OCamlCodeGen::arrayTypeSize( unsigned long maxVal )
{
	long long maxValLL = (long long) maxVal;
	HostType *arrayType = keyOps->typeSubsumes( maxValLL );
	assert( arrayType != 0 );
	return arrayType->size;
}

string OCamlCodeGen::ARRAY_TYPE( unsigned long maxVal )
{
	return ARRAY_TYPE( maxVal, false );
}

string OCamlCodeGen::ARRAY_TYPE( unsigned long maxVal, bool forceSigned )
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
string OCamlCodeGen::FSM_NAME()
{
	return fsmName;
}

/* Emit the offset of the start state as a decimal integer. */
string OCamlCodeGen::START_STATE_ID()
{
	ostringstream ret;
	ret << redFsm->startState->id;
	return ret.str();
};

/* Write out the array of actions. */
std::ostream &OCamlCodeGen::ACTIONS_ARRAY()
{
	out << "\t0; ";
	int totalActions = 1;
	for ( GenActionTableMap::Iter act = redFsm->actionMap; act.lte(); act++ ) {
		/* Write out the length, which will never be the last character. */
		out << act->key.length() << ARR_SEP();
		/* Put in a line break every 8 */
		if ( totalActions++ % 8 == 7 )
			out << "\n\t";

		for ( GenActionTable::Iter item = act->key; item.lte(); item++ ) {
			out << item->value->actionId;
			if ( ! (act.last() && item.last()) )
				out << ARR_SEP();

			/* Put in a line break every 8 */
			if ( totalActions++ % 8 == 7 )
				out << "\n\t";
		}
	}
	out << "\n";
	return out;
}


/*
string OCamlCodeGen::ACCESS()
{
	ostringstream ret;
	if ( accessExpr != 0 )
		INLINE_LIST( ret, accessExpr, 0, false );
	return ret.str();
}
*/

string OCamlCodeGen::make_access(char const* name, GenInlineList* x, bool prefix = true)
{ 
	ostringstream ret;
	if ( x == 0 )
  {
    if (prefix && accessExpr != 0)
    {
      INLINE_LIST( ret, accessExpr, 0, false);
      ret << name;
    }
    else
      ret << name << ".contents"; // ref cell
  }
	else {
		ret << "(";
		INLINE_LIST( ret, x, 0, false );
		ret << ")";
	}
	return ret.str();
}

string OCamlCodeGen::P() { return make_access("p", pExpr, false); }
string OCamlCodeGen::PE() { return make_access("pe", peExpr, false); }
string OCamlCodeGen::vEOF() { return make_access("eof", eofExpr, false); }
string OCamlCodeGen::vCS() { return make_access("cs", csExpr); }
string OCamlCodeGen::TOP() { return make_access("top", topExpr); }
string OCamlCodeGen::STACK() { return make_access("stack", stackExpr); }
string OCamlCodeGen::ACT() { return make_access("act", actExpr); }
string OCamlCodeGen::TOKSTART() { return make_access("ts", tokstartExpr); }
string OCamlCodeGen::TOKEND() { return make_access("te", tokendExpr); }

string OCamlCodeGen::GET_WIDE_KEY()
{
	if ( redFsm->anyConditions() ) 
		return "_widec";
	else
    { ostringstream ret; ret << "Char.code " << GET_KEY(); return ret.str(); }
}

string OCamlCodeGen::GET_WIDE_KEY( RedStateAp *state )
{
	if ( state->stateCondList.length() > 0 )
		return "_widec";
	else
    { ostringstream ret; ret << "Char.code " << GET_KEY(); return ret.str(); }
}

/* Write out level number of tabs. Makes the nested binary search nice
 * looking. */
string OCamlCodeGen::TABS( int level )
{
	string result;
	while ( level-- > 0 )
		result += "\t";
	return result;
}

/* Write out a key from the fsm code gen. Depends on wether or not the key is
 * signed. */
string OCamlCodeGen::KEY( Key key )
{
	ostringstream ret;
	if ( keyOps->isSigned || !hostLang->explicitUnsigned )
		ret << key.getVal();
	else
		ret << (unsigned long) key.getVal() << 'u';
	return ret.str();
}

string OCamlCodeGen::ALPHA_KEY( Key key )
{
	ostringstream ret;
  ret << key.getVal();
  /*
	if (key.getVal() > 0xFFFF) {
		ret << key.getVal();
	} else {
		ret << "'\\u" << std::hex << std::setw(4) << std::setfill('0') << 
			key.getVal() << "'";
	}
  */
	//ret << "(char) " << key.getVal();
	return ret.str();
}

void OCamlCodeGen::EXEC( ostream &ret, GenInlineItem *item, int targState, int inFinish )
{
// The parser gives fexec two children.
	ret << "begin " << P() << " <- ";
	INLINE_LIST( ret, item->children, targState, inFinish );
	ret << " - 1 end; ";
}

void OCamlCodeGen::LM_SWITCH( ostream &ret, GenInlineItem *item, 
		int targState, int inFinish )
{
	bool catch_all = false;
	ret << 
		"	begin match " << ACT() << " with\n";

	for ( GenInlineList::Iter lma = *item->children; lma.lte(); lma++ ) {
		/* Write the case label, the action and the case break. */
		if ( lma->lmId < 0 )
		{
			catch_all = true;
			ret << "	| _ ->\n";
		}
		else
			ret << "	| " << lma->lmId << " ->\n";

		/* Write the block and close it off. */
		ret << "	begin ";
		INLINE_LIST( ret, lma->children, targState, inFinish );
		ret << " end\n";
	}

	if (!catch_all)
		ret << "  | _ -> assert false\n";

	ret << 
		"	end;\n"
		"\t";
}

void OCamlCodeGen::SET_ACT( ostream &ret, GenInlineItem *item )
{
	ret << ACT() << " <- " << item->lmId << "; ";
}

void OCamlCodeGen::SET_TOKEND( ostream &ret, GenInlineItem *item )
{
	/* The tokend action sets tokend. */
	ret << TOKEND() << " <- " << P();
	if ( item->offset != 0 ) 
		out << "+" << item->offset;
	out << "; ";
}

void OCamlCodeGen::GET_TOKEND( ostream &ret, GenInlineItem *item )
{
	ret << TOKEND();
}

void OCamlCodeGen::INIT_TOKSTART( ostream &ret, GenInlineItem *item )
{
	ret << TOKSTART() << " <- " << NULL_ITEM() << "; ";
}

void OCamlCodeGen::INIT_ACT( ostream &ret, GenInlineItem *item )
{
	ret << ACT() << " <- 0;";
}

void OCamlCodeGen::SET_TOKSTART( ostream &ret, GenInlineItem *item )
{
	ret << TOKSTART() << " <- " << P() << "; ";
}

void OCamlCodeGen::SUB_ACTION( ostream &ret, GenInlineItem *item, 
		int targState, bool inFinish )
{
	if ( item->children->length() > 0 ) {
		/* Write the block and close it off. */
		ret << "begin ";
		INLINE_LIST( ret, item->children, targState, inFinish );
		ret << " end";
	}
}


/* Write out an inline tree structure. Walks the list and possibly calls out
 * to virtual functions than handle language specific items in the tree. */
void OCamlCodeGen::INLINE_LIST( ostream &ret, GenInlineList *inlineList, 
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
			ret << P() << " <- " << P() << " - 1; ";
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
string OCamlCodeGen::LDIR_PATH( char *path )
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

void OCamlCodeGen::ACTION( ostream &ret, GenAction *action, int targState, bool inFinish )
{
	/* Write the preprocessor line info for going into the source file. */
	ocamlLineDirective( ret, action->loc.fileName, action->loc.line );

	/* Write the block and close it off. */
	ret << "\t\tbegin ";
	INLINE_LIST( ret, action->inlineList, targState, inFinish );
	ret << " end;\n";
}

void OCamlCodeGen::CONDITION( ostream &ret, GenAction *condition )
{
	ret << "\n";
	ocamlLineDirective( ret, condition->loc.fileName, condition->loc.line );
	INLINE_LIST( ret, condition->inlineList, 0, false );
}

string OCamlCodeGen::ERROR_STATE()
{
	ostringstream ret;
	if ( redFsm->errState != 0 )
		ret << redFsm->errState->id;
	else
		ret << "-1";
	return ret.str();
}

string OCamlCodeGen::FIRST_FINAL_STATE()
{
	ostringstream ret;
	if ( redFsm->firstFinState != 0 )
		ret << redFsm->firstFinState->id;
	else
		ret << redFsm->nextStateId;
	return ret.str();
}

void OCamlCodeGen::writeInit()
{
	out << "	begin\n";

	if ( !noCS )
		out << "\t" << vCS() << " <- " << START() << ";\n";
	
	/* If there are any calls, then the stack top needs initialization. */
	if ( redFsm->anyActionCalls() || redFsm->anyActionRets() )
		out << "\t" << TOP() << " <- 0;\n";

	if ( hasLongestMatch ) {
		out << 
			"	" << TOKSTART() << " <- " << NULL_ITEM() << ";\n"
			"	" << TOKEND() << " <- " << NULL_ITEM() << ";\n"
			"	" << ACT() << " <- 0;\n";
	}
	out << "	end;\n";
}

string OCamlCodeGen::PRE_INCR(string val)
{
  ostringstream ret;
  ret << "(" << val << " <- " << val << " + 1; " << val << ")";
  return ret.str();
}

string OCamlCodeGen::POST_INCR(string val)
{
  ostringstream ret;
  ret << "(let temp = " << val << " in " << val << " <- " << val << " + 1; temp)";
  return ret.str();
}

string OCamlCodeGen::PRE_DECR(string val)
{
  ostringstream ret;
  ret << "(" << val << " <- " << val << " - 1; " << val << ")";
  return ret.str();
}

string OCamlCodeGen::POST_DECR(string val)
{
  ostringstream ret;
  ret << "(let temp = " << val << " in " << val << " <- " << val << " - 1; temp)";
  return ret.str();
}

string OCamlCodeGen::DATA_PREFIX()
{
  if ( data_prefix.empty() ) // init
  {
    data_prefix = string(fsmName) + "_";
    if (data_prefix.size() > 0)
      data_prefix[0] = ::tolower(data_prefix[0]); // uncapitalize
  }
	if ( !noPrefix )
		return data_prefix;
	return "";
}

/* Emit the alphabet data type. */
string OCamlCodeGen::ALPH_TYPE()
{
	string ret = keyOps->alphType->data1;
	if ( keyOps->alphType->data2 != 0 ) {
		ret += " ";
		ret += + keyOps->alphType->data2;
	}
	return ret;
}

/* Emit the alphabet data type. */
string OCamlCodeGen::WIDE_ALPH_TYPE()
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

void OCamlCodeGen::STATE_IDS()
{
	if ( redFsm->startState != 0 )
		STATIC_VAR( "int", START() ) << " = " << START_STATE_ID() << TOP_SEP ();

	if ( !noFinal )
		STATIC_VAR( "int" , FIRST_FINAL() ) << " = " << FIRST_FINAL_STATE() << TOP_SEP();

	if ( !noError )
		STATIC_VAR( "int", ERROR() ) << " = " << ERROR_STATE() << TOP_SEP();

	out << "\n";

	if ( !noEntry && entryPointNames.length() > 0 ) {
		for ( EntryNameVect::Iter en = entryPointNames; en.lte(); en++ ) {
			STATIC_VAR( "int", DATA_PREFIX() + "en_" + *en ) << 
					" = " << entryPointIds[en.pos()] << TOP_SEP();
		}
		out << "\n";
	}
}


void OCamlCodeGen::writeStart()
{
	out << START_STATE_ID();
}

void OCamlCodeGen::writeFirstFinal()
{
	out << FIRST_FINAL_STATE();
}

void OCamlCodeGen::writeError()
{
	out << ERROR_STATE();
}

string OCamlCodeGen::GET_KEY()
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
		ret << "data.[" << P() << "]";
	}
	return ret.str();
}
string OCamlCodeGen::NULL_ITEM()
{
	return "-1";
}

string OCamlCodeGen::POINTER()
{
	// XXX C# has no pointers
	// multiple items seperated by commas can also be pointer types.
	return " ";
}

string OCamlCodeGen::PTR_CONST()
{
	return "";
}

std::ostream &OCamlCodeGen::OPEN_ARRAY( string type, string name )
{
	out << "let " << name << " : " << type << " array = [|" << endl;
	return out;
}

std::ostream &OCamlCodeGen::CLOSE_ARRAY()
{
	return out << "|]" << TOP_SEP();
}

string OCamlCodeGen::TOP_SEP()
{
  return "\n"; // original syntax
}

string OCamlCodeGen::ARR_SEP()
{
  return "; ";
}

string OCamlCodeGen::AT(const string& array, const string& index)
{
  ostringstream ret;
  ret << array << ".(" << index << ")";
  return ret.str();
}

std::ostream &OCamlCodeGen::STATIC_VAR( string type, string name )
{
	out << "let " << name << " : " << type;
	return out;
}

string OCamlCodeGen::ARR_OFF( string ptr, string offset )
{
	// XXX C# can't do pointer arithmetic
	return "&" + ptr + "[" + offset + "]";
}

string OCamlCodeGen::CAST( string type )
{
  return "";
//	return "(" + type + ")";
}

string OCamlCodeGen::UINT( )
{
	return "uint";
}

std::ostream &OCamlCodeGen::SWITCH_DEFAULT()
{
	out << "		| _ -> ()\n";
	return out;
}

string OCamlCodeGen::CTRL_FLOW()
{
	return "if true then ";
}

void OCamlCodeGen::finishRagelDef()
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

ostream &OCamlCodeGen::source_warning( const InputLoc &loc )
{
	cerr << sourceFileName << ":" << loc.line << ":" << loc.col << ": warning: ";
	return cerr;
}

ostream &OCamlCodeGen::source_error( const InputLoc &loc )
{
	gblErrorCount += 1;
	assert( sourceFileName != 0 );
	cerr << sourceFileName << ":" << loc.line << ":" << loc.col << ": ";
	return cerr;
}

