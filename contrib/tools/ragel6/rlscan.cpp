
#line 1 "rlscan.rl"
/*
 *  Copyright 2006-2007 Adrian Thurston <thurston@complang.org>
 *  Copyright 2011 Josef Goettgens
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

#include <iostream>
#include <fstream>
#include <string.h>

#include "ragel.h"
#include "rlscan.h"
#include "inputdata.h"

//#define LOG_TOKENS

using std::ifstream;
using std::istream;
using std::ostream;
using std::cout;
using std::cerr;
using std::endl;

enum InlineBlockType
{
	CurlyDelimited,
	SemiTerminated
};


/*
 * The Scanner for Importing
 */


#line 125 "rlscan.rl"



#line 65 "rlscan.cpp"
static const int inline_token_scan_start = 2;
static const int inline_token_scan_first_final = 2;
static const int inline_token_scan_error = -1;

static const int inline_token_scan_en_main = 2;


#line 128 "rlscan.rl"

void Scanner::flushImport()
{
	int *p = token_data;
	int *pe = token_data + cur_token;
	int *eof = 0;

	
#line 82 "rlscan.cpp"
	{
	 tok_cs = inline_token_scan_start;
	 tok_ts = 0;
	 tok_te = 0;
	 tok_act = 0;
	}

#line 90 "rlscan.cpp"
	{
	if ( p == pe )
		goto _test_eof;
	switch (  tok_cs )
	{
tr0:
#line 123 "rlscan.rl"
	{{p = (( tok_te))-1;}}
	goto st2;
tr1:
#line 109 "rlscan.rl"
	{ tok_te = p+1;{ 
			int base = tok_ts - token_data;
			int nameOff = 0;
			int litOff = 2;

			directToParser( inclToParser, fileName, line, column, TK_Word, 
					token_strings[base+nameOff], token_lens[base+nameOff] );
			directToParser( inclToParser, fileName, line, column, '=', 0, 0 );
			directToParser( inclToParser, fileName, line, column, TK_Literal,
					token_strings[base+litOff], token_lens[base+litOff] );
			directToParser( inclToParser, fileName, line, column, ';', 0, 0 );
		}}
	goto st2;
tr2:
#line 81 "rlscan.rl"
	{ tok_te = p+1;{ 
			int base = tok_ts - token_data;
			int nameOff = 0;
			int numOff = 2;

			directToParser( inclToParser, fileName, line, column, TK_Word, 
					token_strings[base+nameOff], token_lens[base+nameOff] );
			directToParser( inclToParser, fileName, line, column, '=', 0, 0 );
			directToParser( inclToParser, fileName, line, column, TK_UInt,
					token_strings[base+numOff], token_lens[base+numOff] );
			directToParser( inclToParser, fileName, line, column, ';', 0, 0 );
		}}
	goto st2;
tr3:
#line 95 "rlscan.rl"
	{ tok_te = p+1;{ 
			int base = tok_ts - token_data;
			int nameOff = 1;
			int litOff = 2;

			directToParser( inclToParser, fileName, line, column, TK_Word, 
					token_strings[base+nameOff], token_lens[base+nameOff] );
			directToParser( inclToParser, fileName, line, column, '=', 0, 0 );
			directToParser( inclToParser, fileName, line, column, TK_Literal,
					token_strings[base+litOff], token_lens[base+litOff] );
			directToParser( inclToParser, fileName, line, column, ';', 0, 0 );
		}}
	goto st2;
tr4:
#line 67 "rlscan.rl"
	{ tok_te = p+1;{ 
			int base = tok_ts - token_data;
			int nameOff = 1;
			int numOff = 2;

			directToParser( inclToParser, fileName, line, column, TK_Word, 
					token_strings[base+nameOff], token_lens[base+nameOff] );
			directToParser( inclToParser, fileName, line, column, '=', 0, 0 );
			directToParser( inclToParser, fileName, line, column, TK_UInt,
					token_strings[base+numOff], token_lens[base+numOff] );
			directToParser( inclToParser, fileName, line, column, ';', 0, 0 );
		}}
	goto st2;
tr5:
#line 123 "rlscan.rl"
	{ tok_te = p+1;}
	goto st2;
tr8:
#line 123 "rlscan.rl"
	{ tok_te = p;p--;}
	goto st2;
st2:
#line 1 "NONE"
	{ tok_ts = 0;}
	if ( ++p == pe )
		goto _test_eof2;
case 2:
#line 1 "NONE"
	{ tok_ts = p;}
#line 176 "rlscan.cpp"
	switch( (*p) ) {
		case 128: goto tr6;
		case 131: goto tr7;
	}
	goto tr5;
tr6:
#line 1 "NONE"
	{ tok_te = p+1;}
	goto st3;
st3:
	if ( ++p == pe )
		goto _test_eof3;
case 3:
#line 190 "rlscan.cpp"
	if ( (*p) == 61 )
		goto st0;
	goto tr8;
st0:
	if ( ++p == pe )
		goto _test_eof0;
case 0:
	switch( (*p) ) {
		case 129: goto tr1;
		case 130: goto tr2;
	}
	goto tr0;
tr7:
#line 1 "NONE"
	{ tok_te = p+1;}
	goto st4;
st4:
	if ( ++p == pe )
		goto _test_eof4;
case 4:
#line 211 "rlscan.cpp"
	if ( (*p) == 128 )
		goto st1;
	goto tr8;
st1:
	if ( ++p == pe )
		goto _test_eof1;
case 1:
	switch( (*p) ) {
		case 129: goto tr3;
		case 130: goto tr4;
	}
	goto tr0;
	}
	_test_eof2:  tok_cs = 2; goto _test_eof; 
	_test_eof3:  tok_cs = 3; goto _test_eof; 
	_test_eof0:  tok_cs = 0; goto _test_eof; 
	_test_eof4:  tok_cs = 4; goto _test_eof; 
	_test_eof1:  tok_cs = 1; goto _test_eof; 

	_test_eof: {}
	if ( p == eof )
	{
	switch (  tok_cs ) {
	case 3: goto tr8;
	case 0: goto tr0;
	case 4: goto tr8;
	case 1: goto tr0;
	}
	}

	}

#line 139 "rlscan.rl"


	if ( tok_ts == 0 )
		cur_token = 0;
	else {
		cur_token = pe - tok_ts;
		int ts_offset = tok_ts - token_data;
		memmove( token_data, token_data+ts_offset, cur_token*sizeof(token_data[0]) );
		memmove( token_strings, token_strings+ts_offset, cur_token*sizeof(token_strings[0]) );
		memmove( token_lens, token_lens+ts_offset, cur_token*sizeof(token_lens[0]) );
	}
}

void Scanner::directToParser( Parser *toParser, const char *tokFileName, int tokLine, 
		int tokColumn, int type, char *tokdata, int toklen )
{
	InputLoc loc;

	#ifdef LOG_TOKENS
	cerr << "scanner:" << tokLine << ":" << tokColumn << 
			": sending token to the parser " << Parser_lelNames[type];
	cerr << " " << toklen;
	if ( tokdata != 0 )
		cerr << " " << tokdata;
	cerr << endl;
	#endif

	loc.fileName = tokFileName;
	loc.line = tokLine;
	loc.col = tokColumn;

	toParser->token( loc, type, tokdata, toklen );
}

void Scanner::importToken( int token, char *start, char *end )
{
	if ( cur_token == max_tokens )
		flushImport();

	token_data[cur_token] = token;
	if ( start == 0 ) {
		token_strings[cur_token] = 0;
		token_lens[cur_token] = 0;
	}
	else {
		int toklen = end-start;
		token_lens[cur_token] = toklen;
		token_strings[cur_token] = new char[toklen+1];
		memcpy( token_strings[cur_token], start, toklen );
		token_strings[cur_token][toklen] = 0;
	}
	cur_token++;
}

void Scanner::pass( int token, char *start, char *end )
{
	if ( importMachines )
		importToken( token, start, end );
	pass();
}

void Scanner::pass()
{
	updateCol();

	/* If no errors and we are at the bottom of the include stack (the
	 * source file listed on the command line) then write out the data. */
	if ( includeDepth == 0 && machineSpec == 0 && machineName == 0 )
		id.inputItems.tail->data.write( ts, te-ts );
}

/*
 * The scanner for processing sections, includes, imports, etc.
 */


#line 321 "rlscan.cpp"
static const int section_parse_start = 10;
static const int section_parse_first_final = 10;
static const int section_parse_error = 0;

static const int section_parse_en_main = 10;


#line 218 "rlscan.rl"



void Scanner::init( )
{
	
#line 336 "rlscan.cpp"
	{
	cs = section_parse_start;
	}

#line 224 "rlscan.rl"
}

bool Scanner::active()
{
	if ( ignoreSection )
		return false;

	if ( parser == 0 && ! parserExistsError ) {
		scan_error() << "this specification has no name, nor does any previous"
			" specification" << endl;
		parserExistsError = true;
	}

	if ( parser == 0 )
		return false;

	return true;
}

ostream &Scanner::scan_error()
{
	/* Maintain the error count. */
	gblErrorCount += 1;
	cerr << makeInputLoc( fileName, line, column ) << ": ";
	return cerr;
}

/* An approximate check for duplicate includes. Due to aliasing of files it's
 * possible for duplicates to creep in. */
bool Scanner::duplicateInclude( char *inclFileName, char *inclSectionName )
{
	for ( IncludeHistory::Iter hi = parser->includeHistory; hi.lte(); hi++ ) {
		if ( strcmp( hi->fileName, inclFileName ) == 0 &&
				strcmp( hi->sectionName, inclSectionName ) == 0 )
		{
			return true;
		}
	}
	return false;	
}

void Scanner::updateCol()
{
	char *from = lastnl;
	if ( from == 0 )
		from = ts;
	//cerr << "adding " << te - from << " to column" << endl;
	column += te - from;
	lastnl = 0;
}

void Scanner::handleMachine()
{
	/* Assign a name to the machine. */
	char *machine = word;

	if ( !importMachines && inclSectionTarg == 0 ) {
		ignoreSection = false;

		ParserDictEl *pdEl = id.parserDict.find( machine );
		if ( pdEl == 0 ) {
			pdEl = new ParserDictEl( machine );
			pdEl->value = new Parser( fileName, machine, sectionLoc );
			pdEl->value->init();
			id.parserDict.insert( pdEl );
			id.parserList.append( pdEl->value );
		}

		parser = pdEl->value;
	}
	else if ( !importMachines && strcmp( inclSectionTarg, machine ) == 0 ) {
		/* found include target */
		ignoreSection = false;
		parser = inclToParser;
	}
	else {
		/* ignoring section */
		ignoreSection = true;
		parser = 0;
	}
}

void Scanner::handleInclude()
{
	if ( active() ) {
		char *inclSectionName = word;
		char **includeChecks = 0;

		/* Implement defaults for the input file and section name. */
		if ( inclSectionName == 0 )
			inclSectionName = parser->sectionName;

		if ( lit != 0 )
			includeChecks = makeIncludePathChecks( fileName, lit, lit_len );
		else {
			char *test = new char[strlen(fileName)+1];
			strcpy( test, fileName );

			includeChecks = new char*[2];

			includeChecks[0] = test;
			includeChecks[1] = 0;
		}

		long found = 0;
		ifstream *inFile = tryOpenInclude( includeChecks, found );
		if ( inFile == 0 ) {
			scan_error() << "include: failed to locate file" << endl;
			char **tried = includeChecks;
			while ( *tried != 0 )
				scan_error() << "include: attempted: \"" << *tried++ << '\"' << endl;
		}
		else {
			/* Don't include anything that's already been included. */
			if ( !duplicateInclude( includeChecks[found], inclSectionName ) ) {
				parser->includeHistory.append( IncludeHistoryItem( 
						includeChecks[found], inclSectionName ) );

				Scanner scanner( id, includeChecks[found], *inFile, parser,
						inclSectionName, includeDepth+1, false );
				scanner.do_scan( );
				delete inFile;
			}
		}
	}
}

void Scanner::handleImport()
{
	if ( active() ) {
		char **importChecks = makeIncludePathChecks( fileName, lit, lit_len );

		/* Open the input file for reading. */
		long found = 0;
		ifstream *inFile = tryOpenInclude( importChecks, found );
		if ( inFile == 0 ) {
			scan_error() << "import: could not open import file " <<
					"for reading" << endl;
			char **tried = importChecks;
			while ( *tried != 0 )
				scan_error() << "import: attempted: \"" << *tried++ << '\"' << endl;
		}

		Scanner scanner( id, importChecks[found], *inFile, parser,
				0, includeDepth+1, true );
		scanner.do_scan( );
		scanner.importToken( 0, 0, 0 );
		scanner.flushImport();
		delete inFile;
	}
}


#line 461 "rlscan.rl"


void Scanner::token( int type, char c )
{
	token( type, &c, &c + 1 );
}

void Scanner::token( int type )
{
	token( type, 0, 0 );
}

void Scanner::token( int type, char *start, char *end )
{
	char *tokdata = 0;
	int toklen = 0;
	if ( start != 0 ) {
		toklen = end-start;
		tokdata = new char[toklen+1];
		memcpy( tokdata, start, toklen );
		tokdata[toklen] = 0;
	}

	processToken( type, tokdata, toklen );
}

void Scanner::processToken( int type, char *tokdata, int toklen )
{
	int *p, *pe, *eof;

	if ( type < 0 )
		p = pe = eof = 0;
	else {
		p = &type;
		pe = &type + 1;
		eof = 0;
	}

	
#line 535 "rlscan.cpp"
	{
	if ( p == pe )
		goto _test_eof;
	switch ( cs )
	{
tr2:
#line 391 "rlscan.rl"
	{ handleMachine(); }
	goto st10;
tr6:
#line 392 "rlscan.rl"
	{ handleInclude(); }
	goto st10;
tr10:
#line 393 "rlscan.rl"
	{ handleImport(); }
	goto st10;
tr13:
#line 433 "rlscan.rl"
	{
		if ( active() && machineSpec == 0 && machineName == 0 )
			id.inputItems.tail->writeArgs.append( 0 );
	}
	goto st10;
tr14:
#line 444 "rlscan.rl"
	{
		/* Send the token off to the parser. */
		if ( active() )
			directToParser( parser, fileName, line, column, type, tokdata, toklen );
	}
	goto st10;
st10:
	if ( ++p == pe )
		goto _test_eof10;
case 10:
#line 572 "rlscan.cpp"
	switch( (*p) ) {
		case 191: goto st1;
		case 192: goto st3;
		case 193: goto st6;
		case 194: goto tr18;
	}
	goto tr14;
st1:
	if ( ++p == pe )
		goto _test_eof1;
case 1:
	if ( (*p) == 128 )
		goto tr1;
	goto tr0;
tr0:
#line 386 "rlscan.rl"
	{ scan_error() << "bad machine statement" << endl; }
	goto st0;
tr3:
#line 387 "rlscan.rl"
	{ scan_error() << "bad include statement" << endl; }
	goto st0;
tr8:
#line 388 "rlscan.rl"
	{ scan_error() << "bad import statement" << endl; }
	goto st0;
tr11:
#line 389 "rlscan.rl"
	{ scan_error() << "bad write statement" << endl; }
	goto st0;
#line 603 "rlscan.cpp"
st0:
cs = 0;
	goto _out;
tr1:
#line 383 "rlscan.rl"
	{ word = tokdata; word_len = toklen; }
	goto st2;
st2:
	if ( ++p == pe )
		goto _test_eof2;
case 2:
#line 615 "rlscan.cpp"
	if ( (*p) == 59 )
		goto tr2;
	goto tr0;
st3:
	if ( ++p == pe )
		goto _test_eof3;
case 3:
	switch( (*p) ) {
		case 128: goto tr4;
		case 129: goto tr5;
	}
	goto tr3;
tr4:
#line 382 "rlscan.rl"
	{ word = lit = 0; word_len = lit_len = 0; }
#line 383 "rlscan.rl"
	{ word = tokdata; word_len = toklen; }
	goto st4;
st4:
	if ( ++p == pe )
		goto _test_eof4;
case 4:
#line 638 "rlscan.cpp"
	switch( (*p) ) {
		case 59: goto tr6;
		case 129: goto tr7;
	}
	goto tr3;
tr5:
#line 382 "rlscan.rl"
	{ word = lit = 0; word_len = lit_len = 0; }
#line 384 "rlscan.rl"
	{ lit = tokdata; lit_len = toklen; }
	goto st5;
tr7:
#line 384 "rlscan.rl"
	{ lit = tokdata; lit_len = toklen; }
	goto st5;
st5:
	if ( ++p == pe )
		goto _test_eof5;
case 5:
#line 658 "rlscan.cpp"
	if ( (*p) == 59 )
		goto tr6;
	goto tr3;
st6:
	if ( ++p == pe )
		goto _test_eof6;
case 6:
	if ( (*p) == 129 )
		goto tr9;
	goto tr8;
tr9:
#line 384 "rlscan.rl"
	{ lit = tokdata; lit_len = toklen; }
	goto st7;
st7:
	if ( ++p == pe )
		goto _test_eof7;
case 7:
#line 677 "rlscan.cpp"
	if ( (*p) == 59 )
		goto tr10;
	goto tr8;
tr18:
#line 413 "rlscan.rl"
	{
		if ( active() && machineSpec == 0 && machineName == 0 ) {
			InputItem *inputItem = new InputItem;
			inputItem->type = InputItem::Write;
			inputItem->loc.fileName = fileName;
			inputItem->loc.line = line;
			inputItem->loc.col = column;
			inputItem->name = parser->sectionName;
			inputItem->pd = parser->pd;
			id.inputItems.append( inputItem );
		}
	}
	goto st8;
st8:
	if ( ++p == pe )
		goto _test_eof8;
case 8:
#line 700 "rlscan.cpp"
	if ( (*p) == 128 )
		goto tr12;
	goto tr11;
tr12:
#line 427 "rlscan.rl"
	{
		if ( active() && machineSpec == 0 && machineName == 0 )
			id.inputItems.tail->writeArgs.append( strdup(tokdata) );
	}
	goto st9;
st9:
	if ( ++p == pe )
		goto _test_eof9;
case 9:
#line 715 "rlscan.cpp"
	switch( (*p) ) {
		case 59: goto tr13;
		case 128: goto tr12;
	}
	goto tr11;
	}
	_test_eof10: cs = 10; goto _test_eof; 
	_test_eof1: cs = 1; goto _test_eof; 
	_test_eof2: cs = 2; goto _test_eof; 
	_test_eof3: cs = 3; goto _test_eof; 
	_test_eof4: cs = 4; goto _test_eof; 
	_test_eof5: cs = 5; goto _test_eof; 
	_test_eof6: cs = 6; goto _test_eof; 
	_test_eof7: cs = 7; goto _test_eof; 
	_test_eof8: cs = 8; goto _test_eof; 
	_test_eof9: cs = 9; goto _test_eof; 

	_test_eof: {}
	if ( p == eof )
	{
	switch ( cs ) {
	case 1: 
	case 2: 
#line 386 "rlscan.rl"
	{ scan_error() << "bad machine statement" << endl; }
	break;
	case 3: 
	case 4: 
	case 5: 
#line 387 "rlscan.rl"
	{ scan_error() << "bad include statement" << endl; }
	break;
	case 6: 
	case 7: 
#line 388 "rlscan.rl"
	{ scan_error() << "bad import statement" << endl; }
	break;
	case 8: 
	case 9: 
#line 389 "rlscan.rl"
	{ scan_error() << "bad write statement" << endl; }
	break;
#line 758 "rlscan.cpp"
	}
	}

	_out: {}
	}

#line 502 "rlscan.rl"


	updateCol();

	/* Record the last token for use in controlling the scan of subsequent
	 * tokens. */
	lastToken = type;
}

void Scanner::startSection( )
{
	parserExistsError = false;

	sectionLoc.fileName = fileName;
	sectionLoc.line = line;
	sectionLoc.col = column;
}

void Scanner::endSection( )
{
	/* Execute the eof actions for the section parser. */
	processToken( -1, 0, 0 );

	/* Close off the section with the parser. */
	if ( active() ) {
		InputLoc loc;
		loc.fileName = fileName;
		loc.line = line;
		loc.col = column;

		parser->token( loc, TK_EndSection, 0, 0 );
	}

	if ( includeDepth == 0 ) {
		if ( machineSpec == 0 && machineName == 0 ) {
			/* The end section may include a newline on the end, so
			 * we use the last line, which will count the newline. */
			InputItem *inputItem = new InputItem;
			inputItem->type = InputItem::HostData;
			inputItem->loc.line = line;
			inputItem->loc.col = column;
			id.inputItems.append( inputItem );
		}
	}
}

bool isAbsolutePath( const char *path )
{
#ifdef _WIN32
	return isalpha( path[0] ) && path[1] == ':' && (path[2] == '\\' || path[2] == '/');
#else
	return path[0] == '/';
#endif
}

inline char* resolvePath(const char* rel, const char* abs) {
    const size_t l1 = strlen(rel);
    const size_t l2 = strlen(abs);
    char* ret = new char[l1 + l2 + 1];

    const char* p = strrchr(abs, '/') + 1;
    const size_t l3 = p - abs;

    memcpy(ret, abs, l3);
    strcpy(ret + l3, rel);

    return ret;
}

char **Scanner::makeIncludePathChecks( const char *thisFileName, 
		const char *fileName, int fnlen )
{
	char **checks = 0;
	long nextCheck = 0;
	long length = 0;
	bool caseInsensitive = false;
	char *data = prepareLitString( InputLoc(), fileName, fnlen, 
			length, caseInsensitive );

	/* Absolute path? */
	if ( isAbsolutePath( data ) ) {
		checks = new char*[2];
		checks[nextCheck++] = data;
	}
	else {
		checks = new char*[2 + id.includePaths.length()];

		/* Search from the the location of the current file. */
		const char *lastSlash = strrchr( thisFileName, '/' );
		if ( lastSlash == 0 )
			checks[nextCheck++] = data;
		else {
			checks[nextCheck++] = resolvePath(data, thisFileName);
		}

		/* Search from the include paths given on the command line. */
		for ( ArgsVector::Iter incp = id.includePaths; incp.lte(); incp++ ) {
			long pathLen = strlen( *incp );
			long checkLen = pathLen + 1 + length;
			char *check = new char[checkLen+1];
			memcpy( check, *incp, pathLen );
			check[pathLen] = '/';
			memcpy( check+pathLen+1, data, length );
			check[checkLen] = 0;
			checks[nextCheck++] = check;
		}
	}

	checks[nextCheck] = 0;
	return checks;
}

ifstream *Scanner::tryOpenInclude( char **pathChecks, long &found )
{
	char **check = pathChecks;
	ifstream *inFile = new ifstream;
	
	while ( *check != 0 ) {
		inFile->open( *check );
		if ( inFile->is_open() ) {
			found = check - pathChecks;
			return inFile;
		}

		/* 
		 * 03/26/2011 jg:
		 * Don't rely on sloppy runtime behaviour: reset the state of the stream explicitly.
		 * If inFile->open() fails, which happens when include dirs are tested, the fail bit
		 * is set by the runtime library. Currently the VS runtime library opens new files,
		 * but when it comes to reading it refuses to work.
		 */
		inFile->clear();

		check += 1;
	}

	found = -1;
	delete inFile;
	return 0;
}


#line 1173 "rlscan.rl"



#line 904 "rlscan.cpp"
static const int rlscan_start = 38;
static const int rlscan_first_final = 38;
static const int rlscan_error = 0;

static const int rlscan_en_inline_code_ruby = 52;
static const int rlscan_en_inline_code = 95;
static const int rlscan_en_or_literal = 137;
static const int rlscan_en_ragel_re_literal = 139;
static const int rlscan_en_write_statement = 143;
static const int rlscan_en_parser_def = 146;
static const int rlscan_en_main_ruby = 253;
static const int rlscan_en_main = 38;


#line 1176 "rlscan.rl"

void Scanner::do_scan()
{
	int bufsize = 8;
	char *buf = new char[bufsize];
	int cs, act, have = 0;
	int top;

	/* The stack is two deep, one level for going into ragel defs from the main
	 * machines which process outside code, and another for going into or literals
	 * from either a ragel spec, or a regular expression. */
	int stack[2];
	int curly_count = 0;
	bool execute = true;
	bool singleLineSpec = false;
	InlineBlockType inlineBlockType = CurlyDelimited;

	/* Init the section parser and the character scanner. */
	init();
	
#line 940 "rlscan.cpp"
	{
	cs = rlscan_start;
	top = 0;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 1196 "rlscan.rl"

	/* Set up the start state. FIXME: After 5.20 is released the nocs write
	 * init option should be used, the main machine eliminated and this statement moved
	 * above the write init. */
	if ( hostLang->lang == HostLang::Ruby )
		cs = rlscan_en_main_ruby;
	else
		cs = rlscan_en_main;
	
	while ( execute ) {
		char *p = buf + have;
		int space = bufsize - have;

		if ( space == 0 ) {
			/* We filled up the buffer trying to scan a token. Grow it. */
			bufsize = bufsize * 2;
			char *newbuf = new char[bufsize];

			/* Recompute p and space. */
			p = newbuf + have;
			space = bufsize - have;

			/* Patch up pointers possibly in use. */
			if ( ts != 0 )
				ts = newbuf + ( ts - buf );
			te = newbuf + ( te - buf );

			/* Copy the new buffer in. */
			memcpy( newbuf, buf, have );
			delete[] buf;
			buf = newbuf;
		}

		input.read( p, space );
		int len = input.gcount();
		char *pe = p + len;

		/* If we see eof then append the eof var. */
		char *eof = 0;
	 	if ( len == 0 ) {
			eof = pe;
			execute = false;
		}

		
#line 995 "rlscan.cpp"
	{
	if ( p == pe )
		goto _test_eof;
	goto _resume;

_again:
	switch ( cs ) {
		case 38: goto st38;
		case 39: goto st39;
		case 40: goto st40;
		case 1: goto st1;
		case 2: goto st2;
		case 41: goto st41;
		case 42: goto st42;
		case 43: goto st43;
		case 3: goto st3;
		case 4: goto st4;
		case 44: goto st44;
		case 5: goto st5;
		case 6: goto st6;
		case 7: goto st7;
		case 45: goto st45;
		case 46: goto st46;
		case 47: goto st47;
		case 48: goto st48;
		case 49: goto st49;
		case 50: goto st50;
		case 51: goto st51;
		case 52: goto st52;
		case 53: goto st53;
		case 54: goto st54;
		case 8: goto st8;
		case 9: goto st9;
		case 55: goto st55;
		case 10: goto st10;
		case 56: goto st56;
		case 11: goto st11;
		case 12: goto st12;
		case 57: goto st57;
		case 13: goto st13;
		case 14: goto st14;
		case 58: goto st58;
		case 59: goto st59;
		case 15: goto st15;
		case 60: goto st60;
		case 61: goto st61;
		case 62: goto st62;
		case 63: goto st63;
		case 64: goto st64;
		case 65: goto st65;
		case 66: goto st66;
		case 67: goto st67;
		case 68: goto st68;
		case 69: goto st69;
		case 70: goto st70;
		case 71: goto st71;
		case 72: goto st72;
		case 73: goto st73;
		case 74: goto st74;
		case 75: goto st75;
		case 76: goto st76;
		case 77: goto st77;
		case 78: goto st78;
		case 79: goto st79;
		case 80: goto st80;
		case 81: goto st81;
		case 82: goto st82;
		case 83: goto st83;
		case 84: goto st84;
		case 85: goto st85;
		case 86: goto st86;
		case 87: goto st87;
		case 88: goto st88;
		case 89: goto st89;
		case 90: goto st90;
		case 91: goto st91;
		case 92: goto st92;
		case 93: goto st93;
		case 94: goto st94;
		case 95: goto st95;
		case 96: goto st96;
		case 97: goto st97;
		case 16: goto st16;
		case 17: goto st17;
		case 98: goto st98;
		case 18: goto st18;
		case 19: goto st19;
		case 99: goto st99;
		case 20: goto st20;
		case 21: goto st21;
		case 22: goto st22;
		case 100: goto st100;
		case 101: goto st101;
		case 23: goto st23;
		case 102: goto st102;
		case 103: goto st103;
		case 104: goto st104;
		case 105: goto st105;
		case 106: goto st106;
		case 107: goto st107;
		case 108: goto st108;
		case 109: goto st109;
		case 110: goto st110;
		case 111: goto st111;
		case 112: goto st112;
		case 113: goto st113;
		case 114: goto st114;
		case 115: goto st115;
		case 116: goto st116;
		case 117: goto st117;
		case 118: goto st118;
		case 119: goto st119;
		case 120: goto st120;
		case 121: goto st121;
		case 122: goto st122;
		case 123: goto st123;
		case 124: goto st124;
		case 125: goto st125;
		case 126: goto st126;
		case 127: goto st127;
		case 128: goto st128;
		case 129: goto st129;
		case 130: goto st130;
		case 131: goto st131;
		case 132: goto st132;
		case 133: goto st133;
		case 134: goto st134;
		case 135: goto st135;
		case 136: goto st136;
		case 137: goto st137;
		case 138: goto st138;
		case 139: goto st139;
		case 140: goto st140;
		case 141: goto st141;
		case 142: goto st142;
		case 143: goto st143;
		case 0: goto st0;
		case 144: goto st144;
		case 145: goto st145;
		case 146: goto st146;
		case 147: goto st147;
		case 148: goto st148;
		case 24: goto st24;
		case 149: goto st149;
		case 25: goto st25;
		case 150: goto st150;
		case 26: goto st26;
		case 151: goto st151;
		case 152: goto st152;
		case 153: goto st153;
		case 27: goto st27;
		case 28: goto st28;
		case 154: goto st154;
		case 155: goto st155;
		case 156: goto st156;
		case 157: goto st157;
		case 158: goto st158;
		case 29: goto st29;
		case 159: goto st159;
		case 160: goto st160;
		case 161: goto st161;
		case 162: goto st162;
		case 163: goto st163;
		case 164: goto st164;
		case 165: goto st165;
		case 166: goto st166;
		case 167: goto st167;
		case 168: goto st168;
		case 169: goto st169;
		case 170: goto st170;
		case 171: goto st171;
		case 172: goto st172;
		case 173: goto st173;
		case 174: goto st174;
		case 175: goto st175;
		case 176: goto st176;
		case 177: goto st177;
		case 178: goto st178;
		case 179: goto st179;
		case 180: goto st180;
		case 181: goto st181;
		case 182: goto st182;
		case 183: goto st183;
		case 184: goto st184;
		case 185: goto st185;
		case 186: goto st186;
		case 187: goto st187;
		case 188: goto st188;
		case 189: goto st189;
		case 190: goto st190;
		case 191: goto st191;
		case 192: goto st192;
		case 193: goto st193;
		case 194: goto st194;
		case 195: goto st195;
		case 196: goto st196;
		case 197: goto st197;
		case 198: goto st198;
		case 199: goto st199;
		case 200: goto st200;
		case 201: goto st201;
		case 202: goto st202;
		case 203: goto st203;
		case 204: goto st204;
		case 205: goto st205;
		case 206: goto st206;
		case 207: goto st207;
		case 208: goto st208;
		case 209: goto st209;
		case 210: goto st210;
		case 211: goto st211;
		case 212: goto st212;
		case 213: goto st213;
		case 214: goto st214;
		case 215: goto st215;
		case 216: goto st216;
		case 217: goto st217;
		case 218: goto st218;
		case 219: goto st219;
		case 220: goto st220;
		case 221: goto st221;
		case 222: goto st222;
		case 223: goto st223;
		case 224: goto st224;
		case 225: goto st225;
		case 226: goto st226;
		case 227: goto st227;
		case 228: goto st228;
		case 229: goto st229;
		case 230: goto st230;
		case 231: goto st231;
		case 232: goto st232;
		case 233: goto st233;
		case 234: goto st234;
		case 235: goto st235;
		case 236: goto st236;
		case 237: goto st237;
		case 238: goto st238;
		case 239: goto st239;
		case 240: goto st240;
		case 241: goto st241;
		case 242: goto st242;
		case 243: goto st243;
		case 244: goto st244;
		case 245: goto st245;
		case 246: goto st246;
		case 247: goto st247;
		case 248: goto st248;
		case 249: goto st249;
		case 250: goto st250;
		case 251: goto st251;
		case 252: goto st252;
		case 30: goto st30;
		case 253: goto st253;
		case 254: goto st254;
		case 255: goto st255;
		case 31: goto st31;
		case 32: goto st32;
		case 256: goto st256;
		case 33: goto st33;
		case 257: goto st257;
		case 258: goto st258;
		case 259: goto st259;
		case 34: goto st34;
		case 35: goto st35;
		case 260: goto st260;
		case 36: goto st36;
		case 37: goto st37;
		case 261: goto st261;
		case 262: goto st262;
	default: break;
	}

	if ( ++p == pe )
		goto _test_eof;
_resume:
	switch ( cs )
	{
tr0:
#line 1171 "rlscan.rl"
	{{p = ((te))-1;}{ pass( *ts, 0, 0 ); }}
	goto st38;
tr3:
#line 1155 "rlscan.rl"
	{te = p+1;{ pass( IMP_Literal, ts, te ); }}
	goto st38;
tr11:
#line 1154 "rlscan.rl"
	{te = p+1;{ pass(); }}
	goto st38;
tr13:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
#line 1154 "rlscan.rl"
	{te = p+1;{ pass(); }}
	goto st38;
tr71:
#line 1171 "rlscan.rl"
	{te = p+1;{ pass( *ts, 0, 0 ); }}
	goto st38;
tr72:
#line 1170 "rlscan.rl"
	{te = p+1;}
	goto st38;
tr82:
#line 1169 "rlscan.rl"
	{te = p;p--;{ pass(); }}
	goto st38;
tr83:
#line 1171 "rlscan.rl"
	{te = p;p--;{ pass( *ts, 0, 0 ); }}
	goto st38;
tr85:
#line 1163 "rlscan.rl"
	{te = p;p--;{ 
			updateCol();
			singleLineSpec = true;
			startSection();
			{stack[top++] = 38; goto st146;}
		}}
	goto st38;
tr86:
#line 1157 "rlscan.rl"
	{te = p+1;{ 
			updateCol();
			singleLineSpec = false;
			startSection();
			{stack[top++] = 38; goto st146;}
		}}
	goto st38;
tr87:
#line 1153 "rlscan.rl"
	{te = p;p--;{ pass( IMP_UInt, ts, te ); }}
	goto st38;
tr88:
#line 1 "NONE"
	{	switch( act ) {
	case 176:
	{{p = ((te))-1;} pass( IMP_Define, 0, 0 ); }
	break;
	case 177:
	{{p = ((te))-1;} pass( IMP_Word, ts, te ); }
	break;
	}
	}
	goto st38;
tr89:
#line 1152 "rlscan.rl"
	{te = p;p--;{ pass( IMP_Word, ts, te ); }}
	goto st38;
st38:
#line 1 "NONE"
	{ts = 0;}
	if ( ++p == pe )
		goto _test_eof38;
case 38:
#line 1 "NONE"
	{ts = p;}
#line 1358 "rlscan.cpp"
	switch( (*p) ) {
		case 0: goto tr72;
		case 9: goto st39;
		case 10: goto tr74;
		case 32: goto st39;
		case 34: goto tr75;
		case 37: goto st41;
		case 39: goto tr77;
		case 47: goto tr78;
		case 95: goto tr80;
		case 100: goto st47;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st45;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr80;
	} else
		goto tr80;
	goto tr71;
tr74:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
	goto st39;
st39:
	if ( ++p == pe )
		goto _test_eof39;
case 39:
#line 1392 "rlscan.cpp"
	switch( (*p) ) {
		case 9: goto st39;
		case 10: goto tr74;
		case 32: goto st39;
	}
	goto tr82;
tr75:
#line 1 "NONE"
	{te = p+1;}
	goto st40;
st40:
	if ( ++p == pe )
		goto _test_eof40;
case 40:
#line 1407 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr2;
		case 34: goto tr3;
		case 92: goto st2;
	}
	goto st1;
tr2:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
	goto st1;
st1:
	if ( ++p == pe )
		goto _test_eof1;
case 1:
#line 1426 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr2;
		case 34: goto tr3;
		case 92: goto st2;
	}
	goto st1;
st2:
	if ( ++p == pe )
		goto _test_eof2;
case 2:
	if ( (*p) == 10 )
		goto tr2;
	goto st1;
st41:
	if ( ++p == pe )
		goto _test_eof41;
case 41:
	if ( (*p) == 37 )
		goto st42;
	goto tr83;
st42:
	if ( ++p == pe )
		goto _test_eof42;
case 42:
	if ( (*p) == 123 )
		goto tr86;
	goto tr85;
tr77:
#line 1 "NONE"
	{te = p+1;}
	goto st43;
st43:
	if ( ++p == pe )
		goto _test_eof43;
case 43:
#line 1462 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr6;
		case 39: goto tr3;
		case 92: goto st4;
	}
	goto st3;
tr6:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
	goto st3;
st3:
	if ( ++p == pe )
		goto _test_eof3;
case 3:
#line 1481 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr6;
		case 39: goto tr3;
		case 92: goto st4;
	}
	goto st3;
st4:
	if ( ++p == pe )
		goto _test_eof4;
case 4:
	if ( (*p) == 10 )
		goto tr6;
	goto st3;
tr78:
#line 1 "NONE"
	{te = p+1;}
	goto st44;
st44:
	if ( ++p == pe )
		goto _test_eof44;
case 44:
#line 1503 "rlscan.cpp"
	switch( (*p) ) {
		case 42: goto st5;
		case 47: goto st7;
	}
	goto tr83;
tr9:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
	goto st5;
st5:
	if ( ++p == pe )
		goto _test_eof5;
case 5:
#line 1521 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr9;
		case 42: goto st6;
	}
	goto st5;
st6:
	if ( ++p == pe )
		goto _test_eof6;
case 6:
	switch( (*p) ) {
		case 10: goto tr9;
		case 42: goto st6;
		case 47: goto tr11;
	}
	goto st5;
st7:
	if ( ++p == pe )
		goto _test_eof7;
case 7:
	if ( (*p) == 10 )
		goto tr13;
	goto st7;
st45:
	if ( ++p == pe )
		goto _test_eof45;
case 45:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto st45;
	goto tr87;
tr80:
#line 1 "NONE"
	{te = p+1;}
#line 1152 "rlscan.rl"
	{act = 177;}
	goto st46;
tr94:
#line 1 "NONE"
	{te = p+1;}
#line 1151 "rlscan.rl"
	{act = 176;}
	goto st46;
st46:
	if ( ++p == pe )
		goto _test_eof46;
case 46:
#line 1567 "rlscan.cpp"
	if ( (*p) == 95 )
		goto tr80;
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr80;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr80;
	} else
		goto tr80;
	goto tr88;
st47:
	if ( ++p == pe )
		goto _test_eof47;
case 47:
	switch( (*p) ) {
		case 95: goto tr80;
		case 101: goto st48;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr80;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr80;
	} else
		goto tr80;
	goto tr89;
st48:
	if ( ++p == pe )
		goto _test_eof48;
case 48:
	switch( (*p) ) {
		case 95: goto tr80;
		case 102: goto st49;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr80;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr80;
	} else
		goto tr80;
	goto tr89;
st49:
	if ( ++p == pe )
		goto _test_eof49;
case 49:
	switch( (*p) ) {
		case 95: goto tr80;
		case 105: goto st50;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr80;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr80;
	} else
		goto tr80;
	goto tr89;
st50:
	if ( ++p == pe )
		goto _test_eof50;
case 50:
	switch( (*p) ) {
		case 95: goto tr80;
		case 110: goto st51;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr80;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr80;
	} else
		goto tr80;
	goto tr89;
st51:
	if ( ++p == pe )
		goto _test_eof51;
case 51:
	switch( (*p) ) {
		case 95: goto tr80;
		case 101: goto tr94;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr80;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr80;
	} else
		goto tr80;
	goto tr89;
tr14:
#line 770 "rlscan.rl"
	{{p = ((te))-1;}{ token( IL_Symbol, ts, te ); }}
	goto st52;
tr17:
#line 716 "rlscan.rl"
	{te = p+1;{ token( IL_Literal, ts, te ); }}
	goto st52;
tr20:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
#line 723 "rlscan.rl"
	{te = p+1;{ token( IL_Comment, ts, te ); }}
	goto st52;
tr27:
#line 712 "rlscan.rl"
	{{p = ((te))-1;}{ token( TK_UInt, ts, te ); }}
	goto st52;
tr95:
#line 770 "rlscan.rl"
	{te = p+1;{ token( IL_Symbol, ts, te ); }}
	goto st52;
tr96:
#line 765 "rlscan.rl"
	{te = p+1;{
			scan_error() << "unterminated code block" << endl;
		}}
	goto st52;
tr102:
#line 745 "rlscan.rl"
	{te = p+1;{ token( *ts, ts, te ); }}
	goto st52;
tr103:
#line 740 "rlscan.rl"
	{te = p+1;{ 
			whitespaceOn = true;
			token( *ts, ts, te );
		}}
	goto st52;
tr108:
#line 733 "rlscan.rl"
	{te = p+1;{
			whitespaceOn = true;
			token( *ts, ts, te );
			if ( inlineBlockType == SemiTerminated )
				{cs = stack[--top];goto _again;}
		}}
	goto st52;
tr111:
#line 747 "rlscan.rl"
	{te = p+1;{ 
			token( IL_Symbol, ts, te );
			curly_count += 1; 
		}}
	goto st52;
tr112:
#line 752 "rlscan.rl"
	{te = p+1;{ 
			if ( --curly_count == 0 && inlineBlockType == CurlyDelimited ) {
				/* Inline code block ends. */
				token( '}' );
				{cs = stack[--top];goto _again;}
			}
			else {
				/* Either a semi terminated inline block or only the closing
				 * brace of some inner scope, not the block's closing brace. */
				token( IL_Symbol, ts, te );
			}
		}}
	goto st52;
tr113:
#line 718 "rlscan.rl"
	{te = p;p--;{ 
			if ( whitespaceOn ) 
				token( IL_WhiteSpace, ts, te );
		}}
	goto st52;
tr114:
#line 770 "rlscan.rl"
	{te = p;p--;{ token( IL_Symbol, ts, te ); }}
	goto st52;
tr115:
#line 712 "rlscan.rl"
	{te = p;p--;{ token( TK_UInt, ts, te ); }}
	goto st52;
tr117:
#line 713 "rlscan.rl"
	{te = p;p--;{ token( TK_Hex, ts, te ); }}
	goto st52;
tr118:
#line 725 "rlscan.rl"
	{te = p+1;{ token( TK_NameSep, ts, te ); }}
	goto st52;
tr119:
#line 1 "NONE"
	{	switch( act ) {
	case 1:
	{{p = ((te))-1;} token( KW_PChar ); }
	break;
	case 3:
	{{p = ((te))-1;} token( KW_CurState ); }
	break;
	case 4:
	{{p = ((te))-1;} token( KW_TargState ); }
	break;
	case 5:
	{{p = ((te))-1;} 
			whitespaceOn = false; 
			token( KW_Entry );
		}
	break;
	case 6:
	{{p = ((te))-1;} 
			whitespaceOn = false; 
			token( KW_Hold );
		}
	break;
	case 7:
	{{p = ((te))-1;} token( KW_Exec, 0, 0 ); }
	break;
	case 8:
	{{p = ((te))-1;} 
			whitespaceOn = false; 
			token( KW_Goto );
		}
	break;
	case 9:
	{{p = ((te))-1;} 
			whitespaceOn = false; 
			token( KW_Next );
		}
	break;
	case 10:
	{{p = ((te))-1;} 
			whitespaceOn = false; 
			token( KW_Call );
		}
	break;
	case 11:
	{{p = ((te))-1;} 
			whitespaceOn = false; 
			token( KW_Ret );
		}
	break;
	case 12:
	{{p = ((te))-1;} 
			whitespaceOn = false; 
			token( KW_Break );
		}
	break;
	case 13:
	{{p = ((te))-1;} token( TK_Word, ts, te ); }
	break;
	}
	}
	goto st52;
tr120:
#line 710 "rlscan.rl"
	{te = p;p--;{ token( TK_Word, ts, te ); }}
	goto st52;
tr134:
#line 675 "rlscan.rl"
	{te = p;p--;{ token( KW_Char ); }}
	goto st52;
st52:
#line 1 "NONE"
	{ts = 0;}
	if ( ++p == pe )
		goto _test_eof52;
case 52:
#line 1 "NONE"
	{ts = p;}
#line 1840 "rlscan.cpp"
	switch( (*p) ) {
		case 0: goto tr96;
		case 9: goto st53;
		case 10: goto tr98;
		case 32: goto st53;
		case 34: goto tr99;
		case 35: goto tr100;
		case 39: goto tr101;
		case 40: goto tr102;
		case 44: goto tr102;
		case 47: goto tr104;
		case 48: goto tr105;
		case 58: goto st61;
		case 59: goto tr108;
		case 95: goto tr109;
		case 102: goto st63;
		case 123: goto tr111;
		case 125: goto tr112;
	}
	if ( (*p) < 49 ) {
		if ( 41 <= (*p) && (*p) <= 42 )
			goto tr103;
	} else if ( (*p) > 57 ) {
		if ( (*p) > 90 ) {
			if ( 97 <= (*p) && (*p) <= 122 )
				goto tr109;
		} else if ( (*p) >= 65 )
			goto tr109;
	} else
		goto st59;
	goto tr95;
tr98:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
	goto st53;
st53:
	if ( ++p == pe )
		goto _test_eof53;
case 53:
#line 1884 "rlscan.cpp"
	switch( (*p) ) {
		case 9: goto st53;
		case 10: goto tr98;
		case 32: goto st53;
	}
	goto tr113;
tr99:
#line 1 "NONE"
	{te = p+1;}
	goto st54;
st54:
	if ( ++p == pe )
		goto _test_eof54;
case 54:
#line 1899 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr16;
		case 34: goto tr17;
		case 92: goto st9;
	}
	goto st8;
tr16:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
	goto st8;
st8:
	if ( ++p == pe )
		goto _test_eof8;
case 8:
#line 1918 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr16;
		case 34: goto tr17;
		case 92: goto st9;
	}
	goto st8;
st9:
	if ( ++p == pe )
		goto _test_eof9;
case 9:
	if ( (*p) == 10 )
		goto tr16;
	goto st8;
tr100:
#line 1 "NONE"
	{te = p+1;}
	goto st55;
st55:
	if ( ++p == pe )
		goto _test_eof55;
case 55:
#line 1940 "rlscan.cpp"
	if ( (*p) == 10 )
		goto tr20;
	goto st10;
st10:
	if ( ++p == pe )
		goto _test_eof10;
case 10:
	if ( (*p) == 10 )
		goto tr20;
	goto st10;
tr101:
#line 1 "NONE"
	{te = p+1;}
	goto st56;
st56:
	if ( ++p == pe )
		goto _test_eof56;
case 56:
#line 1959 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr22;
		case 39: goto tr17;
		case 92: goto st12;
	}
	goto st11;
tr22:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
	goto st11;
st11:
	if ( ++p == pe )
		goto _test_eof11;
case 11:
#line 1978 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr22;
		case 39: goto tr17;
		case 92: goto st12;
	}
	goto st11;
st12:
	if ( ++p == pe )
		goto _test_eof12;
case 12:
	if ( (*p) == 10 )
		goto tr22;
	goto st11;
tr104:
#line 1 "NONE"
	{te = p+1;}
	goto st57;
st57:
	if ( ++p == pe )
		goto _test_eof57;
case 57:
#line 2000 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr25;
		case 47: goto tr17;
		case 92: goto st14;
	}
	goto st13;
tr25:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
	goto st13;
st13:
	if ( ++p == pe )
		goto _test_eof13;
case 13:
#line 2019 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr25;
		case 47: goto tr17;
		case 92: goto st14;
	}
	goto st13;
st14:
	if ( ++p == pe )
		goto _test_eof14;
case 14:
	if ( (*p) == 10 )
		goto tr25;
	goto st13;
tr105:
#line 1 "NONE"
	{te = p+1;}
	goto st58;
st58:
	if ( ++p == pe )
		goto _test_eof58;
case 58:
#line 2041 "rlscan.cpp"
	if ( (*p) == 120 )
		goto st15;
	if ( 48 <= (*p) && (*p) <= 57 )
		goto st59;
	goto tr115;
st59:
	if ( ++p == pe )
		goto _test_eof59;
case 59:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto st59;
	goto tr115;
st15:
	if ( ++p == pe )
		goto _test_eof15;
case 15:
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st60;
	} else if ( (*p) > 70 ) {
		if ( 97 <= (*p) && (*p) <= 102 )
			goto st60;
	} else
		goto st60;
	goto tr27;
st60:
	if ( ++p == pe )
		goto _test_eof60;
case 60:
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st60;
	} else if ( (*p) > 70 ) {
		if ( 97 <= (*p) && (*p) <= 102 )
			goto st60;
	} else
		goto st60;
	goto tr117;
st61:
	if ( ++p == pe )
		goto _test_eof61;
case 61:
	if ( (*p) == 58 )
		goto tr118;
	goto tr114;
tr109:
#line 1 "NONE"
	{te = p+1;}
#line 710 "rlscan.rl"
	{act = 13;}
	goto st62;
tr133:
#line 1 "NONE"
	{te = p+1;}
#line 705 "rlscan.rl"
	{act = 12;}
	goto st62;
tr138:
#line 1 "NONE"
	{te = p+1;}
#line 697 "rlscan.rl"
	{act = 10;}
	goto st62;
tr140:
#line 1 "NONE"
	{te = p+1;}
#line 676 "rlscan.rl"
	{act = 3;}
	goto st62;
tr145:
#line 1 "NONE"
	{te = p+1;}
#line 678 "rlscan.rl"
	{act = 5;}
	goto st62;
tr147:
#line 1 "NONE"
	{te = p+1;}
#line 688 "rlscan.rl"
	{act = 7;}
	goto st62;
tr150:
#line 1 "NONE"
	{te = p+1;}
#line 689 "rlscan.rl"
	{act = 8;}
	goto st62;
tr153:
#line 1 "NONE"
	{te = p+1;}
#line 684 "rlscan.rl"
	{act = 6;}
	goto st62;
tr156:
#line 1 "NONE"
	{te = p+1;}
#line 693 "rlscan.rl"
	{act = 9;}
	goto st62;
tr157:
#line 1 "NONE"
	{te = p+1;}
#line 674 "rlscan.rl"
	{act = 1;}
	goto st62;
tr159:
#line 1 "NONE"
	{te = p+1;}
#line 701 "rlscan.rl"
	{act = 11;}
	goto st62;
tr163:
#line 1 "NONE"
	{te = p+1;}
#line 677 "rlscan.rl"
	{act = 4;}
	goto st62;
st62:
	if ( ++p == pe )
		goto _test_eof62;
case 62:
#line 2163 "rlscan.cpp"
	if ( (*p) == 95 )
		goto tr109;
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr119;
st63:
	if ( ++p == pe )
		goto _test_eof63;
case 63:
	switch( (*p) ) {
		case 95: goto tr109;
		case 98: goto st64;
		case 99: goto st68;
		case 101: goto st73;
		case 103: goto st79;
		case 104: goto st82;
		case 110: goto st85;
		case 112: goto st88;
		case 114: goto st89;
		case 116: goto st91;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st64:
	if ( ++p == pe )
		goto _test_eof64;
case 64:
	switch( (*p) ) {
		case 95: goto tr109;
		case 114: goto st65;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st65:
	if ( ++p == pe )
		goto _test_eof65;
case 65:
	switch( (*p) ) {
		case 95: goto tr109;
		case 101: goto st66;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st66:
	if ( ++p == pe )
		goto _test_eof66;
case 66:
	switch( (*p) ) {
		case 95: goto tr109;
		case 97: goto st67;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 98 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st67:
	if ( ++p == pe )
		goto _test_eof67;
case 67:
	switch( (*p) ) {
		case 95: goto tr109;
		case 107: goto tr133;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st68:
	if ( ++p == pe )
		goto _test_eof68;
case 68:
	switch( (*p) ) {
		case 95: goto tr109;
		case 97: goto st69;
		case 117: goto st71;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 98 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr134;
st69:
	if ( ++p == pe )
		goto _test_eof69;
case 69:
	switch( (*p) ) {
		case 95: goto tr109;
		case 108: goto st70;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st70:
	if ( ++p == pe )
		goto _test_eof70;
case 70:
	switch( (*p) ) {
		case 95: goto tr109;
		case 108: goto tr138;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st71:
	if ( ++p == pe )
		goto _test_eof71;
case 71:
	switch( (*p) ) {
		case 95: goto tr109;
		case 114: goto st72;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st72:
	if ( ++p == pe )
		goto _test_eof72;
case 72:
	switch( (*p) ) {
		case 95: goto tr109;
		case 115: goto tr140;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st73:
	if ( ++p == pe )
		goto _test_eof73;
case 73:
	switch( (*p) ) {
		case 95: goto tr109;
		case 110: goto st74;
		case 120: goto st77;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st74:
	if ( ++p == pe )
		goto _test_eof74;
case 74:
	switch( (*p) ) {
		case 95: goto tr109;
		case 116: goto st75;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st75:
	if ( ++p == pe )
		goto _test_eof75;
case 75:
	switch( (*p) ) {
		case 95: goto tr109;
		case 114: goto st76;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st76:
	if ( ++p == pe )
		goto _test_eof76;
case 76:
	switch( (*p) ) {
		case 95: goto tr109;
		case 121: goto tr145;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st77:
	if ( ++p == pe )
		goto _test_eof77;
case 77:
	switch( (*p) ) {
		case 95: goto tr109;
		case 101: goto st78;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st78:
	if ( ++p == pe )
		goto _test_eof78;
case 78:
	switch( (*p) ) {
		case 95: goto tr109;
		case 99: goto tr147;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st79:
	if ( ++p == pe )
		goto _test_eof79;
case 79:
	switch( (*p) ) {
		case 95: goto tr109;
		case 111: goto st80;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st80:
	if ( ++p == pe )
		goto _test_eof80;
case 80:
	switch( (*p) ) {
		case 95: goto tr109;
		case 116: goto st81;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st81:
	if ( ++p == pe )
		goto _test_eof81;
case 81:
	switch( (*p) ) {
		case 95: goto tr109;
		case 111: goto tr150;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st82:
	if ( ++p == pe )
		goto _test_eof82;
case 82:
	switch( (*p) ) {
		case 95: goto tr109;
		case 111: goto st83;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st83:
	if ( ++p == pe )
		goto _test_eof83;
case 83:
	switch( (*p) ) {
		case 95: goto tr109;
		case 108: goto st84;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st84:
	if ( ++p == pe )
		goto _test_eof84;
case 84:
	switch( (*p) ) {
		case 95: goto tr109;
		case 100: goto tr153;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st85:
	if ( ++p == pe )
		goto _test_eof85;
case 85:
	switch( (*p) ) {
		case 95: goto tr109;
		case 101: goto st86;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st86:
	if ( ++p == pe )
		goto _test_eof86;
case 86:
	switch( (*p) ) {
		case 95: goto tr109;
		case 120: goto st87;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st87:
	if ( ++p == pe )
		goto _test_eof87;
case 87:
	switch( (*p) ) {
		case 95: goto tr109;
		case 116: goto tr156;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st88:
	if ( ++p == pe )
		goto _test_eof88;
case 88:
	switch( (*p) ) {
		case 95: goto tr109;
		case 99: goto tr157;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st89:
	if ( ++p == pe )
		goto _test_eof89;
case 89:
	switch( (*p) ) {
		case 95: goto tr109;
		case 101: goto st90;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st90:
	if ( ++p == pe )
		goto _test_eof90;
case 90:
	switch( (*p) ) {
		case 95: goto tr109;
		case 116: goto tr159;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st91:
	if ( ++p == pe )
		goto _test_eof91;
case 91:
	switch( (*p) ) {
		case 95: goto tr109;
		case 97: goto st92;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 98 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st92:
	if ( ++p == pe )
		goto _test_eof92;
case 92:
	switch( (*p) ) {
		case 95: goto tr109;
		case 114: goto st93;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st93:
	if ( ++p == pe )
		goto _test_eof93;
case 93:
	switch( (*p) ) {
		case 95: goto tr109;
		case 103: goto st94;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
st94:
	if ( ++p == pe )
		goto _test_eof94;
case 94:
	switch( (*p) ) {
		case 95: goto tr109;
		case 115: goto tr163;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr109;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr109;
	} else
		goto tr109;
	goto tr120;
tr29:
#line 873 "rlscan.rl"
	{{p = ((te))-1;}{ token( IL_Symbol, ts, te ); }}
	goto st95;
tr32:
#line 819 "rlscan.rl"
	{te = p+1;{ token( IL_Literal, ts, te ); }}
	goto st95;
tr40:
#line 826 "rlscan.rl"
	{te = p+1;{ token( IL_Comment, ts, te ); }}
	goto st95;
tr42:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
#line 826 "rlscan.rl"
	{te = p+1;{ token( IL_Comment, ts, te ); }}
	goto st95;
tr43:
#line 815 "rlscan.rl"
	{{p = ((te))-1;}{ token( TK_UInt, ts, te ); }}
	goto st95;
tr164:
#line 873 "rlscan.rl"
	{te = p+1;{ token( IL_Symbol, ts, te ); }}
	goto st95;
tr165:
#line 868 "rlscan.rl"
	{te = p+1;{
			scan_error() << "unterminated code block" << endl;
		}}
	goto st95;
tr170:
#line 848 "rlscan.rl"
	{te = p+1;{ token( *ts, ts, te ); }}
	goto st95;
tr171:
#line 843 "rlscan.rl"
	{te = p+1;{ 
			whitespaceOn = true;
			token( *ts, ts, te );
		}}
	goto st95;
tr176:
#line 836 "rlscan.rl"
	{te = p+1;{
			whitespaceOn = true;
			token( *ts, ts, te );
			if ( inlineBlockType == SemiTerminated )
				{cs = stack[--top];goto _again;}
		}}
	goto st95;
tr179:
#line 850 "rlscan.rl"
	{te = p+1;{ 
			token( IL_Symbol, ts, te );
			curly_count += 1; 
		}}
	goto st95;
tr180:
#line 855 "rlscan.rl"
	{te = p+1;{ 
			if ( --curly_count == 0 && inlineBlockType == CurlyDelimited ) {
				/* Inline code block ends. */
				token( '}' );
				{cs = stack[--top];goto _again;}
			}
			else {
				/* Either a semi terminated inline block or only the closing
				 * brace of some inner scope, not the block's closing brace. */
				token( IL_Symbol, ts, te );
			}
		}}
	goto st95;
tr181:
#line 821 "rlscan.rl"
	{te = p;p--;{ 
			if ( whitespaceOn ) 
				token( IL_WhiteSpace, ts, te );
		}}
	goto st95;
tr182:
#line 873 "rlscan.rl"
	{te = p;p--;{ token( IL_Symbol, ts, te ); }}
	goto st95;
tr183:
#line 815 "rlscan.rl"
	{te = p;p--;{ token( TK_UInt, ts, te ); }}
	goto st95;
tr185:
#line 816 "rlscan.rl"
	{te = p;p--;{ token( TK_Hex, ts, te ); }}
	goto st95;
tr186:
#line 828 "rlscan.rl"
	{te = p+1;{ token( TK_NameSep, ts, te ); }}
	goto st95;
tr187:
#line 1 "NONE"
	{	switch( act ) {
	case 27:
	{{p = ((te))-1;} token( KW_PChar ); }
	break;
	case 29:
	{{p = ((te))-1;} token( KW_CurState ); }
	break;
	case 30:
	{{p = ((te))-1;} token( KW_TargState ); }
	break;
	case 31:
	{{p = ((te))-1;} 
			whitespaceOn = false; 
			token( KW_Entry );
		}
	break;
	case 32:
	{{p = ((te))-1;} 
			whitespaceOn = false; 
			token( KW_Hold );
		}
	break;
	case 33:
	{{p = ((te))-1;} token( KW_Exec, 0, 0 ); }
	break;
	case 34:
	{{p = ((te))-1;} 
			whitespaceOn = false; 
			token( KW_Goto );
		}
	break;
	case 35:
	{{p = ((te))-1;} 
			whitespaceOn = false; 
			token( KW_Next );
		}
	break;
	case 36:
	{{p = ((te))-1;} 
			whitespaceOn = false; 
			token( KW_Call );
		}
	break;
	case 37:
	{{p = ((te))-1;} 
			whitespaceOn = false; 
			token( KW_Ret );
		}
	break;
	case 38:
	{{p = ((te))-1;} 
			whitespaceOn = false; 
			token( KW_Break );
		}
	break;
	case 39:
	{{p = ((te))-1;} token( TK_Word, ts, te ); }
	break;
	}
	}
	goto st95;
tr188:
#line 813 "rlscan.rl"
	{te = p;p--;{ token( TK_Word, ts, te ); }}
	goto st95;
tr202:
#line 778 "rlscan.rl"
	{te = p;p--;{ token( KW_Char ); }}
	goto st95;
st95:
#line 1 "NONE"
	{ts = 0;}
	if ( ++p == pe )
		goto _test_eof95;
case 95:
#line 1 "NONE"
	{ts = p;}
#line 2909 "rlscan.cpp"
	switch( (*p) ) {
		case 0: goto tr165;
		case 9: goto st96;
		case 10: goto tr167;
		case 32: goto st96;
		case 34: goto tr168;
		case 39: goto tr169;
		case 40: goto tr170;
		case 44: goto tr170;
		case 47: goto tr172;
		case 48: goto tr173;
		case 58: goto st103;
		case 59: goto tr176;
		case 95: goto tr177;
		case 102: goto st105;
		case 123: goto tr179;
		case 125: goto tr180;
	}
	if ( (*p) < 49 ) {
		if ( 41 <= (*p) && (*p) <= 42 )
			goto tr171;
	} else if ( (*p) > 57 ) {
		if ( (*p) > 90 ) {
			if ( 97 <= (*p) && (*p) <= 122 )
				goto tr177;
		} else if ( (*p) >= 65 )
			goto tr177;
	} else
		goto st101;
	goto tr164;
tr167:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
	goto st96;
st96:
	if ( ++p == pe )
		goto _test_eof96;
case 96:
#line 2952 "rlscan.cpp"
	switch( (*p) ) {
		case 9: goto st96;
		case 10: goto tr167;
		case 32: goto st96;
	}
	goto tr181;
tr168:
#line 1 "NONE"
	{te = p+1;}
	goto st97;
st97:
	if ( ++p == pe )
		goto _test_eof97;
case 97:
#line 2967 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr31;
		case 34: goto tr32;
		case 92: goto st17;
	}
	goto st16;
tr31:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
	goto st16;
st16:
	if ( ++p == pe )
		goto _test_eof16;
case 16:
#line 2986 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr31;
		case 34: goto tr32;
		case 92: goto st17;
	}
	goto st16;
st17:
	if ( ++p == pe )
		goto _test_eof17;
case 17:
	if ( (*p) == 10 )
		goto tr31;
	goto st16;
tr169:
#line 1 "NONE"
	{te = p+1;}
	goto st98;
st98:
	if ( ++p == pe )
		goto _test_eof98;
case 98:
#line 3008 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr35;
		case 39: goto tr32;
		case 92: goto st19;
	}
	goto st18;
tr35:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
	goto st18;
st18:
	if ( ++p == pe )
		goto _test_eof18;
case 18:
#line 3027 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr35;
		case 39: goto tr32;
		case 92: goto st19;
	}
	goto st18;
st19:
	if ( ++p == pe )
		goto _test_eof19;
case 19:
	if ( (*p) == 10 )
		goto tr35;
	goto st18;
tr172:
#line 1 "NONE"
	{te = p+1;}
	goto st99;
st99:
	if ( ++p == pe )
		goto _test_eof99;
case 99:
#line 3049 "rlscan.cpp"
	switch( (*p) ) {
		case 42: goto st20;
		case 47: goto st22;
	}
	goto tr182;
tr38:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
	goto st20;
st20:
	if ( ++p == pe )
		goto _test_eof20;
case 20:
#line 3067 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr38;
		case 42: goto st21;
	}
	goto st20;
st21:
	if ( ++p == pe )
		goto _test_eof21;
case 21:
	switch( (*p) ) {
		case 10: goto tr38;
		case 42: goto st21;
		case 47: goto tr40;
	}
	goto st20;
st22:
	if ( ++p == pe )
		goto _test_eof22;
case 22:
	if ( (*p) == 10 )
		goto tr42;
	goto st22;
tr173:
#line 1 "NONE"
	{te = p+1;}
	goto st100;
st100:
	if ( ++p == pe )
		goto _test_eof100;
case 100:
#line 3098 "rlscan.cpp"
	if ( (*p) == 120 )
		goto st23;
	if ( 48 <= (*p) && (*p) <= 57 )
		goto st101;
	goto tr183;
st101:
	if ( ++p == pe )
		goto _test_eof101;
case 101:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto st101;
	goto tr183;
st23:
	if ( ++p == pe )
		goto _test_eof23;
case 23:
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st102;
	} else if ( (*p) > 70 ) {
		if ( 97 <= (*p) && (*p) <= 102 )
			goto st102;
	} else
		goto st102;
	goto tr43;
st102:
	if ( ++p == pe )
		goto _test_eof102;
case 102:
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st102;
	} else if ( (*p) > 70 ) {
		if ( 97 <= (*p) && (*p) <= 102 )
			goto st102;
	} else
		goto st102;
	goto tr185;
st103:
	if ( ++p == pe )
		goto _test_eof103;
case 103:
	if ( (*p) == 58 )
		goto tr186;
	goto tr182;
tr177:
#line 1 "NONE"
	{te = p+1;}
#line 813 "rlscan.rl"
	{act = 39;}
	goto st104;
tr201:
#line 1 "NONE"
	{te = p+1;}
#line 808 "rlscan.rl"
	{act = 38;}
	goto st104;
tr206:
#line 1 "NONE"
	{te = p+1;}
#line 800 "rlscan.rl"
	{act = 36;}
	goto st104;
tr208:
#line 1 "NONE"
	{te = p+1;}
#line 779 "rlscan.rl"
	{act = 29;}
	goto st104;
tr213:
#line 1 "NONE"
	{te = p+1;}
#line 781 "rlscan.rl"
	{act = 31;}
	goto st104;
tr215:
#line 1 "NONE"
	{te = p+1;}
#line 791 "rlscan.rl"
	{act = 33;}
	goto st104;
tr218:
#line 1 "NONE"
	{te = p+1;}
#line 792 "rlscan.rl"
	{act = 34;}
	goto st104;
tr221:
#line 1 "NONE"
	{te = p+1;}
#line 787 "rlscan.rl"
	{act = 32;}
	goto st104;
tr224:
#line 1 "NONE"
	{te = p+1;}
#line 796 "rlscan.rl"
	{act = 35;}
	goto st104;
tr225:
#line 1 "NONE"
	{te = p+1;}
#line 777 "rlscan.rl"
	{act = 27;}
	goto st104;
tr227:
#line 1 "NONE"
	{te = p+1;}
#line 804 "rlscan.rl"
	{act = 37;}
	goto st104;
tr231:
#line 1 "NONE"
	{te = p+1;}
#line 780 "rlscan.rl"
	{act = 30;}
	goto st104;
st104:
	if ( ++p == pe )
		goto _test_eof104;
case 104:
#line 3220 "rlscan.cpp"
	if ( (*p) == 95 )
		goto tr177;
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr187;
st105:
	if ( ++p == pe )
		goto _test_eof105;
case 105:
	switch( (*p) ) {
		case 95: goto tr177;
		case 98: goto st106;
		case 99: goto st110;
		case 101: goto st115;
		case 103: goto st121;
		case 104: goto st124;
		case 110: goto st127;
		case 112: goto st130;
		case 114: goto st131;
		case 116: goto st133;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st106:
	if ( ++p == pe )
		goto _test_eof106;
case 106:
	switch( (*p) ) {
		case 95: goto tr177;
		case 114: goto st107;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st107:
	if ( ++p == pe )
		goto _test_eof107;
case 107:
	switch( (*p) ) {
		case 95: goto tr177;
		case 101: goto st108;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st108:
	if ( ++p == pe )
		goto _test_eof108;
case 108:
	switch( (*p) ) {
		case 95: goto tr177;
		case 97: goto st109;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 98 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st109:
	if ( ++p == pe )
		goto _test_eof109;
case 109:
	switch( (*p) ) {
		case 95: goto tr177;
		case 107: goto tr201;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st110:
	if ( ++p == pe )
		goto _test_eof110;
case 110:
	switch( (*p) ) {
		case 95: goto tr177;
		case 97: goto st111;
		case 117: goto st113;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 98 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr202;
st111:
	if ( ++p == pe )
		goto _test_eof111;
case 111:
	switch( (*p) ) {
		case 95: goto tr177;
		case 108: goto st112;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st112:
	if ( ++p == pe )
		goto _test_eof112;
case 112:
	switch( (*p) ) {
		case 95: goto tr177;
		case 108: goto tr206;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st113:
	if ( ++p == pe )
		goto _test_eof113;
case 113:
	switch( (*p) ) {
		case 95: goto tr177;
		case 114: goto st114;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st114:
	if ( ++p == pe )
		goto _test_eof114;
case 114:
	switch( (*p) ) {
		case 95: goto tr177;
		case 115: goto tr208;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st115:
	if ( ++p == pe )
		goto _test_eof115;
case 115:
	switch( (*p) ) {
		case 95: goto tr177;
		case 110: goto st116;
		case 120: goto st119;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st116:
	if ( ++p == pe )
		goto _test_eof116;
case 116:
	switch( (*p) ) {
		case 95: goto tr177;
		case 116: goto st117;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st117:
	if ( ++p == pe )
		goto _test_eof117;
case 117:
	switch( (*p) ) {
		case 95: goto tr177;
		case 114: goto st118;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st118:
	if ( ++p == pe )
		goto _test_eof118;
case 118:
	switch( (*p) ) {
		case 95: goto tr177;
		case 121: goto tr213;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st119:
	if ( ++p == pe )
		goto _test_eof119;
case 119:
	switch( (*p) ) {
		case 95: goto tr177;
		case 101: goto st120;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st120:
	if ( ++p == pe )
		goto _test_eof120;
case 120:
	switch( (*p) ) {
		case 95: goto tr177;
		case 99: goto tr215;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st121:
	if ( ++p == pe )
		goto _test_eof121;
case 121:
	switch( (*p) ) {
		case 95: goto tr177;
		case 111: goto st122;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st122:
	if ( ++p == pe )
		goto _test_eof122;
case 122:
	switch( (*p) ) {
		case 95: goto tr177;
		case 116: goto st123;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st123:
	if ( ++p == pe )
		goto _test_eof123;
case 123:
	switch( (*p) ) {
		case 95: goto tr177;
		case 111: goto tr218;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st124:
	if ( ++p == pe )
		goto _test_eof124;
case 124:
	switch( (*p) ) {
		case 95: goto tr177;
		case 111: goto st125;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st125:
	if ( ++p == pe )
		goto _test_eof125;
case 125:
	switch( (*p) ) {
		case 95: goto tr177;
		case 108: goto st126;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st126:
	if ( ++p == pe )
		goto _test_eof126;
case 126:
	switch( (*p) ) {
		case 95: goto tr177;
		case 100: goto tr221;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st127:
	if ( ++p == pe )
		goto _test_eof127;
case 127:
	switch( (*p) ) {
		case 95: goto tr177;
		case 101: goto st128;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st128:
	if ( ++p == pe )
		goto _test_eof128;
case 128:
	switch( (*p) ) {
		case 95: goto tr177;
		case 120: goto st129;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st129:
	if ( ++p == pe )
		goto _test_eof129;
case 129:
	switch( (*p) ) {
		case 95: goto tr177;
		case 116: goto tr224;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st130:
	if ( ++p == pe )
		goto _test_eof130;
case 130:
	switch( (*p) ) {
		case 95: goto tr177;
		case 99: goto tr225;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st131:
	if ( ++p == pe )
		goto _test_eof131;
case 131:
	switch( (*p) ) {
		case 95: goto tr177;
		case 101: goto st132;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st132:
	if ( ++p == pe )
		goto _test_eof132;
case 132:
	switch( (*p) ) {
		case 95: goto tr177;
		case 116: goto tr227;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st133:
	if ( ++p == pe )
		goto _test_eof133;
case 133:
	switch( (*p) ) {
		case 95: goto tr177;
		case 97: goto st134;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 98 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st134:
	if ( ++p == pe )
		goto _test_eof134;
case 134:
	switch( (*p) ) {
		case 95: goto tr177;
		case 114: goto st135;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st135:
	if ( ++p == pe )
		goto _test_eof135;
case 135:
	switch( (*p) ) {
		case 95: goto tr177;
		case 103: goto st136;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
st136:
	if ( ++p == pe )
		goto _test_eof136;
case 136:
	switch( (*p) ) {
		case 95: goto tr177;
		case 115: goto tr231;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr177;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr177;
	} else
		goto tr177;
	goto tr188;
tr232:
#line 900 "rlscan.rl"
	{te = p+1;{ token( RE_Char, ts, te ); }}
	goto st137;
tr233:
#line 895 "rlscan.rl"
	{te = p+1;{
			scan_error() << "unterminated OR literal" << endl;
		}}
	goto st137;
tr234:
#line 890 "rlscan.rl"
	{te = p+1;{ token( RE_Dash, 0, 0 ); }}
	goto st137;
tr236:
#line 893 "rlscan.rl"
	{te = p+1;{ token( RE_SqClose ); {cs = stack[--top];goto _again;} }}
	goto st137;
tr237:
#line 900 "rlscan.rl"
	{te = p;p--;{ token( RE_Char, ts, te ); }}
	goto st137;
tr238:
#line 887 "rlscan.rl"
	{te = p+1;{ token( RE_Char, ts+1, te ); }}
	goto st137;
tr239:
#line 886 "rlscan.rl"
	{te = p+1;{ updateCol(); }}
	goto st137;
tr240:
#line 878 "rlscan.rl"
	{te = p+1;{ token( RE_Char, '\0' ); }}
	goto st137;
tr241:
#line 879 "rlscan.rl"
	{te = p+1;{ token( RE_Char, '\a' ); }}
	goto st137;
tr242:
#line 880 "rlscan.rl"
	{te = p+1;{ token( RE_Char, '\b' ); }}
	goto st137;
tr243:
#line 884 "rlscan.rl"
	{te = p+1;{ token( RE_Char, '\f' ); }}
	goto st137;
tr244:
#line 882 "rlscan.rl"
	{te = p+1;{ token( RE_Char, '\n' ); }}
	goto st137;
tr245:
#line 885 "rlscan.rl"
	{te = p+1;{ token( RE_Char, '\r' ); }}
	goto st137;
tr246:
#line 881 "rlscan.rl"
	{te = p+1;{ token( RE_Char, '\t' ); }}
	goto st137;
tr247:
#line 883 "rlscan.rl"
	{te = p+1;{ token( RE_Char, '\v' ); }}
	goto st137;
st137:
#line 1 "NONE"
	{ts = 0;}
	if ( ++p == pe )
		goto _test_eof137;
case 137:
#line 1 "NONE"
	{ts = p;}
#line 3856 "rlscan.cpp"
	switch( (*p) ) {
		case 0: goto tr233;
		case 45: goto tr234;
		case 92: goto st138;
		case 93: goto tr236;
	}
	goto tr232;
st138:
	if ( ++p == pe )
		goto _test_eof138;
case 138:
	switch( (*p) ) {
		case 10: goto tr239;
		case 48: goto tr240;
		case 97: goto tr241;
		case 98: goto tr242;
		case 102: goto tr243;
		case 110: goto tr244;
		case 114: goto tr245;
		case 116: goto tr246;
		case 118: goto tr247;
	}
	goto tr238;
tr248:
#line 935 "rlscan.rl"
	{te = p+1;{ token( RE_Char, ts, te ); }}
	goto st139;
tr249:
#line 930 "rlscan.rl"
	{te = p+1;{
			scan_error() << "unterminated regular expression" << endl;
		}}
	goto st139;
tr250:
#line 925 "rlscan.rl"
	{te = p+1;{ token( RE_Star ); }}
	goto st139;
tr251:
#line 924 "rlscan.rl"
	{te = p+1;{ token( RE_Dot ); }}
	goto st139;
tr255:
#line 918 "rlscan.rl"
	{te = p;p--;{ 
			token( RE_Slash, ts, te ); 
			{goto st146;}
		}}
	goto st139;
tr256:
#line 918 "rlscan.rl"
	{te = p+1;{ 
			token( RE_Slash, ts, te ); 
			{goto st146;}
		}}
	goto st139;
tr257:
#line 927 "rlscan.rl"
	{te = p;p--;{ token( RE_SqOpen ); {stack[top++] = 139; goto st137;} }}
	goto st139;
tr258:
#line 928 "rlscan.rl"
	{te = p+1;{ token( RE_SqOpenNeg ); {stack[top++] = 139; goto st137;} }}
	goto st139;
tr259:
#line 935 "rlscan.rl"
	{te = p;p--;{ token( RE_Char, ts, te ); }}
	goto st139;
tr260:
#line 915 "rlscan.rl"
	{te = p+1;{ token( RE_Char, ts+1, te ); }}
	goto st139;
tr261:
#line 914 "rlscan.rl"
	{te = p+1;{ updateCol(); }}
	goto st139;
tr262:
#line 906 "rlscan.rl"
	{te = p+1;{ token( RE_Char, '\0' ); }}
	goto st139;
tr263:
#line 907 "rlscan.rl"
	{te = p+1;{ token( RE_Char, '\a' ); }}
	goto st139;
tr264:
#line 908 "rlscan.rl"
	{te = p+1;{ token( RE_Char, '\b' ); }}
	goto st139;
tr265:
#line 912 "rlscan.rl"
	{te = p+1;{ token( RE_Char, '\f' ); }}
	goto st139;
tr266:
#line 910 "rlscan.rl"
	{te = p+1;{ token( RE_Char, '\n' ); }}
	goto st139;
tr267:
#line 913 "rlscan.rl"
	{te = p+1;{ token( RE_Char, '\r' ); }}
	goto st139;
tr268:
#line 909 "rlscan.rl"
	{te = p+1;{ token( RE_Char, '\t' ); }}
	goto st139;
tr269:
#line 911 "rlscan.rl"
	{te = p+1;{ token( RE_Char, '\v' ); }}
	goto st139;
st139:
#line 1 "NONE"
	{ts = 0;}
	if ( ++p == pe )
		goto _test_eof139;
case 139:
#line 1 "NONE"
	{ts = p;}
#line 3972 "rlscan.cpp"
	switch( (*p) ) {
		case 0: goto tr249;
		case 42: goto tr250;
		case 46: goto tr251;
		case 47: goto st140;
		case 91: goto st141;
		case 92: goto st142;
	}
	goto tr248;
st140:
	if ( ++p == pe )
		goto _test_eof140;
case 140:
	if ( (*p) == 105 )
		goto tr256;
	goto tr255;
st141:
	if ( ++p == pe )
		goto _test_eof141;
case 141:
	if ( (*p) == 94 )
		goto tr258;
	goto tr257;
st142:
	if ( ++p == pe )
		goto _test_eof142;
case 142:
	switch( (*p) ) {
		case 10: goto tr261;
		case 48: goto tr262;
		case 97: goto tr263;
		case 98: goto tr264;
		case 102: goto tr265;
		case 110: goto tr266;
		case 114: goto tr267;
		case 116: goto tr268;
		case 118: goto tr269;
	}
	goto tr260;
tr270:
#line 944 "rlscan.rl"
	{te = p+1;{
			scan_error() << "unterminated write statement" << endl;
		}}
	goto st143;
tr273:
#line 942 "rlscan.rl"
	{te = p+1;{ token( ';' ); {goto st146;} }}
	goto st143;
tr275:
#line 941 "rlscan.rl"
	{te = p;p--;{ updateCol(); }}
	goto st143;
tr276:
#line 940 "rlscan.rl"
	{te = p;p--;{ token( TK_Word, ts, te ); }}
	goto st143;
st143:
#line 1 "NONE"
	{ts = 0;}
	if ( ++p == pe )
		goto _test_eof143;
case 143:
#line 1 "NONE"
	{ts = p;}
#line 4038 "rlscan.cpp"
	switch( (*p) ) {
		case 0: goto tr270;
		case 32: goto st144;
		case 59: goto tr273;
		case 95: goto st145;
	}
	if ( (*p) < 65 ) {
		if ( 9 <= (*p) && (*p) <= 10 )
			goto st144;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto st145;
	} else
		goto st145;
	goto st0;
st0:
cs = 0;
	goto _out;
st144:
	if ( ++p == pe )
		goto _test_eof144;
case 144:
	if ( (*p) == 32 )
		goto st144;
	if ( 9 <= (*p) && (*p) <= 10 )
		goto st144;
	goto tr275;
st145:
	if ( ++p == pe )
		goto _test_eof145;
case 145:
	if ( (*p) == 95 )
		goto st145;
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st145;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto st145;
	} else
		goto st145;
	goto tr276;
tr45:
#line 1121 "rlscan.rl"
	{{p = ((te))-1;}{ token( *ts ); }}
	goto st146;
tr51:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
#line 1018 "rlscan.rl"
	{te = p+1;{ updateCol(); }}
	goto st146;
tr55:
#line 1005 "rlscan.rl"
	{{p = ((te))-1;}{ token( TK_UInt, ts, te ); }}
	goto st146;
tr57:
#line 1086 "rlscan.rl"
	{te = p+1;{ 
			updateCol();
			endSection();
			{cs = stack[--top];goto _again;}
		}}
	goto st146;
tr277:
#line 1121 "rlscan.rl"
	{te = p+1;{ token( *ts ); }}
	goto st146;
tr278:
#line 1117 "rlscan.rl"
	{te = p+1;{
			scan_error() << "unterminated ragel section" << endl;
		}}
	goto st146;
tr280:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
#line 1095 "rlscan.rl"
	{te = p+1;{
			updateCol();
			if ( singleLineSpec ) {
				endSection();
				{cs = stack[--top];goto _again;}
			}
		}}
	goto st146;
tr289:
#line 1015 "rlscan.rl"
	{te = p+1;{ token( RE_Slash ); {goto st139;} }}
	goto st146;
tr311:
#line 1103 "rlscan.rl"
	{te = p+1;{ 
			if ( lastToken == KW_Export || lastToken == KW_Entry )
				token( '{' );
			else {
				token( '{' );
				curly_count = 1; 
				inlineBlockType = CurlyDelimited;
				if ( hostLang->lang == HostLang::Ruby )
					{stack[top++] = 146; goto st52;}
				else
					{stack[top++] = 146; goto st95;}
			}
		}}
	goto st146;
tr314:
#line 1092 "rlscan.rl"
	{te = p;p--;{ updateCol(); }}
	goto st146;
tr315:
#line 1121 "rlscan.rl"
	{te = p;p--;{ token( *ts ); }}
	goto st146;
tr316:
#line 1010 "rlscan.rl"
	{te = p;p--;{ token( TK_Literal, ts, te ); }}
	goto st146;
tr317:
#line 1010 "rlscan.rl"
	{te = p+1;{ token( TK_Literal, ts, te ); }}
	goto st146;
tr318:
#line 1048 "rlscan.rl"
	{te = p+1;{ token( TK_AllGblError ); }}
	goto st146;
tr319:
#line 1032 "rlscan.rl"
	{te = p+1;{ token( TK_AllFromState ); }}
	goto st146;
tr320:
#line 1040 "rlscan.rl"
	{te = p+1;{ token( TK_AllEOF ); }}
	goto st146;
tr321:
#line 1067 "rlscan.rl"
	{te = p+1;{ token( TK_AllCond ); }}
	goto st146;
tr322:
#line 1056 "rlscan.rl"
	{te = p+1;{ token( TK_AllLocalError ); }}
	goto st146;
tr323:
#line 1024 "rlscan.rl"
	{te = p+1;{ token( TK_AllToState ); }}
	goto st146;
tr324:
#line 1049 "rlscan.rl"
	{te = p+1;{ token( TK_FinalGblError ); }}
	goto st146;
tr325:
#line 1033 "rlscan.rl"
	{te = p+1;{ token( TK_FinalFromState ); }}
	goto st146;
tr326:
#line 1041 "rlscan.rl"
	{te = p+1;{ token( TK_FinalEOF ); }}
	goto st146;
tr327:
#line 1068 "rlscan.rl"
	{te = p+1;{ token( TK_LeavingCond ); }}
	goto st146;
tr328:
#line 1057 "rlscan.rl"
	{te = p+1;{ token( TK_FinalLocalError ); }}
	goto st146;
tr329:
#line 1025 "rlscan.rl"
	{te = p+1;{ token( TK_FinalToState ); }}
	goto st146;
tr330:
#line 1071 "rlscan.rl"
	{te = p+1;{ token( TK_StarStar ); }}
	goto st146;
tr331:
#line 1072 "rlscan.rl"
	{te = p+1;{ token( TK_DashDash ); }}
	goto st146;
tr332:
#line 1073 "rlscan.rl"
	{te = p+1;{ token( TK_Arrow ); }}
	goto st146;
tr333:
#line 1070 "rlscan.rl"
	{te = p+1;{ token( TK_DotDot ); }}
	goto st146;
tr334:
#line 1005 "rlscan.rl"
	{te = p;p--;{ token( TK_UInt, ts, te ); }}
	goto st146;
tr336:
#line 1006 "rlscan.rl"
	{te = p;p--;{ token( TK_Hex, ts, te ); }}
	goto st146;
tr337:
#line 1084 "rlscan.rl"
	{te = p+1;{ token( TK_NameSep, ts, te ); }}
	goto st146;
tr338:
#line 1020 "rlscan.rl"
	{te = p+1;{ token( TK_ColonEquals ); }}
	goto st146;
tr340:
#line 1076 "rlscan.rl"
	{te = p;p--;{ token( TK_ColonGt ); }}
	goto st146;
tr341:
#line 1077 "rlscan.rl"
	{te = p+1;{ token( TK_ColonGtGt ); }}
	goto st146;
tr342:
#line 1050 "rlscan.rl"
	{te = p+1;{ token( TK_NotStartGblError ); }}
	goto st146;
tr343:
#line 1034 "rlscan.rl"
	{te = p+1;{ token( TK_NotStartFromState ); }}
	goto st146;
tr344:
#line 1042 "rlscan.rl"
	{te = p+1;{ token( TK_NotStartEOF ); }}
	goto st146;
tr345:
#line 1078 "rlscan.rl"
	{te = p+1;{ token( TK_LtColon ); }}
	goto st146;
tr347:
#line 1058 "rlscan.rl"
	{te = p+1;{ token( TK_NotStartLocalError ); }}
	goto st146;
tr348:
#line 1026 "rlscan.rl"
	{te = p+1;{ token( TK_NotStartToState ); }}
	goto st146;
tr349:
#line 1063 "rlscan.rl"
	{te = p;p--;{ token( TK_Middle ); }}
	goto st146;
tr350:
#line 1052 "rlscan.rl"
	{te = p+1;{ token( TK_MiddleGblError ); }}
	goto st146;
tr351:
#line 1036 "rlscan.rl"
	{te = p+1;{ token( TK_MiddleFromState ); }}
	goto st146;
tr352:
#line 1044 "rlscan.rl"
	{te = p+1;{ token( TK_MiddleEOF ); }}
	goto st146;
tr353:
#line 1060 "rlscan.rl"
	{te = p+1;{ token( TK_MiddleLocalError ); }}
	goto st146;
tr354:
#line 1028 "rlscan.rl"
	{te = p+1;{ token( TK_MiddleToState ); }}
	goto st146;
tr355:
#line 1074 "rlscan.rl"
	{te = p+1;{ token( TK_DoubleArrow ); }}
	goto st146;
tr356:
#line 1047 "rlscan.rl"
	{te = p+1;{ token( TK_StartGblError ); }}
	goto st146;
tr357:
#line 1031 "rlscan.rl"
	{te = p+1;{ token( TK_StartFromState ); }}
	goto st146;
tr358:
#line 1039 "rlscan.rl"
	{te = p+1;{ token( TK_StartEOF ); }}
	goto st146;
tr359:
#line 1066 "rlscan.rl"
	{te = p+1;{ token( TK_StartCond ); }}
	goto st146;
tr360:
#line 1055 "rlscan.rl"
	{te = p+1;{ token( TK_StartLocalError ); }}
	goto st146;
tr361:
#line 1023 "rlscan.rl"
	{te = p+1;{ token( TK_StartToState ); }}
	goto st146;
tr362:
#line 1051 "rlscan.rl"
	{te = p+1;{ token( TK_NotFinalGblError ); }}
	goto st146;
tr363:
#line 1035 "rlscan.rl"
	{te = p+1;{ token( TK_NotFinalFromState ); }}
	goto st146;
tr364:
#line 1043 "rlscan.rl"
	{te = p+1;{ token( TK_NotFinalEOF ); }}
	goto st146;
tr365:
#line 1059 "rlscan.rl"
	{te = p+1;{ token( TK_NotFinalLocalError ); }}
	goto st146;
tr366:
#line 1027 "rlscan.rl"
	{te = p+1;{ token( TK_NotFinalToState ); }}
	goto st146;
tr367:
#line 1 "NONE"
	{	switch( act ) {
	case 88:
	{{p = ((te))-1;} token( KW_Machine ); }
	break;
	case 89:
	{{p = ((te))-1;} token( KW_Include ); }
	break;
	case 90:
	{{p = ((te))-1;} token( KW_Import ); }
	break;
	case 91:
	{{p = ((te))-1;} 
			token( KW_Write );
			{goto st143;}
		}
	break;
	case 92:
	{{p = ((te))-1;} token( KW_Action ); }
	break;
	case 93:
	{{p = ((te))-1;} token( KW_AlphType ); }
	break;
	case 94:
	{{p = ((te))-1;} token( KW_PrePush ); }
	break;
	case 95:
	{{p = ((te))-1;} token( KW_PostPop ); }
	break;
	case 96:
	{{p = ((te))-1;} 
			token( KW_GetKey );
			inlineBlockType = SemiTerminated;
			if ( hostLang->lang == HostLang::Ruby )
				{stack[top++] = 146; goto st52;}
			else
				{stack[top++] = 146; goto st95;}
		}
	break;
	case 97:
	{{p = ((te))-1;} 
			token( KW_Access );
			inlineBlockType = SemiTerminated;
			if ( hostLang->lang == HostLang::Ruby )
				{stack[top++] = 146; goto st52;}
			else
				{stack[top++] = 146; goto st95;}
		}
	break;
	case 98:
	{{p = ((te))-1;} 
			token( KW_Variable );
			inlineBlockType = SemiTerminated;
			if ( hostLang->lang == HostLang::Ruby )
				{stack[top++] = 146; goto st52;}
			else
				{stack[top++] = 146; goto st95;}
		}
	break;
	case 99:
	{{p = ((te))-1;} token( KW_When ); }
	break;
	case 100:
	{{p = ((te))-1;} token( KW_InWhen ); }
	break;
	case 101:
	{{p = ((te))-1;} token( KW_OutWhen ); }
	break;
	case 102:
	{{p = ((te))-1;} token( KW_Eof ); }
	break;
	case 103:
	{{p = ((te))-1;} token( KW_Err ); }
	break;
	case 104:
	{{p = ((te))-1;} token( KW_Lerr ); }
	break;
	case 105:
	{{p = ((te))-1;} token( KW_To ); }
	break;
	case 106:
	{{p = ((te))-1;} token( KW_From ); }
	break;
	case 107:
	{{p = ((te))-1;} token( KW_Export ); }
	break;
	case 108:
	{{p = ((te))-1;} token( TK_Word, ts, te ); }
	break;
	}
	}
	goto st146;
tr368:
#line 1012 "rlscan.rl"
	{te = p;p--;{ token( RE_SqOpen ); {stack[top++] = 146; goto st137;} }}
	goto st146;
tr369:
#line 1013 "rlscan.rl"
	{te = p+1;{ token( RE_SqOpenNeg ); {stack[top++] = 146; goto st137;} }}
	goto st146;
tr370:
#line 1002 "rlscan.rl"
	{te = p;p--;{ token( TK_Word, ts, te ); }}
	goto st146;
tr461:
#line 1081 "rlscan.rl"
	{te = p+1;{ token( TK_BarStar ); }}
	goto st146;
st146:
#line 1 "NONE"
	{ts = 0;}
	if ( ++p == pe )
		goto _test_eof146;
case 146:
#line 1 "NONE"
	{ts = p;}
#line 4470 "rlscan.cpp"
	switch( (*p) ) {
		case 0: goto tr278;
		case 9: goto st147;
		case 10: goto tr280;
		case 13: goto st147;
		case 32: goto st147;
		case 34: goto tr281;
		case 35: goto tr282;
		case 36: goto st151;
		case 37: goto st152;
		case 39: goto tr285;
		case 42: goto st154;
		case 45: goto st155;
		case 46: goto st156;
		case 47: goto tr289;
		case 48: goto tr290;
		case 58: goto st160;
		case 60: goto st162;
		case 61: goto st164;
		case 62: goto st165;
		case 64: goto st166;
		case 91: goto st168;
		case 95: goto tr297;
		case 97: goto st169;
		case 101: goto st183;
		case 102: goto st190;
		case 103: goto st193;
		case 105: goto st198;
		case 108: goto st211;
		case 109: goto st214;
		case 111: goto st220;
		case 112: goto st226;
		case 116: goto st237;
		case 118: goto st238;
		case 119: goto st245;
		case 123: goto tr311;
		case 124: goto st251;
		case 125: goto tr313;
	}
	if ( (*p) < 65 ) {
		if ( 49 <= (*p) && (*p) <= 57 )
			goto st158;
	} else if ( (*p) > 90 ) {
		if ( 98 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr277;
st147:
	if ( ++p == pe )
		goto _test_eof147;
case 147:
	switch( (*p) ) {
		case 9: goto st147;
		case 13: goto st147;
		case 32: goto st147;
	}
	goto tr314;
tr281:
#line 1 "NONE"
	{te = p+1;}
	goto st148;
st148:
	if ( ++p == pe )
		goto _test_eof148;
case 148:
#line 4537 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr47;
		case 34: goto st149;
		case 92: goto st25;
	}
	goto st24;
tr47:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
	goto st24;
st24:
	if ( ++p == pe )
		goto _test_eof24;
case 24:
#line 4556 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr47;
		case 34: goto st149;
		case 92: goto st25;
	}
	goto st24;
st149:
	if ( ++p == pe )
		goto _test_eof149;
case 149:
	if ( (*p) == 105 )
		goto tr317;
	goto tr316;
st25:
	if ( ++p == pe )
		goto _test_eof25;
case 25:
	if ( (*p) == 10 )
		goto tr47;
	goto st24;
tr282:
#line 1 "NONE"
	{te = p+1;}
	goto st150;
st150:
	if ( ++p == pe )
		goto _test_eof150;
case 150:
#line 4585 "rlscan.cpp"
	if ( (*p) == 10 )
		goto tr51;
	goto st26;
st26:
	if ( ++p == pe )
		goto _test_eof26;
case 26:
	if ( (*p) == 10 )
		goto tr51;
	goto st26;
st151:
	if ( ++p == pe )
		goto _test_eof151;
case 151:
	switch( (*p) ) {
		case 33: goto tr318;
		case 42: goto tr319;
		case 47: goto tr320;
		case 63: goto tr321;
		case 94: goto tr322;
		case 126: goto tr323;
	}
	goto tr315;
st152:
	if ( ++p == pe )
		goto _test_eof152;
case 152:
	switch( (*p) ) {
		case 33: goto tr324;
		case 42: goto tr325;
		case 47: goto tr326;
		case 63: goto tr327;
		case 94: goto tr328;
		case 126: goto tr329;
	}
	goto tr315;
tr285:
#line 1 "NONE"
	{te = p+1;}
	goto st153;
st153:
	if ( ++p == pe )
		goto _test_eof153;
case 153:
#line 4630 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr53;
		case 39: goto st149;
		case 92: goto st28;
	}
	goto st27;
tr53:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
	goto st27;
st27:
	if ( ++p == pe )
		goto _test_eof27;
case 27:
#line 4649 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr53;
		case 39: goto st149;
		case 92: goto st28;
	}
	goto st27;
st28:
	if ( ++p == pe )
		goto _test_eof28;
case 28:
	if ( (*p) == 10 )
		goto tr53;
	goto st27;
st154:
	if ( ++p == pe )
		goto _test_eof154;
case 154:
	if ( (*p) == 42 )
		goto tr330;
	goto tr315;
st155:
	if ( ++p == pe )
		goto _test_eof155;
case 155:
	switch( (*p) ) {
		case 45: goto tr331;
		case 62: goto tr332;
	}
	goto tr315;
st156:
	if ( ++p == pe )
		goto _test_eof156;
case 156:
	if ( (*p) == 46 )
		goto tr333;
	goto tr315;
tr290:
#line 1 "NONE"
	{te = p+1;}
	goto st157;
st157:
	if ( ++p == pe )
		goto _test_eof157;
case 157:
#line 4694 "rlscan.cpp"
	if ( (*p) == 120 )
		goto st29;
	if ( 48 <= (*p) && (*p) <= 57 )
		goto st158;
	goto tr334;
st158:
	if ( ++p == pe )
		goto _test_eof158;
case 158:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto st158;
	goto tr334;
st29:
	if ( ++p == pe )
		goto _test_eof29;
case 29:
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st159;
	} else if ( (*p) > 70 ) {
		if ( 97 <= (*p) && (*p) <= 102 )
			goto st159;
	} else
		goto st159;
	goto tr55;
st159:
	if ( ++p == pe )
		goto _test_eof159;
case 159:
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st159;
	} else if ( (*p) > 70 ) {
		if ( 97 <= (*p) && (*p) <= 102 )
			goto st159;
	} else
		goto st159;
	goto tr336;
st160:
	if ( ++p == pe )
		goto _test_eof160;
case 160:
	switch( (*p) ) {
		case 58: goto tr337;
		case 61: goto tr338;
		case 62: goto st161;
	}
	goto tr315;
st161:
	if ( ++p == pe )
		goto _test_eof161;
case 161:
	if ( (*p) == 62 )
		goto tr341;
	goto tr340;
st162:
	if ( ++p == pe )
		goto _test_eof162;
case 162:
	switch( (*p) ) {
		case 33: goto tr342;
		case 42: goto tr343;
		case 47: goto tr344;
		case 58: goto tr345;
		case 62: goto st163;
		case 94: goto tr347;
		case 126: goto tr348;
	}
	goto tr315;
st163:
	if ( ++p == pe )
		goto _test_eof163;
case 163:
	switch( (*p) ) {
		case 33: goto tr350;
		case 42: goto tr351;
		case 47: goto tr352;
		case 94: goto tr353;
		case 126: goto tr354;
	}
	goto tr349;
st164:
	if ( ++p == pe )
		goto _test_eof164;
case 164:
	if ( (*p) == 62 )
		goto tr355;
	goto tr315;
st165:
	if ( ++p == pe )
		goto _test_eof165;
case 165:
	switch( (*p) ) {
		case 33: goto tr356;
		case 42: goto tr357;
		case 47: goto tr358;
		case 63: goto tr359;
		case 94: goto tr360;
		case 126: goto tr361;
	}
	goto tr315;
st166:
	if ( ++p == pe )
		goto _test_eof166;
case 166:
	switch( (*p) ) {
		case 33: goto tr362;
		case 42: goto tr363;
		case 47: goto tr364;
		case 94: goto tr365;
		case 126: goto tr366;
	}
	goto tr315;
tr297:
#line 1 "NONE"
	{te = p+1;}
#line 1002 "rlscan.rl"
	{act = 108;}
	goto st167;
tr377:
#line 1 "NONE"
	{te = p+1;}
#line 975 "rlscan.rl"
	{act = 97;}
	goto st167;
tr380:
#line 1 "NONE"
	{te = p+1;}
#line 959 "rlscan.rl"
	{act = 92;}
	goto st167;
tr386:
#line 1 "NONE"
	{te = p+1;}
#line 960 "rlscan.rl"
	{act = 93;}
	goto st167;
tr390:
#line 1 "NONE"
	{te = p+1;}
#line 994 "rlscan.rl"
	{act = 102;}
	goto st167;
tr391:
#line 1 "NONE"
	{te = p+1;}
#line 995 "rlscan.rl"
	{act = 103;}
	goto st167;
tr395:
#line 1 "NONE"
	{te = p+1;}
#line 999 "rlscan.rl"
	{act = 107;}
	goto st167;
tr398:
#line 1 "NONE"
	{te = p+1;}
#line 998 "rlscan.rl"
	{act = 106;}
	goto st167;
tr403:
#line 1 "NONE"
	{te = p+1;}
#line 967 "rlscan.rl"
	{act = 96;}
	goto st167;
tr409:
#line 1 "NONE"
	{te = p+1;}
#line 954 "rlscan.rl"
	{act = 90;}
	goto st167;
tr415:
#line 1 "NONE"
	{te = p+1;}
#line 953 "rlscan.rl"
	{act = 89;}
	goto st167;
tr418:
#line 1 "NONE"
	{te = p+1;}
#line 992 "rlscan.rl"
	{act = 100;}
	goto st167;
tr421:
#line 1 "NONE"
	{te = p+1;}
#line 996 "rlscan.rl"
	{act = 104;}
	goto st167;
tr427:
#line 1 "NONE"
	{te = p+1;}
#line 952 "rlscan.rl"
	{act = 88;}
	goto st167;
tr433:
#line 1 "NONE"
	{te = p+1;}
#line 993 "rlscan.rl"
	{act = 101;}
	goto st167;
tr440:
#line 1 "NONE"
	{te = p+1;}
#line 962 "rlscan.rl"
	{act = 95;}
	goto st167;
tr445:
#line 1 "NONE"
	{te = p+1;}
#line 961 "rlscan.rl"
	{act = 94;}
	goto st167;
tr446:
#line 1 "NONE"
	{te = p+1;}
#line 997 "rlscan.rl"
	{act = 105;}
	goto st167;
tr453:
#line 1 "NONE"
	{te = p+1;}
#line 983 "rlscan.rl"
	{act = 98;}
	goto st167;
tr457:
#line 1 "NONE"
	{te = p+1;}
#line 991 "rlscan.rl"
	{act = 99;}
	goto st167;
tr460:
#line 1 "NONE"
	{te = p+1;}
#line 955 "rlscan.rl"
	{act = 91;}
	goto st167;
st167:
	if ( ++p == pe )
		goto _test_eof167;
case 167:
#line 4938 "rlscan.cpp"
	if ( (*p) == 95 )
		goto tr297;
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr367;
st168:
	if ( ++p == pe )
		goto _test_eof168;
case 168:
	if ( (*p) == 94 )
		goto tr369;
	goto tr368;
st169:
	if ( ++p == pe )
		goto _test_eof169;
case 169:
	switch( (*p) ) {
		case 95: goto tr297;
		case 99: goto st170;
		case 108: goto st177;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st170:
	if ( ++p == pe )
		goto _test_eof170;
case 170:
	switch( (*p) ) {
		case 95: goto tr297;
		case 99: goto st171;
		case 116: goto st174;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st171:
	if ( ++p == pe )
		goto _test_eof171;
case 171:
	switch( (*p) ) {
		case 95: goto tr297;
		case 101: goto st172;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st172:
	if ( ++p == pe )
		goto _test_eof172;
case 172:
	switch( (*p) ) {
		case 95: goto tr297;
		case 115: goto st173;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st173:
	if ( ++p == pe )
		goto _test_eof173;
case 173:
	switch( (*p) ) {
		case 95: goto tr297;
		case 115: goto tr377;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st174:
	if ( ++p == pe )
		goto _test_eof174;
case 174:
	switch( (*p) ) {
		case 95: goto tr297;
		case 105: goto st175;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st175:
	if ( ++p == pe )
		goto _test_eof175;
case 175:
	switch( (*p) ) {
		case 95: goto tr297;
		case 111: goto st176;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st176:
	if ( ++p == pe )
		goto _test_eof176;
case 176:
	switch( (*p) ) {
		case 95: goto tr297;
		case 110: goto tr380;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st177:
	if ( ++p == pe )
		goto _test_eof177;
case 177:
	switch( (*p) ) {
		case 95: goto tr297;
		case 112: goto st178;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st178:
	if ( ++p == pe )
		goto _test_eof178;
case 178:
	switch( (*p) ) {
		case 95: goto tr297;
		case 104: goto st179;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st179:
	if ( ++p == pe )
		goto _test_eof179;
case 179:
	switch( (*p) ) {
		case 95: goto tr297;
		case 116: goto st180;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st180:
	if ( ++p == pe )
		goto _test_eof180;
case 180:
	switch( (*p) ) {
		case 95: goto tr297;
		case 121: goto st181;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st181:
	if ( ++p == pe )
		goto _test_eof181;
case 181:
	switch( (*p) ) {
		case 95: goto tr297;
		case 112: goto st182;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st182:
	if ( ++p == pe )
		goto _test_eof182;
case 182:
	switch( (*p) ) {
		case 95: goto tr297;
		case 101: goto tr386;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st183:
	if ( ++p == pe )
		goto _test_eof183;
case 183:
	switch( (*p) ) {
		case 95: goto tr297;
		case 111: goto st184;
		case 114: goto st185;
		case 120: goto st186;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st184:
	if ( ++p == pe )
		goto _test_eof184;
case 184:
	switch( (*p) ) {
		case 95: goto tr297;
		case 102: goto tr390;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st185:
	if ( ++p == pe )
		goto _test_eof185;
case 185:
	switch( (*p) ) {
		case 95: goto tr297;
		case 114: goto tr391;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st186:
	if ( ++p == pe )
		goto _test_eof186;
case 186:
	switch( (*p) ) {
		case 95: goto tr297;
		case 112: goto st187;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st187:
	if ( ++p == pe )
		goto _test_eof187;
case 187:
	switch( (*p) ) {
		case 95: goto tr297;
		case 111: goto st188;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st188:
	if ( ++p == pe )
		goto _test_eof188;
case 188:
	switch( (*p) ) {
		case 95: goto tr297;
		case 114: goto st189;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st189:
	if ( ++p == pe )
		goto _test_eof189;
case 189:
	switch( (*p) ) {
		case 95: goto tr297;
		case 116: goto tr395;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st190:
	if ( ++p == pe )
		goto _test_eof190;
case 190:
	switch( (*p) ) {
		case 95: goto tr297;
		case 114: goto st191;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st191:
	if ( ++p == pe )
		goto _test_eof191;
case 191:
	switch( (*p) ) {
		case 95: goto tr297;
		case 111: goto st192;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st192:
	if ( ++p == pe )
		goto _test_eof192;
case 192:
	switch( (*p) ) {
		case 95: goto tr297;
		case 109: goto tr398;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st193:
	if ( ++p == pe )
		goto _test_eof193;
case 193:
	switch( (*p) ) {
		case 95: goto tr297;
		case 101: goto st194;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st194:
	if ( ++p == pe )
		goto _test_eof194;
case 194:
	switch( (*p) ) {
		case 95: goto tr297;
		case 116: goto st195;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st195:
	if ( ++p == pe )
		goto _test_eof195;
case 195:
	switch( (*p) ) {
		case 95: goto tr297;
		case 107: goto st196;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st196:
	if ( ++p == pe )
		goto _test_eof196;
case 196:
	switch( (*p) ) {
		case 95: goto tr297;
		case 101: goto st197;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st197:
	if ( ++p == pe )
		goto _test_eof197;
case 197:
	switch( (*p) ) {
		case 95: goto tr297;
		case 121: goto tr403;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st198:
	if ( ++p == pe )
		goto _test_eof198;
case 198:
	switch( (*p) ) {
		case 95: goto tr297;
		case 109: goto st199;
		case 110: goto st203;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st199:
	if ( ++p == pe )
		goto _test_eof199;
case 199:
	switch( (*p) ) {
		case 95: goto tr297;
		case 112: goto st200;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st200:
	if ( ++p == pe )
		goto _test_eof200;
case 200:
	switch( (*p) ) {
		case 95: goto tr297;
		case 111: goto st201;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st201:
	if ( ++p == pe )
		goto _test_eof201;
case 201:
	switch( (*p) ) {
		case 95: goto tr297;
		case 114: goto st202;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st202:
	if ( ++p == pe )
		goto _test_eof202;
case 202:
	switch( (*p) ) {
		case 95: goto tr297;
		case 116: goto tr409;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st203:
	if ( ++p == pe )
		goto _test_eof203;
case 203:
	switch( (*p) ) {
		case 95: goto tr297;
		case 99: goto st204;
		case 119: goto st208;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st204:
	if ( ++p == pe )
		goto _test_eof204;
case 204:
	switch( (*p) ) {
		case 95: goto tr297;
		case 108: goto st205;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st205:
	if ( ++p == pe )
		goto _test_eof205;
case 205:
	switch( (*p) ) {
		case 95: goto tr297;
		case 117: goto st206;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st206:
	if ( ++p == pe )
		goto _test_eof206;
case 206:
	switch( (*p) ) {
		case 95: goto tr297;
		case 100: goto st207;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st207:
	if ( ++p == pe )
		goto _test_eof207;
case 207:
	switch( (*p) ) {
		case 95: goto tr297;
		case 101: goto tr415;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st208:
	if ( ++p == pe )
		goto _test_eof208;
case 208:
	switch( (*p) ) {
		case 95: goto tr297;
		case 104: goto st209;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st209:
	if ( ++p == pe )
		goto _test_eof209;
case 209:
	switch( (*p) ) {
		case 95: goto tr297;
		case 101: goto st210;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st210:
	if ( ++p == pe )
		goto _test_eof210;
case 210:
	switch( (*p) ) {
		case 95: goto tr297;
		case 110: goto tr418;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st211:
	if ( ++p == pe )
		goto _test_eof211;
case 211:
	switch( (*p) ) {
		case 95: goto tr297;
		case 101: goto st212;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st212:
	if ( ++p == pe )
		goto _test_eof212;
case 212:
	switch( (*p) ) {
		case 95: goto tr297;
		case 114: goto st213;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st213:
	if ( ++p == pe )
		goto _test_eof213;
case 213:
	switch( (*p) ) {
		case 95: goto tr297;
		case 114: goto tr421;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st214:
	if ( ++p == pe )
		goto _test_eof214;
case 214:
	switch( (*p) ) {
		case 95: goto tr297;
		case 97: goto st215;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 98 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st215:
	if ( ++p == pe )
		goto _test_eof215;
case 215:
	switch( (*p) ) {
		case 95: goto tr297;
		case 99: goto st216;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st216:
	if ( ++p == pe )
		goto _test_eof216;
case 216:
	switch( (*p) ) {
		case 95: goto tr297;
		case 104: goto st217;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st217:
	if ( ++p == pe )
		goto _test_eof217;
case 217:
	switch( (*p) ) {
		case 95: goto tr297;
		case 105: goto st218;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st218:
	if ( ++p == pe )
		goto _test_eof218;
case 218:
	switch( (*p) ) {
		case 95: goto tr297;
		case 110: goto st219;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st219:
	if ( ++p == pe )
		goto _test_eof219;
case 219:
	switch( (*p) ) {
		case 95: goto tr297;
		case 101: goto tr427;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st220:
	if ( ++p == pe )
		goto _test_eof220;
case 220:
	switch( (*p) ) {
		case 95: goto tr297;
		case 117: goto st221;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st221:
	if ( ++p == pe )
		goto _test_eof221;
case 221:
	switch( (*p) ) {
		case 95: goto tr297;
		case 116: goto st222;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st222:
	if ( ++p == pe )
		goto _test_eof222;
case 222:
	switch( (*p) ) {
		case 95: goto tr297;
		case 119: goto st223;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st223:
	if ( ++p == pe )
		goto _test_eof223;
case 223:
	switch( (*p) ) {
		case 95: goto tr297;
		case 104: goto st224;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st224:
	if ( ++p == pe )
		goto _test_eof224;
case 224:
	switch( (*p) ) {
		case 95: goto tr297;
		case 101: goto st225;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st225:
	if ( ++p == pe )
		goto _test_eof225;
case 225:
	switch( (*p) ) {
		case 95: goto tr297;
		case 110: goto tr433;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st226:
	if ( ++p == pe )
		goto _test_eof226;
case 226:
	switch( (*p) ) {
		case 95: goto tr297;
		case 111: goto st227;
		case 114: goto st232;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st227:
	if ( ++p == pe )
		goto _test_eof227;
case 227:
	switch( (*p) ) {
		case 95: goto tr297;
		case 115: goto st228;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st228:
	if ( ++p == pe )
		goto _test_eof228;
case 228:
	switch( (*p) ) {
		case 95: goto tr297;
		case 116: goto st229;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st229:
	if ( ++p == pe )
		goto _test_eof229;
case 229:
	switch( (*p) ) {
		case 95: goto tr297;
		case 112: goto st230;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st230:
	if ( ++p == pe )
		goto _test_eof230;
case 230:
	switch( (*p) ) {
		case 95: goto tr297;
		case 111: goto st231;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st231:
	if ( ++p == pe )
		goto _test_eof231;
case 231:
	switch( (*p) ) {
		case 95: goto tr297;
		case 112: goto tr440;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st232:
	if ( ++p == pe )
		goto _test_eof232;
case 232:
	switch( (*p) ) {
		case 95: goto tr297;
		case 101: goto st233;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st233:
	if ( ++p == pe )
		goto _test_eof233;
case 233:
	switch( (*p) ) {
		case 95: goto tr297;
		case 112: goto st234;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st234:
	if ( ++p == pe )
		goto _test_eof234;
case 234:
	switch( (*p) ) {
		case 95: goto tr297;
		case 117: goto st235;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st235:
	if ( ++p == pe )
		goto _test_eof235;
case 235:
	switch( (*p) ) {
		case 95: goto tr297;
		case 115: goto st236;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st236:
	if ( ++p == pe )
		goto _test_eof236;
case 236:
	switch( (*p) ) {
		case 95: goto tr297;
		case 104: goto tr445;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st237:
	if ( ++p == pe )
		goto _test_eof237;
case 237:
	switch( (*p) ) {
		case 95: goto tr297;
		case 111: goto tr446;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st238:
	if ( ++p == pe )
		goto _test_eof238;
case 238:
	switch( (*p) ) {
		case 95: goto tr297;
		case 97: goto st239;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 98 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st239:
	if ( ++p == pe )
		goto _test_eof239;
case 239:
	switch( (*p) ) {
		case 95: goto tr297;
		case 114: goto st240;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st240:
	if ( ++p == pe )
		goto _test_eof240;
case 240:
	switch( (*p) ) {
		case 95: goto tr297;
		case 105: goto st241;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st241:
	if ( ++p == pe )
		goto _test_eof241;
case 241:
	switch( (*p) ) {
		case 95: goto tr297;
		case 97: goto st242;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 98 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st242:
	if ( ++p == pe )
		goto _test_eof242;
case 242:
	switch( (*p) ) {
		case 95: goto tr297;
		case 98: goto st243;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st243:
	if ( ++p == pe )
		goto _test_eof243;
case 243:
	switch( (*p) ) {
		case 95: goto tr297;
		case 108: goto st244;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st244:
	if ( ++p == pe )
		goto _test_eof244;
case 244:
	switch( (*p) ) {
		case 95: goto tr297;
		case 101: goto tr453;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st245:
	if ( ++p == pe )
		goto _test_eof245;
case 245:
	switch( (*p) ) {
		case 95: goto tr297;
		case 104: goto st246;
		case 114: goto st248;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st246:
	if ( ++p == pe )
		goto _test_eof246;
case 246:
	switch( (*p) ) {
		case 95: goto tr297;
		case 101: goto st247;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st247:
	if ( ++p == pe )
		goto _test_eof247;
case 247:
	switch( (*p) ) {
		case 95: goto tr297;
		case 110: goto tr457;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st248:
	if ( ++p == pe )
		goto _test_eof248;
case 248:
	switch( (*p) ) {
		case 95: goto tr297;
		case 105: goto st249;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st249:
	if ( ++p == pe )
		goto _test_eof249;
case 249:
	switch( (*p) ) {
		case 95: goto tr297;
		case 116: goto st250;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st250:
	if ( ++p == pe )
		goto _test_eof250;
case 250:
	switch( (*p) ) {
		case 95: goto tr297;
		case 101: goto tr460;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr297;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr297;
	} else
		goto tr297;
	goto tr370;
st251:
	if ( ++p == pe )
		goto _test_eof251;
case 251:
	if ( (*p) == 42 )
		goto tr461;
	goto tr315;
tr313:
#line 1 "NONE"
	{te = p+1;}
	goto st252;
st252:
	if ( ++p == pe )
		goto _test_eof252;
case 252:
#line 6374 "rlscan.cpp"
	if ( (*p) == 37 )
		goto st30;
	goto tr315;
st30:
	if ( ++p == pe )
		goto _test_eof30;
case 30:
	if ( (*p) == 37 )
		goto tr57;
	goto tr45;
tr58:
#line 1146 "rlscan.rl"
	{{p = ((te))-1;}{ pass( *ts, 0, 0 ); }}
	goto st253;
tr61:
#line 1130 "rlscan.rl"
	{te = p+1;{ pass( IMP_Literal, ts, te ); }}
	goto st253;
tr64:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
#line 1128 "rlscan.rl"
	{te = p+1;{ pass(); }}
	goto st253;
tr463:
#line 1146 "rlscan.rl"
	{te = p+1;{ pass( *ts, 0, 0 ); }}
	goto st253;
tr464:
#line 1145 "rlscan.rl"
	{te = p+1;}
	goto st253;
tr474:
#line 1144 "rlscan.rl"
	{te = p;p--;{ pass(); }}
	goto st253;
tr475:
#line 1146 "rlscan.rl"
	{te = p;p--;{ pass( *ts, 0, 0 ); }}
	goto st253;
tr477:
#line 1138 "rlscan.rl"
	{te = p;p--;{ 
			updateCol();
			singleLineSpec = true;
			startSection();
			{stack[top++] = 253; goto st146;}
		}}
	goto st253;
tr478:
#line 1132 "rlscan.rl"
	{te = p+1;{ 
			updateCol();
			singleLineSpec = false;
			startSection();
			{stack[top++] = 253; goto st146;}
		}}
	goto st253;
tr479:
#line 1127 "rlscan.rl"
	{te = p;p--;{ pass( IMP_UInt, ts, te ); }}
	goto st253;
tr480:
#line 1126 "rlscan.rl"
	{te = p;p--;{ pass( IMP_Word, ts, te ); }}
	goto st253;
st253:
#line 1 "NONE"
	{ts = 0;}
	if ( ++p == pe )
		goto _test_eof253;
case 253:
#line 1 "NONE"
	{ts = p;}
#line 6453 "rlscan.cpp"
	switch( (*p) ) {
		case 0: goto tr464;
		case 9: goto st254;
		case 10: goto tr466;
		case 32: goto st254;
		case 34: goto tr467;
		case 35: goto tr468;
		case 37: goto st257;
		case 39: goto tr470;
		case 47: goto tr471;
		case 95: goto st262;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st261;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto st262;
	} else
		goto st262;
	goto tr463;
tr466:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
	goto st254;
st254:
	if ( ++p == pe )
		goto _test_eof254;
case 254:
#line 6487 "rlscan.cpp"
	switch( (*p) ) {
		case 9: goto st254;
		case 10: goto tr466;
		case 32: goto st254;
	}
	goto tr474;
tr467:
#line 1 "NONE"
	{te = p+1;}
	goto st255;
st255:
	if ( ++p == pe )
		goto _test_eof255;
case 255:
#line 6502 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr60;
		case 34: goto tr61;
		case 92: goto st32;
	}
	goto st31;
tr60:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
	goto st31;
st31:
	if ( ++p == pe )
		goto _test_eof31;
case 31:
#line 6521 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr60;
		case 34: goto tr61;
		case 92: goto st32;
	}
	goto st31;
st32:
	if ( ++p == pe )
		goto _test_eof32;
case 32:
	if ( (*p) == 10 )
		goto tr60;
	goto st31;
tr468:
#line 1 "NONE"
	{te = p+1;}
	goto st256;
st256:
	if ( ++p == pe )
		goto _test_eof256;
case 256:
#line 6543 "rlscan.cpp"
	if ( (*p) == 10 )
		goto tr64;
	goto st33;
st33:
	if ( ++p == pe )
		goto _test_eof33;
case 33:
	if ( (*p) == 10 )
		goto tr64;
	goto st33;
st257:
	if ( ++p == pe )
		goto _test_eof257;
case 257:
	if ( (*p) == 37 )
		goto st258;
	goto tr475;
st258:
	if ( ++p == pe )
		goto _test_eof258;
case 258:
	if ( (*p) == 123 )
		goto tr478;
	goto tr477;
tr470:
#line 1 "NONE"
	{te = p+1;}
	goto st259;
st259:
	if ( ++p == pe )
		goto _test_eof259;
case 259:
#line 6576 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr66;
		case 39: goto tr61;
		case 92: goto st35;
	}
	goto st34;
tr66:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
	goto st34;
st34:
	if ( ++p == pe )
		goto _test_eof34;
case 34:
#line 6595 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr66;
		case 39: goto tr61;
		case 92: goto st35;
	}
	goto st34;
st35:
	if ( ++p == pe )
		goto _test_eof35;
case 35:
	if ( (*p) == 10 )
		goto tr66;
	goto st34;
tr471:
#line 1 "NONE"
	{te = p+1;}
	goto st260;
st260:
	if ( ++p == pe )
		goto _test_eof260;
case 260:
#line 6617 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr69;
		case 47: goto tr61;
		case 92: goto st37;
	}
	goto st36;
tr69:
#line 641 "rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
	goto st36;
st36:
	if ( ++p == pe )
		goto _test_eof36;
case 36:
#line 6636 "rlscan.cpp"
	switch( (*p) ) {
		case 10: goto tr69;
		case 47: goto tr61;
		case 92: goto st37;
	}
	goto st36;
st37:
	if ( ++p == pe )
		goto _test_eof37;
case 37:
	if ( (*p) == 10 )
		goto tr69;
	goto st36;
st261:
	if ( ++p == pe )
		goto _test_eof261;
case 261:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto st261;
	goto tr479;
st262:
	if ( ++p == pe )
		goto _test_eof262;
case 262:
	if ( (*p) == 95 )
		goto st262;
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st262;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto st262;
	} else
		goto st262;
	goto tr480;
	}
	_test_eof38: cs = 38; goto _test_eof; 
	_test_eof39: cs = 39; goto _test_eof; 
	_test_eof40: cs = 40; goto _test_eof; 
	_test_eof1: cs = 1; goto _test_eof; 
	_test_eof2: cs = 2; goto _test_eof; 
	_test_eof41: cs = 41; goto _test_eof; 
	_test_eof42: cs = 42; goto _test_eof; 
	_test_eof43: cs = 43; goto _test_eof; 
	_test_eof3: cs = 3; goto _test_eof; 
	_test_eof4: cs = 4; goto _test_eof; 
	_test_eof44: cs = 44; goto _test_eof; 
	_test_eof5: cs = 5; goto _test_eof; 
	_test_eof6: cs = 6; goto _test_eof; 
	_test_eof7: cs = 7; goto _test_eof; 
	_test_eof45: cs = 45; goto _test_eof; 
	_test_eof46: cs = 46; goto _test_eof; 
	_test_eof47: cs = 47; goto _test_eof; 
	_test_eof48: cs = 48; goto _test_eof; 
	_test_eof49: cs = 49; goto _test_eof; 
	_test_eof50: cs = 50; goto _test_eof; 
	_test_eof51: cs = 51; goto _test_eof; 
	_test_eof52: cs = 52; goto _test_eof; 
	_test_eof53: cs = 53; goto _test_eof; 
	_test_eof54: cs = 54; goto _test_eof; 
	_test_eof8: cs = 8; goto _test_eof; 
	_test_eof9: cs = 9; goto _test_eof; 
	_test_eof55: cs = 55; goto _test_eof; 
	_test_eof10: cs = 10; goto _test_eof; 
	_test_eof56: cs = 56; goto _test_eof; 
	_test_eof11: cs = 11; goto _test_eof; 
	_test_eof12: cs = 12; goto _test_eof; 
	_test_eof57: cs = 57; goto _test_eof; 
	_test_eof13: cs = 13; goto _test_eof; 
	_test_eof14: cs = 14; goto _test_eof; 
	_test_eof58: cs = 58; goto _test_eof; 
	_test_eof59: cs = 59; goto _test_eof; 
	_test_eof15: cs = 15; goto _test_eof; 
	_test_eof60: cs = 60; goto _test_eof; 
	_test_eof61: cs = 61; goto _test_eof; 
	_test_eof62: cs = 62; goto _test_eof; 
	_test_eof63: cs = 63; goto _test_eof; 
	_test_eof64: cs = 64; goto _test_eof; 
	_test_eof65: cs = 65; goto _test_eof; 
	_test_eof66: cs = 66; goto _test_eof; 
	_test_eof67: cs = 67; goto _test_eof; 
	_test_eof68: cs = 68; goto _test_eof; 
	_test_eof69: cs = 69; goto _test_eof; 
	_test_eof70: cs = 70; goto _test_eof; 
	_test_eof71: cs = 71; goto _test_eof; 
	_test_eof72: cs = 72; goto _test_eof; 
	_test_eof73: cs = 73; goto _test_eof; 
	_test_eof74: cs = 74; goto _test_eof; 
	_test_eof75: cs = 75; goto _test_eof; 
	_test_eof76: cs = 76; goto _test_eof; 
	_test_eof77: cs = 77; goto _test_eof; 
	_test_eof78: cs = 78; goto _test_eof; 
	_test_eof79: cs = 79; goto _test_eof; 
	_test_eof80: cs = 80; goto _test_eof; 
	_test_eof81: cs = 81; goto _test_eof; 
	_test_eof82: cs = 82; goto _test_eof; 
	_test_eof83: cs = 83; goto _test_eof; 
	_test_eof84: cs = 84; goto _test_eof; 
	_test_eof85: cs = 85; goto _test_eof; 
	_test_eof86: cs = 86; goto _test_eof; 
	_test_eof87: cs = 87; goto _test_eof; 
	_test_eof88: cs = 88; goto _test_eof; 
	_test_eof89: cs = 89; goto _test_eof; 
	_test_eof90: cs = 90; goto _test_eof; 
	_test_eof91: cs = 91; goto _test_eof; 
	_test_eof92: cs = 92; goto _test_eof; 
	_test_eof93: cs = 93; goto _test_eof; 
	_test_eof94: cs = 94; goto _test_eof; 
	_test_eof95: cs = 95; goto _test_eof; 
	_test_eof96: cs = 96; goto _test_eof; 
	_test_eof97: cs = 97; goto _test_eof; 
	_test_eof16: cs = 16; goto _test_eof; 
	_test_eof17: cs = 17; goto _test_eof; 
	_test_eof98: cs = 98; goto _test_eof; 
	_test_eof18: cs = 18; goto _test_eof; 
	_test_eof19: cs = 19; goto _test_eof; 
	_test_eof99: cs = 99; goto _test_eof; 
	_test_eof20: cs = 20; goto _test_eof; 
	_test_eof21: cs = 21; goto _test_eof; 
	_test_eof22: cs = 22; goto _test_eof; 
	_test_eof100: cs = 100; goto _test_eof; 
	_test_eof101: cs = 101; goto _test_eof; 
	_test_eof23: cs = 23; goto _test_eof; 
	_test_eof102: cs = 102; goto _test_eof; 
	_test_eof103: cs = 103; goto _test_eof; 
	_test_eof104: cs = 104; goto _test_eof; 
	_test_eof105: cs = 105; goto _test_eof; 
	_test_eof106: cs = 106; goto _test_eof; 
	_test_eof107: cs = 107; goto _test_eof; 
	_test_eof108: cs = 108; goto _test_eof; 
	_test_eof109: cs = 109; goto _test_eof; 
	_test_eof110: cs = 110; goto _test_eof; 
	_test_eof111: cs = 111; goto _test_eof; 
	_test_eof112: cs = 112; goto _test_eof; 
	_test_eof113: cs = 113; goto _test_eof; 
	_test_eof114: cs = 114; goto _test_eof; 
	_test_eof115: cs = 115; goto _test_eof; 
	_test_eof116: cs = 116; goto _test_eof; 
	_test_eof117: cs = 117; goto _test_eof; 
	_test_eof118: cs = 118; goto _test_eof; 
	_test_eof119: cs = 119; goto _test_eof; 
	_test_eof120: cs = 120; goto _test_eof; 
	_test_eof121: cs = 121; goto _test_eof; 
	_test_eof122: cs = 122; goto _test_eof; 
	_test_eof123: cs = 123; goto _test_eof; 
	_test_eof124: cs = 124; goto _test_eof; 
	_test_eof125: cs = 125; goto _test_eof; 
	_test_eof126: cs = 126; goto _test_eof; 
	_test_eof127: cs = 127; goto _test_eof; 
	_test_eof128: cs = 128; goto _test_eof; 
	_test_eof129: cs = 129; goto _test_eof; 
	_test_eof130: cs = 130; goto _test_eof; 
	_test_eof131: cs = 131; goto _test_eof; 
	_test_eof132: cs = 132; goto _test_eof; 
	_test_eof133: cs = 133; goto _test_eof; 
	_test_eof134: cs = 134; goto _test_eof; 
	_test_eof135: cs = 135; goto _test_eof; 
	_test_eof136: cs = 136; goto _test_eof; 
	_test_eof137: cs = 137; goto _test_eof; 
	_test_eof138: cs = 138; goto _test_eof; 
	_test_eof139: cs = 139; goto _test_eof; 
	_test_eof140: cs = 140; goto _test_eof; 
	_test_eof141: cs = 141; goto _test_eof; 
	_test_eof142: cs = 142; goto _test_eof; 
	_test_eof143: cs = 143; goto _test_eof; 
	_test_eof144: cs = 144; goto _test_eof; 
	_test_eof145: cs = 145; goto _test_eof; 
	_test_eof146: cs = 146; goto _test_eof; 
	_test_eof147: cs = 147; goto _test_eof; 
	_test_eof148: cs = 148; goto _test_eof; 
	_test_eof24: cs = 24; goto _test_eof; 
	_test_eof149: cs = 149; goto _test_eof; 
	_test_eof25: cs = 25; goto _test_eof; 
	_test_eof150: cs = 150; goto _test_eof; 
	_test_eof26: cs = 26; goto _test_eof; 
	_test_eof151: cs = 151; goto _test_eof; 
	_test_eof152: cs = 152; goto _test_eof; 
	_test_eof153: cs = 153; goto _test_eof; 
	_test_eof27: cs = 27; goto _test_eof; 
	_test_eof28: cs = 28; goto _test_eof; 
	_test_eof154: cs = 154; goto _test_eof; 
	_test_eof155: cs = 155; goto _test_eof; 
	_test_eof156: cs = 156; goto _test_eof; 
	_test_eof157: cs = 157; goto _test_eof; 
	_test_eof158: cs = 158; goto _test_eof; 
	_test_eof29: cs = 29; goto _test_eof; 
	_test_eof159: cs = 159; goto _test_eof; 
	_test_eof160: cs = 160; goto _test_eof; 
	_test_eof161: cs = 161; goto _test_eof; 
	_test_eof162: cs = 162; goto _test_eof; 
	_test_eof163: cs = 163; goto _test_eof; 
	_test_eof164: cs = 164; goto _test_eof; 
	_test_eof165: cs = 165; goto _test_eof; 
	_test_eof166: cs = 166; goto _test_eof; 
	_test_eof167: cs = 167; goto _test_eof; 
	_test_eof168: cs = 168; goto _test_eof; 
	_test_eof169: cs = 169; goto _test_eof; 
	_test_eof170: cs = 170; goto _test_eof; 
	_test_eof171: cs = 171; goto _test_eof; 
	_test_eof172: cs = 172; goto _test_eof; 
	_test_eof173: cs = 173; goto _test_eof; 
	_test_eof174: cs = 174; goto _test_eof; 
	_test_eof175: cs = 175; goto _test_eof; 
	_test_eof176: cs = 176; goto _test_eof; 
	_test_eof177: cs = 177; goto _test_eof; 
	_test_eof178: cs = 178; goto _test_eof; 
	_test_eof179: cs = 179; goto _test_eof; 
	_test_eof180: cs = 180; goto _test_eof; 
	_test_eof181: cs = 181; goto _test_eof; 
	_test_eof182: cs = 182; goto _test_eof; 
	_test_eof183: cs = 183; goto _test_eof; 
	_test_eof184: cs = 184; goto _test_eof; 
	_test_eof185: cs = 185; goto _test_eof; 
	_test_eof186: cs = 186; goto _test_eof; 
	_test_eof187: cs = 187; goto _test_eof; 
	_test_eof188: cs = 188; goto _test_eof; 
	_test_eof189: cs = 189; goto _test_eof; 
	_test_eof190: cs = 190; goto _test_eof; 
	_test_eof191: cs = 191; goto _test_eof; 
	_test_eof192: cs = 192; goto _test_eof; 
	_test_eof193: cs = 193; goto _test_eof; 
	_test_eof194: cs = 194; goto _test_eof; 
	_test_eof195: cs = 195; goto _test_eof; 
	_test_eof196: cs = 196; goto _test_eof; 
	_test_eof197: cs = 197; goto _test_eof; 
	_test_eof198: cs = 198; goto _test_eof; 
	_test_eof199: cs = 199; goto _test_eof; 
	_test_eof200: cs = 200; goto _test_eof; 
	_test_eof201: cs = 201; goto _test_eof; 
	_test_eof202: cs = 202; goto _test_eof; 
	_test_eof203: cs = 203; goto _test_eof; 
	_test_eof204: cs = 204; goto _test_eof; 
	_test_eof205: cs = 205; goto _test_eof; 
	_test_eof206: cs = 206; goto _test_eof; 
	_test_eof207: cs = 207; goto _test_eof; 
	_test_eof208: cs = 208; goto _test_eof; 
	_test_eof209: cs = 209; goto _test_eof; 
	_test_eof210: cs = 210; goto _test_eof; 
	_test_eof211: cs = 211; goto _test_eof; 
	_test_eof212: cs = 212; goto _test_eof; 
	_test_eof213: cs = 213; goto _test_eof; 
	_test_eof214: cs = 214; goto _test_eof; 
	_test_eof215: cs = 215; goto _test_eof; 
	_test_eof216: cs = 216; goto _test_eof; 
	_test_eof217: cs = 217; goto _test_eof; 
	_test_eof218: cs = 218; goto _test_eof; 
	_test_eof219: cs = 219; goto _test_eof; 
	_test_eof220: cs = 220; goto _test_eof; 
	_test_eof221: cs = 221; goto _test_eof; 
	_test_eof222: cs = 222; goto _test_eof; 
	_test_eof223: cs = 223; goto _test_eof; 
	_test_eof224: cs = 224; goto _test_eof; 
	_test_eof225: cs = 225; goto _test_eof; 
	_test_eof226: cs = 226; goto _test_eof; 
	_test_eof227: cs = 227; goto _test_eof; 
	_test_eof228: cs = 228; goto _test_eof; 
	_test_eof229: cs = 229; goto _test_eof; 
	_test_eof230: cs = 230; goto _test_eof; 
	_test_eof231: cs = 231; goto _test_eof; 
	_test_eof232: cs = 232; goto _test_eof; 
	_test_eof233: cs = 233; goto _test_eof; 
	_test_eof234: cs = 234; goto _test_eof; 
	_test_eof235: cs = 235; goto _test_eof; 
	_test_eof236: cs = 236; goto _test_eof; 
	_test_eof237: cs = 237; goto _test_eof; 
	_test_eof238: cs = 238; goto _test_eof; 
	_test_eof239: cs = 239; goto _test_eof; 
	_test_eof240: cs = 240; goto _test_eof; 
	_test_eof241: cs = 241; goto _test_eof; 
	_test_eof242: cs = 242; goto _test_eof; 
	_test_eof243: cs = 243; goto _test_eof; 
	_test_eof244: cs = 244; goto _test_eof; 
	_test_eof245: cs = 245; goto _test_eof; 
	_test_eof246: cs = 246; goto _test_eof; 
	_test_eof247: cs = 247; goto _test_eof; 
	_test_eof248: cs = 248; goto _test_eof; 
	_test_eof249: cs = 249; goto _test_eof; 
	_test_eof250: cs = 250; goto _test_eof; 
	_test_eof251: cs = 251; goto _test_eof; 
	_test_eof252: cs = 252; goto _test_eof; 
	_test_eof30: cs = 30; goto _test_eof; 
	_test_eof253: cs = 253; goto _test_eof; 
	_test_eof254: cs = 254; goto _test_eof; 
	_test_eof255: cs = 255; goto _test_eof; 
	_test_eof31: cs = 31; goto _test_eof; 
	_test_eof32: cs = 32; goto _test_eof; 
	_test_eof256: cs = 256; goto _test_eof; 
	_test_eof33: cs = 33; goto _test_eof; 
	_test_eof257: cs = 257; goto _test_eof; 
	_test_eof258: cs = 258; goto _test_eof; 
	_test_eof259: cs = 259; goto _test_eof; 
	_test_eof34: cs = 34; goto _test_eof; 
	_test_eof35: cs = 35; goto _test_eof; 
	_test_eof260: cs = 260; goto _test_eof; 
	_test_eof36: cs = 36; goto _test_eof; 
	_test_eof37: cs = 37; goto _test_eof; 
	_test_eof261: cs = 261; goto _test_eof; 
	_test_eof262: cs = 262; goto _test_eof; 

	_test_eof: {}
	if ( p == eof )
	{
	switch ( cs ) {
	case 39: goto tr82;
	case 40: goto tr83;
	case 1: goto tr0;
	case 2: goto tr0;
	case 41: goto tr83;
	case 42: goto tr85;
	case 43: goto tr83;
	case 3: goto tr0;
	case 4: goto tr0;
	case 44: goto tr83;
	case 5: goto tr0;
	case 6: goto tr0;
	case 7: goto tr0;
	case 45: goto tr87;
	case 46: goto tr88;
	case 47: goto tr89;
	case 48: goto tr89;
	case 49: goto tr89;
	case 50: goto tr89;
	case 51: goto tr89;
	case 53: goto tr113;
	case 54: goto tr114;
	case 8: goto tr14;
	case 9: goto tr14;
	case 55: goto tr114;
	case 10: goto tr14;
	case 56: goto tr114;
	case 11: goto tr14;
	case 12: goto tr14;
	case 57: goto tr114;
	case 13: goto tr14;
	case 14: goto tr14;
	case 58: goto tr115;
	case 59: goto tr115;
	case 15: goto tr27;
	case 60: goto tr117;
	case 61: goto tr114;
	case 62: goto tr119;
	case 63: goto tr120;
	case 64: goto tr120;
	case 65: goto tr120;
	case 66: goto tr120;
	case 67: goto tr120;
	case 68: goto tr134;
	case 69: goto tr120;
	case 70: goto tr120;
	case 71: goto tr120;
	case 72: goto tr120;
	case 73: goto tr120;
	case 74: goto tr120;
	case 75: goto tr120;
	case 76: goto tr120;
	case 77: goto tr120;
	case 78: goto tr120;
	case 79: goto tr120;
	case 80: goto tr120;
	case 81: goto tr120;
	case 82: goto tr120;
	case 83: goto tr120;
	case 84: goto tr120;
	case 85: goto tr120;
	case 86: goto tr120;
	case 87: goto tr120;
	case 88: goto tr120;
	case 89: goto tr120;
	case 90: goto tr120;
	case 91: goto tr120;
	case 92: goto tr120;
	case 93: goto tr120;
	case 94: goto tr120;
	case 96: goto tr181;
	case 97: goto tr182;
	case 16: goto tr29;
	case 17: goto tr29;
	case 98: goto tr182;
	case 18: goto tr29;
	case 19: goto tr29;
	case 99: goto tr182;
	case 20: goto tr29;
	case 21: goto tr29;
	case 22: goto tr29;
	case 100: goto tr183;
	case 101: goto tr183;
	case 23: goto tr43;
	case 102: goto tr185;
	case 103: goto tr182;
	case 104: goto tr187;
	case 105: goto tr188;
	case 106: goto tr188;
	case 107: goto tr188;
	case 108: goto tr188;
	case 109: goto tr188;
	case 110: goto tr202;
	case 111: goto tr188;
	case 112: goto tr188;
	case 113: goto tr188;
	case 114: goto tr188;
	case 115: goto tr188;
	case 116: goto tr188;
	case 117: goto tr188;
	case 118: goto tr188;
	case 119: goto tr188;
	case 120: goto tr188;
	case 121: goto tr188;
	case 122: goto tr188;
	case 123: goto tr188;
	case 124: goto tr188;
	case 125: goto tr188;
	case 126: goto tr188;
	case 127: goto tr188;
	case 128: goto tr188;
	case 129: goto tr188;
	case 130: goto tr188;
	case 131: goto tr188;
	case 132: goto tr188;
	case 133: goto tr188;
	case 134: goto tr188;
	case 135: goto tr188;
	case 136: goto tr188;
	case 138: goto tr237;
	case 140: goto tr255;
	case 141: goto tr257;
	case 142: goto tr259;
	case 144: goto tr275;
	case 145: goto tr276;
	case 147: goto tr314;
	case 148: goto tr315;
	case 24: goto tr45;
	case 149: goto tr316;
	case 25: goto tr45;
	case 150: goto tr315;
	case 26: goto tr45;
	case 151: goto tr315;
	case 152: goto tr315;
	case 153: goto tr315;
	case 27: goto tr45;
	case 28: goto tr45;
	case 154: goto tr315;
	case 155: goto tr315;
	case 156: goto tr315;
	case 157: goto tr334;
	case 158: goto tr334;
	case 29: goto tr55;
	case 159: goto tr336;
	case 160: goto tr315;
	case 161: goto tr340;
	case 162: goto tr315;
	case 163: goto tr349;
	case 164: goto tr315;
	case 165: goto tr315;
	case 166: goto tr315;
	case 167: goto tr367;
	case 168: goto tr368;
	case 169: goto tr370;
	case 170: goto tr370;
	case 171: goto tr370;
	case 172: goto tr370;
	case 173: goto tr370;
	case 174: goto tr370;
	case 175: goto tr370;
	case 176: goto tr370;
	case 177: goto tr370;
	case 178: goto tr370;
	case 179: goto tr370;
	case 180: goto tr370;
	case 181: goto tr370;
	case 182: goto tr370;
	case 183: goto tr370;
	case 184: goto tr370;
	case 185: goto tr370;
	case 186: goto tr370;
	case 187: goto tr370;
	case 188: goto tr370;
	case 189: goto tr370;
	case 190: goto tr370;
	case 191: goto tr370;
	case 192: goto tr370;
	case 193: goto tr370;
	case 194: goto tr370;
	case 195: goto tr370;
	case 196: goto tr370;
	case 197: goto tr370;
	case 198: goto tr370;
	case 199: goto tr370;
	case 200: goto tr370;
	case 201: goto tr370;
	case 202: goto tr370;
	case 203: goto tr370;
	case 204: goto tr370;
	case 205: goto tr370;
	case 206: goto tr370;
	case 207: goto tr370;
	case 208: goto tr370;
	case 209: goto tr370;
	case 210: goto tr370;
	case 211: goto tr370;
	case 212: goto tr370;
	case 213: goto tr370;
	case 214: goto tr370;
	case 215: goto tr370;
	case 216: goto tr370;
	case 217: goto tr370;
	case 218: goto tr370;
	case 219: goto tr370;
	case 220: goto tr370;
	case 221: goto tr370;
	case 222: goto tr370;
	case 223: goto tr370;
	case 224: goto tr370;
	case 225: goto tr370;
	case 226: goto tr370;
	case 227: goto tr370;
	case 228: goto tr370;
	case 229: goto tr370;
	case 230: goto tr370;
	case 231: goto tr370;
	case 232: goto tr370;
	case 233: goto tr370;
	case 234: goto tr370;
	case 235: goto tr370;
	case 236: goto tr370;
	case 237: goto tr370;
	case 238: goto tr370;
	case 239: goto tr370;
	case 240: goto tr370;
	case 241: goto tr370;
	case 242: goto tr370;
	case 243: goto tr370;
	case 244: goto tr370;
	case 245: goto tr370;
	case 246: goto tr370;
	case 247: goto tr370;
	case 248: goto tr370;
	case 249: goto tr370;
	case 250: goto tr370;
	case 251: goto tr315;
	case 252: goto tr315;
	case 30: goto tr45;
	case 254: goto tr474;
	case 255: goto tr475;
	case 31: goto tr58;
	case 32: goto tr58;
	case 256: goto tr475;
	case 33: goto tr58;
	case 257: goto tr475;
	case 258: goto tr477;
	case 259: goto tr475;
	case 34: goto tr58;
	case 35: goto tr58;
	case 260: goto tr475;
	case 36: goto tr58;
	case 37: goto tr58;
	case 261: goto tr479;
	case 262: goto tr480;
	}
	}

	_out: {}
	}

#line 1241 "rlscan.rl"

		/* Check if we failed. */
		if ( cs == rlscan_error ) {
			/* Machine failed before finding a token. I'm not yet sure if this
			 * is reachable. */
			scan_error() << "scanner error" << endl;
			exit(1);
		}

		/* Decide if we need to preserve anything. */
		char *preserve = ts;

		/* Now set up the prefix. */
		if ( preserve == 0 )
			have = 0;
		else {
			/* There is data that needs to be shifted over. */
			have = pe - preserve;
			memmove( buf, preserve, have );
			unsigned int shiftback = preserve - buf;
			if ( ts != 0 )
				ts -= shiftback;
			te -= shiftback;

			preserve = buf;
		}
	}

	delete[] buf;
}
