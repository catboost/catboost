
#line 1 "./rlscan.rl"
/*
 *  Copyright 2006-2007 Adrian Thurston <thurston@complang.org>
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


#line 118 "./rlscan.rl"



#line 58 "rlscan.fast_build.cpp"
static const int _inline_token_scan_trans_keys[] = {
	129, 130, 129, 130, 128, 131, 61, 61, 128, 128, 0
};

static const char _inline_token_scan_key_spans[] = {
	2, 2, 4, 1, 1
};

static const char _inline_token_scan_index_offsets[] = {
	0, 3, 6, 11, 13
};

static const char _inline_token_scan_indicies[] = {
	1, 2, 0, 3, 4, 0, 6, 5, 
	5, 7, 5, 9, 8, 10, 8, 0
};

static const char _inline_token_scan_trans_targs[] = {
	2, 2, 2, 2, 2, 2, 3, 4, 
	2, 0, 1
};

static const char _inline_token_scan_trans_actions[] = {
	1, 2, 3, 4, 5, 8, 9, 9, 
	10, 0, 0
};

static const char _inline_token_scan_to_state_actions[] = {
	0, 0, 6, 0, 0
};

static const char _inline_token_scan_from_state_actions[] = {
	0, 0, 7, 0, 0
};

static const char _inline_token_scan_eof_trans[] = {
	1, 1, 0, 9, 9
};

enum {inline_token_scan_start = 2};
enum {inline_token_scan_first_final = 2};
enum {inline_token_scan_error = -1};

enum {inline_token_scan_en_main = 2};


#line 121 "./rlscan.rl"

void Scanner::flushImport()
{
	int *p = token_data;
	int *pe = token_data + cur_token;
	int *eof = 0;

	
#line 114 "rlscan.fast_build.cpp"
	{
	 tok_cs = inline_token_scan_start;
	 tok_ts = 0;
	 tok_te = 0;
	 tok_act = 0;
	}

#line 122 "rlscan.fast_build.cpp"
	{
	int _slen;
	int _trans;
	const int *_keys;
	const char *_inds;
	if ( p == pe )
		goto _test_eof;
_resume:
	switch ( _inline_token_scan_from_state_actions[ tok_cs] ) {
	case 7:
#line 1 "NONE"
	{ tok_ts = p;}
	break;
#line 136 "rlscan.fast_build.cpp"
	}

	_keys = _inline_token_scan_trans_keys + ( tok_cs<<1);
	_inds = _inline_token_scan_indicies + _inline_token_scan_index_offsets[ tok_cs];

	_slen = _inline_token_scan_key_spans[ tok_cs];
	_trans = _inds[ _slen > 0 && _keys[0] <=(*p) &&
		(*p) <= _keys[1] ?
		(*p) - _keys[0] : _slen ];

_eof_trans:
	 tok_cs = _inline_token_scan_trans_targs[_trans];

	if ( _inline_token_scan_trans_actions[_trans] == 0 )
		goto _again;

	switch ( _inline_token_scan_trans_actions[_trans] ) {
	case 9:
#line 1 "NONE"
	{ tok_te = p+1;}
	break;
	case 5:
#line 60 "./rlscan.rl"
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
	break;
	case 3:
#line 74 "./rlscan.rl"
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
	break;
	case 4:
#line 88 "./rlscan.rl"
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
	break;
	case 2:
#line 102 "./rlscan.rl"
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
	break;
	case 8:
#line 116 "./rlscan.rl"
	{ tok_te = p+1;}
	break;
	case 10:
#line 116 "./rlscan.rl"
	{ tok_te = p;p--;}
	break;
	case 1:
#line 116 "./rlscan.rl"
	{{p = (( tok_te))-1;}}
	break;
#line 230 "rlscan.fast_build.cpp"
	}

_again:
	switch ( _inline_token_scan_to_state_actions[ tok_cs] ) {
	case 6:
#line 1 "NONE"
	{ tok_ts = 0;}
	break;
#line 239 "rlscan.fast_build.cpp"
	}

	if ( ++p != pe )
		goto _resume;
	_test_eof: {}
	if ( p == eof )
	{
	if ( _inline_token_scan_eof_trans[ tok_cs] > 0 ) {
		_trans = _inline_token_scan_eof_trans[ tok_cs] - 1;
		goto _eof_trans;
	}
	}

	}

#line 132 "./rlscan.rl"


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


#line 332 "rlscan.fast_build.cpp"
static const int _section_parse_trans_keys[] = {
	0, 0, 128, 128, 59, 59, 128, 129, 59, 129, 59, 59, 129, 129, 59, 59, 
	128, 128, 59, 128, 191, 194, 0
};

static const char _section_parse_key_spans[] = {
	0, 1, 1, 2, 71, 1, 1, 1, 
	1, 70, 4
};

static const unsigned char _section_parse_index_offsets[] = {
	0, 0, 2, 4, 7, 79, 81, 83, 
	85, 87, 158
};

static const char _section_parse_indicies[] = {
	1, 0, 2, 0, 4, 5, 3, 
	6, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 7, 3, 
	6, 3, 9, 8, 10, 8, 12, 11, 
	13, 11, 11, 11, 11, 11, 11, 11, 
	11, 11, 11, 11, 11, 11, 11, 11, 
	11, 11, 11, 11, 11, 11, 11, 11, 
	11, 11, 11, 11, 11, 11, 11, 11, 
	11, 11, 11, 11, 11, 11, 11, 11, 
	11, 11, 11, 11, 11, 11, 11, 11, 
	11, 11, 11, 11, 11, 11, 11, 11, 
	11, 11, 11, 11, 11, 11, 11, 11, 
	11, 11, 11, 11, 11, 12, 11, 15, 
	16, 17, 18, 14, 0
};

static const char _section_parse_trans_targs[] = {
	0, 2, 10, 0, 4, 5, 10, 5, 
	0, 7, 10, 0, 9, 10, 10, 1, 
	3, 6, 8
};

static const char _section_parse_trans_actions[] = {
	1, 2, 3, 4, 5, 6, 7, 8, 
	9, 8, 10, 11, 12, 13, 14, 0, 
	0, 0, 15
};

static const char _section_parse_eof_actions[] = {
	0, 1, 1, 4, 4, 4, 9, 9, 
	11, 11, 0
};

enum {section_parse_start = 10};
enum {section_parse_first_final = 10};
enum {section_parse_error = 0};

enum {section_parse_en_main = 10};


#line 211 "./rlscan.rl"



void Scanner::init( )
{
	
#line 403 "rlscan.fast_build.cpp"
	{
	cs = section_parse_start;
	}

#line 217 "./rlscan.rl"
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


#line 454 "./rlscan.rl"


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

	
#line 602 "rlscan.fast_build.cpp"
	{
	int _slen;
	int _trans;
	const int *_keys;
	const char *_inds;
	if ( p == pe )
		goto _test_eof;
	if ( cs == 0 )
		goto _out;
_resume:
	_keys = _section_parse_trans_keys + (cs<<1);
	_inds = _section_parse_indicies + _section_parse_index_offsets[cs];

	_slen = _section_parse_key_spans[cs];
	_trans = _inds[ _slen > 0 && _keys[0] <=(*p) &&
		(*p) <= _keys[1] ?
		(*p) - _keys[0] : _slen ];

	cs = _section_parse_trans_targs[_trans];

	if ( _section_parse_trans_actions[_trans] == 0 )
		goto _again;

	switch ( _section_parse_trans_actions[_trans] ) {
	case 2:
#line 376 "./rlscan.rl"
	{ word = tokdata; word_len = toklen; }
	break;
	case 8:
#line 377 "./rlscan.rl"
	{ lit = tokdata; lit_len = toklen; }
	break;
	case 1:
#line 379 "./rlscan.rl"
	{ scan_error() << "bad machine statement" << endl; }
	break;
	case 4:
#line 380 "./rlscan.rl"
	{ scan_error() << "bad include statement" << endl; }
	break;
	case 9:
#line 381 "./rlscan.rl"
	{ scan_error() << "bad import statement" << endl; }
	break;
	case 11:
#line 382 "./rlscan.rl"
	{ scan_error() << "bad write statement" << endl; }
	break;
	case 3:
#line 384 "./rlscan.rl"
	{ handleMachine(); }
	break;
	case 7:
#line 385 "./rlscan.rl"
	{ handleInclude(); }
	break;
	case 10:
#line 386 "./rlscan.rl"
	{ handleImport(); }
	break;
	case 15:
#line 406 "./rlscan.rl"
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
	break;
	case 12:
#line 420 "./rlscan.rl"
	{
		if ( active() && machineSpec == 0 && machineName == 0 )
			id.inputItems.tail->writeArgs.append( strdup(tokdata) );
	}
	break;
	case 13:
#line 426 "./rlscan.rl"
	{
		if ( active() && machineSpec == 0 && machineName == 0 )
			id.inputItems.tail->writeArgs.append( 0 );
	}
	break;
	case 14:
#line 437 "./rlscan.rl"
	{
		/* Send the token off to the parser. */
		if ( active() )
			directToParser( parser, fileName, line, column, type, tokdata, toklen );
	}
	break;
	case 5:
#line 375 "./rlscan.rl"
	{ word = lit = 0; word_len = lit_len = 0; }
#line 376 "./rlscan.rl"
	{ word = tokdata; word_len = toklen; }
	break;
	case 6:
#line 375 "./rlscan.rl"
	{ word = lit = 0; word_len = lit_len = 0; }
#line 377 "./rlscan.rl"
	{ lit = tokdata; lit_len = toklen; }
	break;
#line 712 "rlscan.fast_build.cpp"
	}

_again:
	if ( cs == 0 )
		goto _out;
	if ( ++p != pe )
		goto _resume;
	_test_eof: {}
	if ( p == eof )
	{
	switch ( _section_parse_eof_actions[cs] ) {
	case 1:
#line 379 "./rlscan.rl"
	{ scan_error() << "bad machine statement" << endl; }
	break;
	case 4:
#line 380 "./rlscan.rl"
	{ scan_error() << "bad include statement" << endl; }
	break;
	case 9:
#line 381 "./rlscan.rl"
	{ scan_error() << "bad import statement" << endl; }
	break;
	case 11:
#line 382 "./rlscan.rl"
	{ scan_error() << "bad write statement" << endl; }
	break;
#line 740 "rlscan.fast_build.cpp"
	}
	}

	_out: {}
	}

#line 495 "./rlscan.rl"


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

inline void resolvePath(const char* rel, const char* abs, char res[]) {
    strcpy(res, abs);
    char* p = strrchr(res, '/');
    *p = 0;
    while (*rel == '.' && *(rel + 1) == '.' && *(rel + 2) == '/') {
        rel += 3;
        p = strrchr(res, '/');
        *p = 0;
    }
    strcat(res, "/");
    strcat(res, rel);
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
			long givenPathLen = (lastSlash - thisFileName) + 1;
			long checklen = givenPathLen + length;
            long abslen = strlen(thisFileName);
            checklen = abslen > checklen ? abslen : checklen;
            char *check = new char[checklen+1];
            resolvePath(data, thisFileName, check);
			checks[nextCheck++] = check;
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
		check += 1;
	}

	found = -1;
	delete inFile;
	return 0;
}


#line 1169 "./rlscan.rl"



#line 889 "rlscan.fast_build.cpp"
static const char _rlscan_trans_keys[] = {
	0, 0, 10, 92, 10, 10, 10, 92, 10, 10, 10, 42, 10, 47, 10, 10, 
	10, 92, 10, 10, 10, 10, 10, 92, 10, 10, 10, 92, 10, 10, 48, 102, 
	10, 92, 10, 10, 10, 92, 10, 10, 10, 42, 10, 47, 10, 10, 48, 102, 
	10, 92, 10, 10, 10, 10, 10, 92, 10, 10, 48, 102, 37, 37, 10, 92, 
	10, 10, 10, 10, 10, 92, 10, 10, 10, 92, 10, 10, 0, 122, 9, 32, 
	10, 92, 37, 37, 123, 123, 10, 92, 42, 47, 48, 57, 48, 122, 48, 122, 
	48, 122, 48, 122, 48, 122, 48, 122, 0, 125, 9, 32, 10, 92, 10, 10, 
	10, 92, 10, 92, 48, 120, 48, 57, 48, 102, 58, 58, 48, 122, 48, 122, 
	48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 
	48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 
	48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 
	48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 0, 125, 
	9, 32, 10, 92, 10, 92, 42, 47, 48, 120, 48, 57, 48, 102, 58, 58, 
	48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 
	48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 
	48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 
	48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 
	48, 122, 0, 93, 10, 118, 0, 92, 105, 105, 94, 94, 10, 118, 0, 122, 
	9, 32, 48, 122, 0, 125, 9, 32, 10, 92, 105, 105, 10, 10, 33, 126, 
	33, 126, 10, 92, 42, 42, 45, 62, 46, 46, 48, 120, 48, 57, 48, 102, 
	58, 62, 62, 62, 33, 126, 33, 126, 62, 62, 33, 126, 33, 126, 48, 122, 
	94, 94, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 
	48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 
	48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 
	48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 
	48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 
	48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 
	48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 
	48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 
	48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 
	48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 48, 122, 
	48, 122, 48, 122, 48, 122, 42, 42, 37, 37, 0, 122, 9, 32, 10, 92, 
	10, 10, 37, 37, 123, 123, 10, 92, 10, 92, 48, 57, 48, 122, 0
};

static const char _rlscan_key_spans[] = {
	0, 83, 1, 83, 1, 33, 38, 1, 
	83, 1, 1, 83, 1, 83, 1, 55, 
	83, 1, 83, 1, 33, 38, 1, 55, 
	83, 1, 1, 83, 1, 55, 1, 83, 
	1, 1, 83, 1, 83, 1, 123, 24, 
	83, 1, 1, 83, 6, 10, 75, 75, 
	75, 75, 75, 75, 126, 24, 83, 1, 
	83, 83, 73, 10, 55, 1, 75, 75, 
	75, 75, 75, 75, 75, 75, 75, 75, 
	75, 75, 75, 75, 75, 75, 75, 75, 
	75, 75, 75, 75, 75, 75, 75, 75, 
	75, 75, 75, 75, 75, 75, 75, 126, 
	24, 83, 83, 6, 73, 10, 55, 1, 
	75, 75, 75, 75, 75, 75, 75, 75, 
	75, 75, 75, 75, 75, 75, 75, 75, 
	75, 75, 75, 75, 75, 75, 75, 75, 
	75, 75, 75, 75, 75, 75, 75, 75, 
	75, 94, 109, 93, 1, 1, 109, 123, 
	24, 75, 126, 24, 83, 1, 1, 94, 
	94, 83, 1, 18, 1, 73, 10, 55, 
	5, 1, 94, 94, 1, 94, 94, 75, 
	1, 75, 75, 75, 75, 75, 75, 75, 
	75, 75, 75, 75, 75, 75, 75, 75, 
	75, 75, 75, 75, 75, 75, 75, 75, 
	75, 75, 75, 75, 75, 75, 75, 75, 
	75, 75, 75, 75, 75, 75, 75, 75, 
	75, 75, 75, 75, 75, 75, 75, 75, 
	75, 75, 75, 75, 75, 75, 75, 75, 
	75, 75, 75, 75, 75, 75, 75, 75, 
	75, 75, 75, 75, 75, 75, 75, 75, 
	75, 75, 75, 75, 75, 75, 75, 75, 
	75, 75, 75, 1, 1, 123, 24, 83, 
	1, 1, 1, 83, 83, 10, 75
};

static const short _rlscan_index_offsets[] = {
	0, 0, 84, 86, 170, 172, 206, 245, 
	247, 331, 333, 335, 419, 421, 505, 507, 
	563, 647, 649, 733, 735, 769, 808, 810, 
	866, 950, 952, 954, 1038, 1040, 1096, 1098, 
	1182, 1184, 1186, 1270, 1272, 1356, 1358, 1482, 
	1507, 1591, 1593, 1595, 1679, 1686, 1697, 1773, 
	1849, 1925, 2001, 2077, 2153, 2280, 2305, 2389, 
	2391, 2475, 2559, 2633, 2644, 2700, 2702, 2778, 
	2854, 2930, 3006, 3082, 3158, 3234, 3310, 3386, 
	3462, 3538, 3614, 3690, 3766, 3842, 3918, 3994, 
	4070, 4146, 4222, 4298, 4374, 4450, 4526, 4602, 
	4678, 4754, 4830, 4906, 4982, 5058, 5134, 5210, 
	5337, 5362, 5446, 5530, 5537, 5611, 5622, 5678, 
	5680, 5756, 5832, 5908, 5984, 6060, 6136, 6212, 
	6288, 6364, 6440, 6516, 6592, 6668, 6744, 6820, 
	6896, 6972, 7048, 7124, 7200, 7276, 7352, 7428, 
	7504, 7580, 7656, 7732, 7808, 7884, 7960, 8036, 
	8112, 8188, 8283, 8393, 8487, 8489, 8491, 8601, 
	8725, 8750, 8826, 8953, 8978, 9062, 9064, 9066, 
	9161, 9256, 9340, 9342, 9361, 9363, 9437, 9448, 
	9504, 9510, 9512, 9607, 9702, 9704, 9799, 9894, 
	9970, 9972, 10048, 10124, 10200, 10276, 10352, 10428, 
	10504, 10580, 10656, 10732, 10808, 10884, 10960, 11036, 
	11112, 11188, 11264, 11340, 11416, 11492, 11568, 11644, 
	11720, 11796, 11872, 11948, 12024, 12100, 12176, 12252, 
	12328, 12404, 12480, 12556, 12632, 12708, 12784, 12860, 
	12936, 13012, 13088, 13164, 13240, 13316, 13392, 13468, 
	13544, 13620, 13696, 13772, 13848, 13924, 14000, 14076, 
	14152, 14228, 14304, 14380, 14456, 14532, 14608, 14684, 
	14760, 14836, 14912, 14988, 15064, 15140, 15216, 15292, 
	15368, 15444, 15520, 15596, 15672, 15748, 15824, 15900, 
	15976, 16052, 16128, 16204, 16206, 16208, 16332, 16357, 
	16441, 16443, 16445, 16447, 16531, 16615, 16626
};

static const short _rlscan_indicies[] = {
	2, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 3, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 4, 1, 2, 1, 6, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 3, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	5, 7, 5, 6, 5, 9, 8, 8, 
	8, 8, 8, 8, 8, 8, 8, 8, 
	8, 8, 8, 8, 8, 8, 8, 8, 
	8, 8, 8, 8, 8, 8, 8, 8, 
	8, 8, 8, 8, 8, 10, 8, 9, 
	8, 8, 8, 8, 8, 8, 8, 8, 
	8, 8, 8, 8, 8, 8, 8, 8, 
	8, 8, 8, 8, 8, 8, 8, 8, 
	8, 8, 8, 8, 8, 8, 8, 10, 
	8, 8, 8, 8, 11, 8, 13, 12, 
	16, 15, 15, 15, 15, 15, 15, 15, 
	15, 15, 15, 15, 15, 15, 15, 15, 
	15, 15, 15, 15, 15, 15, 15, 15, 
	17, 15, 15, 15, 15, 15, 15, 15, 
	15, 15, 15, 15, 15, 15, 15, 15, 
	15, 15, 15, 15, 15, 15, 15, 15, 
	15, 15, 15, 15, 15, 15, 15, 15, 
	15, 15, 15, 15, 15, 15, 15, 15, 
	15, 15, 15, 15, 15, 15, 15, 15, 
	15, 15, 15, 15, 15, 15, 15, 15, 
	15, 15, 18, 15, 16, 15, 20, 19, 
	22, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 17, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 23, 21, 22, 21, 25, 24, 
	24, 24, 24, 24, 24, 24, 24, 24, 
	24, 24, 24, 24, 24, 24, 24, 24, 
	24, 24, 24, 24, 24, 24, 24, 24, 
	24, 24, 24, 24, 24, 24, 24, 24, 
	24, 24, 24, 17, 24, 24, 24, 24, 
	24, 24, 24, 24, 24, 24, 24, 24, 
	24, 24, 24, 24, 24, 24, 24, 24, 
	24, 24, 24, 24, 24, 24, 24, 24, 
	24, 24, 24, 24, 24, 24, 24, 24, 
	24, 24, 24, 24, 24, 24, 24, 24, 
	26, 24, 25, 24, 28, 28, 28, 28, 
	28, 28, 28, 28, 28, 28, 27, 27, 
	27, 27, 27, 27, 27, 28, 28, 28, 
	28, 28, 28, 27, 27, 27, 27, 27, 
	27, 27, 27, 27, 27, 27, 27, 27, 
	27, 27, 27, 27, 27, 27, 27, 27, 
	27, 27, 27, 27, 27, 28, 28, 28, 
	28, 28, 28, 27, 31, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 32, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 33, 30, 
	31, 30, 35, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 32, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 36, 34, 35, 34, 
	38, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	39, 37, 38, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 37, 39, 37, 37, 37, 37, 40, 
	37, 42, 41, 44, 44, 44, 44, 44, 
	44, 44, 44, 44, 44, 43, 43, 43, 
	43, 43, 43, 43, 44, 44, 44, 44, 
	44, 44, 43, 43, 43, 43, 43, 43, 
	43, 43, 43, 43, 43, 43, 43, 43, 
	43, 43, 43, 43, 43, 43, 43, 43, 
	43, 43, 43, 43, 44, 44, 44, 44, 
	44, 44, 43, 47, 46, 46, 46, 46, 
	46, 46, 46, 46, 46, 46, 46, 46, 
	46, 46, 46, 46, 46, 46, 46, 46, 
	46, 46, 46, 48, 46, 46, 46, 46, 
	46, 46, 46, 46, 46, 46, 46, 46, 
	46, 46, 46, 46, 46, 46, 46, 46, 
	46, 46, 46, 46, 46, 46, 46, 46, 
	46, 46, 46, 46, 46, 46, 46, 46, 
	46, 46, 46, 46, 46, 46, 46, 46, 
	46, 46, 46, 46, 46, 46, 46, 46, 
	46, 46, 46, 46, 46, 49, 46, 47, 
	46, 51, 50, 53, 52, 52, 52, 52, 
	52, 52, 52, 52, 52, 52, 52, 52, 
	52, 52, 52, 52, 52, 52, 52, 52, 
	52, 52, 52, 52, 52, 52, 52, 52, 
	48, 52, 52, 52, 52, 52, 52, 52, 
	52, 52, 52, 52, 52, 52, 52, 52, 
	52, 52, 52, 52, 52, 52, 52, 52, 
	52, 52, 52, 52, 52, 52, 52, 52, 
	52, 52, 52, 52, 52, 52, 52, 52, 
	52, 52, 52, 52, 52, 52, 52, 52, 
	52, 52, 52, 52, 52, 54, 52, 53, 
	52, 56, 56, 56, 56, 56, 56, 56, 
	56, 56, 56, 55, 55, 55, 55, 55, 
	55, 55, 56, 56, 56, 56, 56, 56, 
	55, 55, 55, 55, 55, 55, 55, 55, 
	55, 55, 55, 55, 55, 55, 55, 55, 
	55, 55, 55, 55, 55, 55, 55, 55, 
	55, 55, 56, 56, 56, 56, 56, 56, 
	55, 57, 45, 60, 59, 59, 59, 59, 
	59, 59, 59, 59, 59, 59, 59, 59, 
	59, 59, 59, 59, 59, 59, 59, 59, 
	59, 59, 59, 61, 59, 59, 59, 59, 
	59, 59, 59, 59, 59, 59, 59, 59, 
	59, 59, 59, 59, 59, 59, 59, 59, 
	59, 59, 59, 59, 59, 59, 59, 59, 
	59, 59, 59, 59, 59, 59, 59, 59, 
	59, 59, 59, 59, 59, 59, 59, 59, 
	59, 59, 59, 59, 59, 59, 59, 59, 
	59, 59, 59, 59, 59, 62, 59, 60, 
	59, 64, 63, 66, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	61, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 67, 65, 66, 
	65, 69, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 61, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 70, 68, 69, 68, 72, 
	71, 71, 71, 71, 71, 71, 71, 71, 
	73, 74, 71, 71, 71, 71, 71, 71, 
	71, 71, 71, 71, 71, 71, 71, 71, 
	71, 71, 71, 71, 71, 71, 71, 73, 
	71, 75, 71, 71, 76, 71, 77, 71, 
	71, 71, 71, 71, 71, 71, 78, 79, 
	79, 79, 79, 79, 79, 79, 79, 79, 
	79, 71, 71, 71, 71, 71, 71, 71, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 71, 71, 71, 71, 80, 71, 
	80, 80, 80, 81, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 71, 73, 74, 82, 82, 82, 
	82, 82, 82, 82, 82, 82, 82, 82, 
	82, 82, 82, 82, 82, 82, 82, 82, 
	82, 82, 73, 82, 2, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 3, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 4, 1, 
	84, 83, 86, 85, 6, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	5, 3, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 7, 5, 
	8, 83, 83, 83, 83, 12, 83, 79, 
	79, 79, 79, 79, 79, 79, 79, 79, 
	79, 87, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 88, 88, 88, 88, 
	88, 88, 88, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 88, 88, 88, 
	88, 80, 88, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 88, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	89, 89, 89, 89, 89, 89, 89, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 89, 89, 89, 89, 80, 89, 80, 
	80, 80, 80, 90, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 89, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 89, 89, 89, 89, 
	89, 89, 89, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 89, 89, 89, 
	89, 80, 89, 80, 80, 80, 80, 80, 
	91, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 89, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	89, 89, 89, 89, 89, 89, 89, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 89, 89, 89, 89, 80, 89, 80, 
	80, 80, 80, 80, 80, 80, 80, 92, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 89, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 89, 89, 89, 89, 
	89, 89, 89, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 89, 89, 89, 
	89, 80, 89, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	93, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 89, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	89, 89, 89, 89, 89, 89, 89, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 89, 89, 89, 89, 80, 89, 80, 
	80, 80, 80, 94, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 89, 96, 95, 95, 95, 95, 95, 
	95, 95, 95, 97, 98, 95, 95, 95, 
	95, 95, 95, 95, 95, 95, 95, 95, 
	95, 95, 95, 95, 95, 95, 95, 95, 
	95, 95, 97, 95, 99, 100, 95, 95, 
	95, 101, 102, 103, 103, 95, 102, 95, 
	95, 104, 105, 106, 106, 106, 106, 106, 
	106, 106, 106, 106, 107, 108, 95, 95, 
	95, 95, 95, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 95, 95, 95, 
	95, 109, 95, 109, 109, 109, 109, 109, 
	110, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 111, 95, 112, 
	95, 97, 98, 113, 113, 113, 113, 113, 
	113, 113, 113, 113, 113, 113, 113, 113, 
	113, 113, 113, 113, 113, 113, 113, 113, 
	97, 113, 16, 15, 15, 15, 15, 15, 
	15, 15, 15, 15, 15, 15, 15, 15, 
	15, 15, 15, 15, 15, 15, 15, 15, 
	15, 15, 17, 15, 15, 15, 15, 15, 
	15, 15, 15, 15, 15, 15, 15, 15, 
	15, 15, 15, 15, 15, 15, 15, 15, 
	15, 15, 15, 15, 15, 15, 15, 15, 
	15, 15, 15, 15, 15, 15, 15, 15, 
	15, 15, 15, 15, 15, 15, 15, 15, 
	15, 15, 15, 15, 15, 15, 15, 15, 
	15, 15, 15, 15, 18, 15, 20, 19, 
	22, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 17, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 21, 21, 
	21, 21, 23, 21, 25, 24, 24, 24, 
	24, 24, 24, 24, 24, 24, 24, 24, 
	24, 24, 24, 24, 24, 24, 24, 24, 
	24, 24, 24, 24, 24, 24, 24, 24, 
	24, 24, 24, 24, 24, 24, 24, 24, 
	24, 17, 24, 24, 24, 24, 24, 24, 
	24, 24, 24, 24, 24, 24, 24, 24, 
	24, 24, 24, 24, 24, 24, 24, 24, 
	24, 24, 24, 24, 24, 24, 24, 24, 
	24, 24, 24, 24, 24, 24, 24, 24, 
	24, 24, 24, 24, 24, 24, 26, 24, 
	106, 106, 106, 106, 106, 106, 106, 106, 
	106, 106, 115, 115, 115, 115, 115, 115, 
	115, 115, 115, 115, 115, 115, 115, 115, 
	115, 115, 115, 115, 115, 115, 115, 115, 
	115, 115, 115, 115, 115, 115, 115, 115, 
	115, 115, 115, 115, 115, 115, 115, 115, 
	115, 115, 115, 115, 115, 115, 115, 115, 
	115, 115, 115, 115, 115, 115, 115, 115, 
	115, 115, 115, 115, 115, 115, 115, 115, 
	116, 115, 106, 106, 106, 106, 106, 106, 
	106, 106, 106, 106, 115, 28, 28, 28, 
	28, 28, 28, 28, 28, 28, 28, 117, 
	117, 117, 117, 117, 117, 117, 28, 28, 
	28, 28, 28, 28, 117, 117, 117, 117, 
	117, 117, 117, 117, 117, 117, 117, 117, 
	117, 117, 117, 117, 117, 117, 117, 117, 
	117, 117, 117, 117, 117, 117, 28, 28, 
	28, 28, 28, 28, 117, 118, 114, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 119, 119, 119, 119, 119, 119, 119, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 119, 119, 119, 119, 109, 119, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 119, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 120, 120, 120, 
	120, 120, 120, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 120, 
	120, 120, 109, 120, 109, 121, 122, 109, 
	123, 109, 124, 125, 109, 109, 109, 109, 
	109, 126, 109, 127, 109, 128, 109, 129, 
	109, 109, 109, 109, 109, 109, 120, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 120, 120, 120, 120, 120, 120, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 120, 120, 120, 109, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 130, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 120, 120, 120, 
	120, 120, 120, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 120, 
	120, 120, 109, 120, 109, 109, 109, 109, 
	131, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 120, 120, 120, 120, 120, 120, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 120, 120, 120, 109, 120, 
	132, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 120, 120, 120, 
	120, 120, 120, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 120, 
	120, 120, 109, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 133, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 134, 134, 134, 134, 134, 134, 134, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 134, 134, 134, 134, 109, 134, 
	135, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 136, 109, 109, 109, 
	109, 109, 134, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 120, 120, 120, 
	120, 120, 120, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 120, 
	120, 120, 109, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 137, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 120, 120, 120, 120, 120, 120, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 120, 120, 120, 109, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 138, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 120, 120, 120, 
	120, 120, 120, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 120, 
	120, 120, 109, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 139, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 120, 120, 120, 120, 120, 120, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 120, 120, 120, 109, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 140, 109, 109, 109, 109, 109, 
	109, 109, 120, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 120, 120, 120, 
	120, 120, 120, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 120, 
	120, 120, 109, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 141, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 142, 109, 109, 120, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 120, 120, 120, 120, 120, 120, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 120, 120, 120, 109, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 143, 109, 109, 109, 109, 
	109, 109, 120, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 120, 120, 120, 
	120, 120, 120, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 120, 
	120, 120, 109, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 144, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 120, 120, 120, 120, 120, 120, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 120, 120, 120, 109, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	145, 109, 120, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 120, 120, 120, 
	120, 120, 120, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 120, 
	120, 120, 109, 120, 109, 109, 109, 109, 
	146, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 120, 120, 120, 120, 120, 120, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 120, 120, 120, 109, 120, 
	109, 109, 147, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 120, 120, 120, 
	120, 120, 120, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 120, 
	120, 120, 109, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 148, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 120, 120, 120, 120, 120, 120, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 120, 120, 120, 109, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 149, 109, 109, 109, 109, 
	109, 109, 120, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 120, 120, 120, 
	120, 120, 120, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 120, 
	120, 120, 109, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 150, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 120, 120, 120, 120, 120, 120, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 120, 120, 120, 109, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 151, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 120, 120, 120, 
	120, 120, 120, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 120, 
	120, 120, 109, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 152, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 120, 120, 120, 120, 120, 120, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 120, 120, 120, 109, 120, 
	109, 109, 109, 153, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 120, 120, 120, 
	120, 120, 120, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 120, 
	120, 120, 109, 120, 109, 109, 109, 109, 
	154, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 120, 120, 120, 120, 120, 120, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 120, 120, 120, 109, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 155, 
	109, 109, 120, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 120, 120, 120, 
	120, 120, 120, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 120, 
	120, 120, 109, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 156, 
	109, 109, 109, 109, 109, 109, 120, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 120, 120, 120, 120, 120, 120, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 120, 120, 120, 109, 120, 
	109, 109, 157, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 120, 120, 120, 
	120, 120, 120, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 120, 
	120, 120, 109, 120, 109, 109, 109, 109, 
	158, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 120, 120, 120, 120, 120, 120, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 120, 120, 120, 109, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 159, 109, 109, 109, 109, 
	109, 109, 120, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 120, 120, 120, 
	120, 120, 120, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 120, 
	120, 120, 109, 120, 160, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 120, 120, 120, 120, 120, 120, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 120, 120, 120, 109, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 161, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 120, 120, 120, 
	120, 120, 120, 120, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 120, 
	120, 120, 109, 120, 109, 109, 109, 109, 
	109, 109, 162, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 120, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 120, 120, 120, 120, 120, 120, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 120, 120, 120, 120, 109, 120, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 109, 109, 109, 109, 109, 109, 
	109, 109, 163, 109, 109, 109, 109, 109, 
	109, 109, 120, 165, 164, 164, 164, 164, 
	164, 164, 164, 164, 166, 167, 164, 164, 
	164, 164, 164, 164, 164, 164, 164, 164, 
	164, 164, 164, 164, 164, 164, 164, 164, 
	164, 164, 164, 166, 164, 168, 164, 164, 
	164, 164, 169, 170, 171, 171, 164, 170, 
	164, 164, 172, 173, 174, 174, 174, 174, 
	174, 174, 174, 174, 174, 175, 176, 164, 
	164, 164, 164, 164, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 164, 164, 
	164, 164, 177, 164, 177, 177, 177, 177, 
	177, 178, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 179, 164, 
	180, 164, 166, 167, 181, 181, 181, 181, 
	181, 181, 181, 181, 181, 181, 181, 181, 
	181, 181, 181, 181, 181, 181, 181, 181, 
	181, 166, 181, 31, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 32, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 33, 30, 35, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 32, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 34, 34, 34, 34, 34, 34, 34, 
	34, 36, 34, 37, 182, 182, 182, 182, 
	41, 182, 174, 174, 174, 174, 174, 174, 
	174, 174, 174, 174, 183, 183, 183, 183, 
	183, 183, 183, 183, 183, 183, 183, 183, 
	183, 183, 183, 183, 183, 183, 183, 183, 
	183, 183, 183, 183, 183, 183, 183, 183, 
	183, 183, 183, 183, 183, 183, 183, 183, 
	183, 183, 183, 183, 183, 183, 183, 183, 
	183, 183, 183, 183, 183, 183, 183, 183, 
	183, 183, 183, 183, 183, 183, 183, 183, 
	183, 183, 184, 183, 174, 174, 174, 174, 
	174, 174, 174, 174, 174, 174, 183, 44, 
	44, 44, 44, 44, 44, 44, 44, 44, 
	44, 185, 185, 185, 185, 185, 185, 185, 
	44, 44, 44, 44, 44, 44, 185, 185, 
	185, 185, 185, 185, 185, 185, 185, 185, 
	185, 185, 185, 185, 185, 185, 185, 185, 
	185, 185, 185, 185, 185, 185, 185, 185, 
	44, 44, 44, 44, 44, 44, 185, 186, 
	182, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 187, 187, 187, 187, 187, 
	187, 187, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 187, 187, 187, 187, 
	177, 187, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 187, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 188, 
	188, 188, 188, 188, 188, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 188, 188, 188, 177, 188, 177, 189, 
	190, 177, 191, 177, 192, 193, 177, 177, 
	177, 177, 177, 194, 177, 195, 177, 196, 
	177, 197, 177, 177, 177, 177, 177, 177, 
	188, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 188, 188, 188, 188, 188, 
	188, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 188, 188, 188, 
	177, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 198, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 188, 
	188, 188, 188, 188, 188, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 188, 188, 188, 177, 188, 177, 177, 
	177, 177, 199, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 188, 188, 188, 188, 188, 
	188, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 188, 188, 188, 
	177, 188, 200, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 188, 
	188, 188, 188, 188, 188, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 188, 188, 188, 177, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	201, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 202, 202, 202, 202, 202, 
	202, 202, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 202, 202, 202, 202, 
	177, 202, 203, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 204, 177, 
	177, 177, 177, 177, 202, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 188, 
	188, 188, 188, 188, 188, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 188, 188, 188, 177, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 205, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 188, 188, 188, 188, 188, 
	188, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 188, 188, 188, 
	177, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 206, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 188, 
	188, 188, 188, 188, 188, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 188, 188, 188, 177, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 207, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 188, 188, 188, 188, 188, 
	188, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 188, 188, 188, 
	177, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 208, 177, 177, 177, 
	177, 177, 177, 177, 188, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 188, 
	188, 188, 188, 188, 188, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 188, 188, 188, 177, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 209, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 210, 177, 177, 
	188, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 188, 188, 188, 188, 188, 
	188, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 188, 188, 188, 
	177, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 211, 177, 177, 
	177, 177, 177, 177, 188, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 188, 
	188, 188, 188, 188, 188, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 188, 188, 188, 177, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 212, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 188, 188, 188, 188, 188, 
	188, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 188, 188, 188, 
	177, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 213, 177, 188, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 188, 
	188, 188, 188, 188, 188, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 188, 188, 188, 177, 188, 177, 177, 
	177, 177, 214, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 188, 188, 188, 188, 188, 
	188, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 188, 188, 188, 
	177, 188, 177, 177, 215, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 188, 
	188, 188, 188, 188, 188, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 188, 188, 188, 177, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 216, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 188, 188, 188, 188, 188, 
	188, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 188, 188, 188, 
	177, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 217, 177, 177, 
	177, 177, 177, 177, 188, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 188, 
	188, 188, 188, 188, 188, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 188, 188, 188, 177, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 218, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 188, 188, 188, 188, 188, 
	188, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 188, 188, 188, 
	177, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	219, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 188, 
	188, 188, 188, 188, 188, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 188, 188, 188, 177, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 220, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 188, 188, 188, 188, 188, 
	188, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 188, 188, 188, 
	177, 188, 177, 177, 177, 221, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 188, 
	188, 188, 188, 188, 188, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 188, 188, 188, 177, 188, 177, 177, 
	177, 177, 222, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 188, 188, 188, 188, 188, 
	188, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 188, 188, 188, 
	177, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 223, 177, 177, 188, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 188, 
	188, 188, 188, 188, 188, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 188, 188, 188, 177, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 224, 177, 177, 177, 177, 177, 177, 
	188, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 188, 188, 188, 188, 188, 
	188, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 188, 188, 188, 
	177, 188, 177, 177, 225, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 188, 
	188, 188, 188, 188, 188, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 188, 188, 188, 177, 188, 177, 177, 
	177, 177, 226, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 188, 188, 188, 188, 188, 
	188, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 188, 188, 188, 
	177, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 227, 177, 177, 
	177, 177, 177, 177, 188, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 188, 
	188, 188, 188, 188, 188, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 188, 188, 188, 177, 188, 228, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 188, 188, 188, 188, 188, 
	188, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 188, 188, 188, 
	177, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 229, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 188, 
	188, 188, 188, 188, 188, 188, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 188, 188, 188, 177, 188, 177, 177, 
	177, 177, 177, 177, 230, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	188, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 188, 188, 188, 188, 188, 
	188, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 188, 188, 188, 188, 
	177, 188, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 177, 177, 177, 177, 
	177, 177, 177, 177, 231, 177, 177, 177, 
	177, 177, 177, 177, 188, 233, 232, 232, 
	232, 232, 232, 232, 232, 232, 232, 232, 
	232, 232, 232, 232, 232, 232, 232, 232, 
	232, 232, 232, 232, 232, 232, 232, 232, 
	232, 232, 232, 232, 232, 232, 232, 232, 
	232, 232, 232, 232, 232, 232, 232, 232, 
	232, 232, 234, 232, 232, 232, 232, 232, 
	232, 232, 232, 232, 232, 232, 232, 232, 
	232, 232, 232, 232, 232, 232, 232, 232, 
	232, 232, 232, 232, 232, 232, 232, 232, 
	232, 232, 232, 232, 232, 232, 232, 232, 
	232, 232, 232, 232, 232, 232, 232, 232, 
	232, 235, 236, 232, 239, 238, 238, 238, 
	238, 238, 238, 238, 238, 238, 238, 238, 
	238, 238, 238, 238, 238, 238, 238, 238, 
	238, 238, 238, 238, 238, 238, 238, 238, 
	238, 238, 238, 238, 238, 238, 238, 238, 
	238, 238, 240, 238, 238, 238, 238, 238, 
	238, 238, 238, 238, 238, 238, 238, 238, 
	238, 238, 238, 238, 238, 238, 238, 238, 
	238, 238, 238, 238, 238, 238, 238, 238, 
	238, 238, 238, 238, 238, 238, 238, 238, 
	238, 238, 238, 238, 238, 238, 238, 238, 
	238, 238, 238, 241, 242, 238, 238, 238, 
	243, 238, 238, 238, 238, 238, 238, 238, 
	244, 238, 238, 238, 245, 238, 246, 238, 
	247, 238, 249, 248, 248, 248, 248, 248, 
	248, 248, 248, 248, 248, 248, 248, 248, 
	248, 248, 248, 248, 248, 248, 248, 248, 
	248, 248, 248, 248, 248, 248, 248, 248, 
	248, 248, 248, 248, 248, 248, 248, 248, 
	248, 248, 248, 248, 250, 248, 248, 248, 
	251, 252, 248, 248, 248, 248, 248, 248, 
	248, 248, 248, 248, 248, 248, 248, 248, 
	248, 248, 248, 248, 248, 248, 248, 248, 
	248, 248, 248, 248, 248, 248, 248, 248, 
	248, 248, 248, 248, 248, 248, 248, 248, 
	248, 248, 248, 248, 248, 253, 254, 248, 
	256, 255, 258, 257, 261, 260, 260, 260, 
	260, 260, 260, 260, 260, 260, 260, 260, 
	260, 260, 260, 260, 260, 260, 260, 260, 
	260, 260, 260, 260, 260, 260, 260, 260, 
	260, 260, 260, 260, 260, 260, 260, 260, 
	260, 260, 262, 260, 260, 260, 260, 260, 
	260, 260, 260, 260, 260, 260, 260, 260, 
	260, 260, 260, 260, 260, 260, 260, 260, 
	260, 260, 260, 260, 260, 260, 260, 260, 
	260, 260, 260, 260, 260, 260, 260, 260, 
	260, 260, 260, 260, 260, 260, 260, 260, 
	260, 260, 260, 263, 264, 260, 260, 260, 
	265, 260, 260, 260, 260, 260, 260, 260, 
	266, 260, 260, 260, 267, 260, 268, 260, 
	269, 260, 270, 271, 271, 271, 271, 271, 
	271, 271, 271, 272, 272, 271, 271, 271, 
	271, 271, 271, 271, 271, 271, 271, 271, 
	271, 271, 271, 271, 271, 271, 271, 271, 
	271, 271, 272, 271, 271, 271, 271, 271, 
	271, 271, 271, 271, 271, 271, 271, 271, 
	271, 271, 271, 271, 271, 271, 271, 271, 
	271, 271, 271, 271, 271, 273, 271, 271, 
	271, 271, 271, 274, 274, 274, 274, 274, 
	274, 274, 274, 274, 274, 274, 274, 274, 
	274, 274, 274, 274, 274, 274, 274, 274, 
	274, 274, 274, 274, 274, 271, 271, 271, 
	271, 274, 271, 274, 274, 274, 274, 274, 
	274, 274, 274, 274, 274, 274, 274, 274, 
	274, 274, 274, 274, 274, 274, 274, 274, 
	274, 274, 274, 274, 274, 271, 272, 272, 
	275, 275, 275, 275, 275, 275, 275, 275, 
	275, 275, 275, 275, 275, 275, 275, 275, 
	275, 275, 275, 275, 275, 272, 275, 274, 
	274, 274, 274, 274, 274, 274, 274, 274, 
	274, 276, 276, 276, 276, 276, 276, 276, 
	274, 274, 274, 274, 274, 274, 274, 274, 
	274, 274, 274, 274, 274, 274, 274, 274, 
	274, 274, 274, 274, 274, 274, 274, 274, 
	274, 274, 276, 276, 276, 276, 274, 276, 
	274, 274, 274, 274, 274, 274, 274, 274, 
	274, 274, 274, 274, 274, 274, 274, 274, 
	274, 274, 274, 274, 274, 274, 274, 274, 
	274, 274, 276, 278, 277, 277, 277, 277, 
	277, 277, 277, 277, 279, 280, 277, 277, 
	279, 277, 277, 277, 277, 277, 277, 277, 
	277, 277, 277, 277, 277, 277, 277, 277, 
	277, 277, 277, 279, 277, 281, 282, 283, 
	284, 277, 285, 277, 277, 286, 277, 277, 
	287, 288, 289, 290, 291, 291, 291, 291, 
	291, 291, 291, 291, 291, 292, 277, 293, 
	294, 295, 277, 296, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 298, 277, 
	277, 277, 297, 277, 299, 297, 297, 297, 
	300, 301, 302, 297, 303, 297, 297, 304, 
	305, 297, 306, 307, 297, 297, 297, 308, 
	297, 309, 310, 297, 297, 297, 311, 312, 
	313, 277, 279, 314, 314, 314, 279, 314, 
	314, 314, 314, 314, 314, 314, 314, 314, 
	314, 314, 314, 314, 314, 314, 314, 314, 
	314, 279, 314, 47, 46, 46, 46, 46, 
	46, 46, 46, 46, 46, 46, 46, 46, 
	46, 46, 46, 46, 46, 46, 46, 46, 
	46, 46, 46, 48, 46, 46, 46, 46, 
	46, 46, 46, 46, 46, 46, 46, 46, 
	46, 46, 46, 46, 46, 46, 46, 46, 
	46, 46, 46, 46, 46, 46, 46, 46, 
	46, 46, 46, 46, 46, 46, 46, 46, 
	46, 46, 46, 46, 46, 46, 46, 46, 
	46, 46, 46, 46, 46, 46, 46, 46, 
	46, 46, 46, 46, 46, 49, 46, 317, 
	316, 51, 50, 318, 315, 315, 315, 315, 
	315, 315, 315, 315, 319, 315, 315, 315, 
	315, 320, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 321, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	322, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	323, 315, 324, 315, 315, 315, 315, 315, 
	315, 315, 315, 325, 315, 315, 315, 315, 
	326, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	327, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 328, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 329, 
	315, 53, 52, 52, 52, 52, 52, 52, 
	52, 52, 52, 52, 52, 52, 52, 52, 
	52, 52, 52, 52, 52, 52, 52, 52, 
	52, 52, 52, 52, 52, 52, 48, 52, 
	52, 52, 52, 52, 52, 52, 52, 52, 
	52, 52, 52, 52, 52, 52, 52, 52, 
	52, 52, 52, 52, 52, 52, 52, 52, 
	52, 52, 52, 52, 52, 52, 52, 52, 
	52, 52, 52, 52, 52, 52, 52, 52, 
	52, 52, 52, 52, 52, 52, 52, 52, 
	52, 52, 52, 54, 52, 330, 315, 331, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	332, 315, 333, 315, 291, 291, 291, 291, 
	291, 291, 291, 291, 291, 291, 334, 334, 
	334, 334, 334, 334, 334, 334, 334, 334, 
	334, 334, 334, 334, 334, 334, 334, 334, 
	334, 334, 334, 334, 334, 334, 334, 334, 
	334, 334, 334, 334, 334, 334, 334, 334, 
	334, 334, 334, 334, 334, 334, 334, 334, 
	334, 334, 334, 334, 334, 334, 334, 334, 
	334, 334, 334, 334, 334, 334, 334, 334, 
	334, 334, 334, 334, 335, 334, 291, 291, 
	291, 291, 291, 291, 291, 291, 291, 291, 
	334, 56, 56, 56, 56, 56, 56, 56, 
	56, 56, 56, 336, 336, 336, 336, 336, 
	336, 336, 56, 56, 56, 56, 56, 56, 
	336, 336, 336, 336, 336, 336, 336, 336, 
	336, 336, 336, 336, 336, 336, 336, 336, 
	336, 336, 336, 336, 336, 336, 336, 336, 
	336, 336, 56, 56, 56, 56, 56, 56, 
	336, 337, 315, 315, 338, 339, 315, 341, 
	340, 342, 315, 315, 315, 315, 315, 315, 
	315, 315, 343, 315, 315, 315, 315, 344, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 345, 315, 315, 315, 346, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 347, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 348, 315, 
	350, 349, 349, 349, 349, 349, 349, 349, 
	349, 351, 349, 349, 349, 349, 352, 349, 
	349, 349, 349, 349, 349, 349, 349, 349, 
	349, 349, 349, 349, 349, 349, 349, 349, 
	349, 349, 349, 349, 349, 349, 349, 349, 
	349, 349, 349, 349, 349, 349, 349, 349, 
	349, 349, 349, 349, 349, 349, 349, 349, 
	349, 349, 349, 349, 349, 353, 349, 349, 
	349, 349, 349, 349, 349, 349, 349, 349, 
	349, 349, 349, 349, 349, 349, 349, 349, 
	349, 349, 349, 349, 349, 349, 349, 349, 
	349, 349, 349, 349, 349, 354, 349, 355, 
	315, 356, 315, 315, 315, 315, 315, 315, 
	315, 315, 357, 315, 315, 315, 315, 358, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 359, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 360, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 361, 315, 
	362, 315, 315, 315, 315, 315, 315, 315, 
	315, 363, 315, 315, 315, 315, 364, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 365, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 315, 315, 315, 
	315, 315, 315, 315, 315, 366, 315, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 367, 367, 367, 367, 367, 367, 367, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 367, 367, 367, 367, 297, 367, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 367, 369, 368, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	371, 297, 297, 297, 297, 297, 297, 297, 
	297, 372, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 373, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 374, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 375, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 376, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	377, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 378, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 379, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 380, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 381, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 382, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 383, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 384, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 385, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 386, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 387, 297, 297, 388, 
	297, 297, 297, 297, 297, 389, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 390, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 391, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 392, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 393, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 394, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 395, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 396, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 397, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 398, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 399, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 400, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	401, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 402, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 403, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 404, 405, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 406, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	407, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 408, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 409, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	410, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 411, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 412, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 413, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 414, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 415, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 416, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 417, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 418, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 419, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 420, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 421, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 422, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	423, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 424, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 425, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 426, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 427, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 428, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 429, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	430, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 431, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 432, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 433, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	434, 297, 297, 435, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	436, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 437, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 438, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	439, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 440, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 441, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 442, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 443, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	444, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 445, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 446, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 447, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 448, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 449, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 450, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 451, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 452, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 453, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 454, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 455, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 456, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 457, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 458, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 370, 
	370, 370, 370, 370, 370, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	370, 370, 370, 370, 297, 370, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 459, 297, 297, 297, 297, 297, 297, 
	370, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 370, 370, 370, 370, 370, 
	370, 370, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 370, 370, 370, 
	297, 370, 297, 297, 297, 297, 460, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 297, 297, 297, 297, 
	297, 297, 297, 297, 370, 461, 315, 462, 
	315, 464, 463, 463, 463, 463, 463, 463, 
	463, 463, 465, 466, 463, 463, 463, 463, 
	463, 463, 463, 463, 463, 463, 463, 463, 
	463, 463, 463, 463, 463, 463, 463, 463, 
	463, 465, 463, 467, 468, 463, 469, 463, 
	470, 463, 463, 463, 463, 463, 463, 463, 
	471, 472, 472, 472, 472, 472, 472, 472, 
	472, 472, 472, 463, 463, 463, 463, 463, 
	463, 463, 473, 473, 473, 473, 473, 473, 
	473, 473, 473, 473, 473, 473, 473, 473, 
	473, 473, 473, 473, 473, 473, 473, 473, 
	473, 473, 473, 473, 463, 463, 463, 463, 
	473, 463, 473, 473, 473, 473, 473, 473, 
	473, 473, 473, 473, 473, 473, 473, 473, 
	473, 473, 473, 473, 473, 473, 473, 473, 
	473, 473, 473, 473, 463, 465, 466, 474, 
	474, 474, 474, 474, 474, 474, 474, 474, 
	474, 474, 474, 474, 474, 474, 474, 474, 
	474, 474, 474, 474, 465, 474, 60, 59, 
	59, 59, 59, 59, 59, 59, 59, 59, 
	59, 59, 59, 59, 59, 59, 59, 59, 
	59, 59, 59, 59, 59, 59, 61, 59, 
	59, 59, 59, 59, 59, 59, 59, 59, 
	59, 59, 59, 59, 59, 59, 59, 59, 
	59, 59, 59, 59, 59, 59, 59, 59, 
	59, 59, 59, 59, 59, 59, 59, 59, 
	59, 59, 59, 59, 59, 59, 59, 59, 
	59, 59, 59, 59, 59, 59, 59, 59, 
	59, 59, 59, 59, 59, 59, 59, 59, 
	62, 59, 64, 63, 476, 475, 478, 477, 
	66, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 61, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 65, 65, 65, 65, 65, 65, 
	65, 65, 67, 65, 69, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 61, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 68, 68, 
	68, 68, 68, 68, 68, 68, 70, 68, 
	472, 472, 472, 472, 472, 472, 472, 472, 
	472, 472, 479, 473, 473, 473, 473, 473, 
	473, 473, 473, 473, 473, 480, 480, 480, 
	480, 480, 480, 480, 473, 473, 473, 473, 
	473, 473, 473, 473, 473, 473, 473, 473, 
	473, 473, 473, 473, 473, 473, 473, 473, 
	473, 473, 473, 473, 473, 473, 480, 480, 
	480, 480, 473, 480, 473, 473, 473, 473, 
	473, 473, 473, 473, 473, 473, 473, 473, 
	473, 473, 473, 473, 473, 473, 473, 473, 
	473, 473, 473, 473, 473, 473, 480, 0
};

static const short _rlscan_trans_targs[] = {
	38, 1, 1, 38, 2, 3, 3, 4, 
	5, 5, 6, 38, 7, 38, 52, 8, 
	8, 52, 9, 10, 52, 11, 11, 12, 
	13, 13, 14, 52, 60, 95, 16, 16, 
	95, 17, 18, 18, 19, 20, 20, 21, 
	95, 22, 95, 95, 102, 146, 24, 24, 
	149, 25, 26, 146, 27, 27, 28, 146, 
	159, 146, 253, 31, 31, 253, 32, 33, 
	253, 34, 34, 35, 36, 36, 37, 38, 
	38, 39, 39, 40, 41, 43, 44, 45, 
	46, 47, 38, 38, 42, 38, 38, 38, 
	38, 38, 48, 49, 50, 51, 46, 52, 
	52, 53, 53, 54, 55, 56, 52, 52, 
	57, 58, 59, 61, 52, 62, 63, 52, 
	52, 52, 52, 52, 15, 52, 52, 52, 
	52, 64, 68, 73, 79, 82, 85, 88, 
	89, 91, 65, 66, 67, 62, 52, 69, 
	71, 70, 62, 72, 62, 74, 77, 75, 
	76, 62, 78, 62, 80, 81, 62, 83, 
	84, 62, 86, 87, 62, 62, 90, 62, 
	92, 93, 94, 62, 95, 95, 96, 96, 
	97, 98, 95, 95, 99, 100, 101, 103, 
	95, 104, 105, 95, 95, 95, 95, 95, 
	23, 95, 95, 95, 95, 106, 110, 115, 
	121, 124, 127, 130, 131, 133, 107, 108, 
	109, 104, 95, 111, 113, 112, 104, 114, 
	104, 116, 119, 117, 118, 104, 120, 104, 
	122, 123, 104, 125, 126, 104, 128, 129, 
	104, 104, 132, 104, 134, 135, 136, 104, 
	137, 137, 137, 138, 137, 137, 137, 137, 
	137, 137, 137, 137, 137, 137, 137, 137, 
	139, 139, 139, 139, 140, 141, 142, 139, 
	139, 139, 139, 139, 139, 139, 139, 139, 
	139, 139, 139, 139, 139, 139, 143, 0, 
	144, 143, 145, 143, 143, 146, 146, 147, 
	146, 148, 150, 151, 152, 153, 154, 155, 
	156, 146, 157, 158, 160, 162, 164, 165, 
	166, 167, 168, 169, 183, 190, 193, 198, 
	211, 214, 220, 226, 237, 238, 245, 146, 
	251, 252, 146, 146, 146, 146, 146, 146, 
	146, 146, 146, 146, 146, 146, 146, 146, 
	146, 146, 146, 146, 146, 146, 146, 29, 
	146, 146, 146, 161, 146, 146, 146, 146, 
	146, 146, 163, 146, 146, 146, 146, 146, 
	146, 146, 146, 146, 146, 146, 146, 146, 
	146, 146, 146, 146, 146, 146, 146, 146, 
	146, 146, 146, 170, 177, 171, 174, 172, 
	173, 167, 175, 176, 167, 178, 179, 180, 
	181, 182, 167, 184, 185, 186, 167, 167, 
	187, 188, 189, 167, 191, 192, 167, 194, 
	195, 196, 197, 167, 199, 203, 200, 201, 
	202, 167, 204, 208, 205, 206, 207, 167, 
	209, 210, 167, 212, 213, 167, 215, 216, 
	217, 218, 219, 167, 221, 222, 223, 224, 
	225, 167, 227, 232, 228, 229, 230, 231, 
	167, 233, 234, 235, 236, 167, 167, 239, 
	240, 241, 242, 243, 244, 167, 246, 248, 
	247, 167, 249, 250, 167, 146, 30, 253, 
	253, 254, 254, 255, 256, 257, 259, 260, 
	261, 262, 253, 253, 258, 253, 253, 253, 
	253
};

static const unsigned char _rlscan_trans_actions[] = {
	1, 0, 2, 3, 0, 0, 2, 0, 
	0, 2, 0, 4, 0, 5, 6, 0, 
	2, 7, 0, 0, 8, 0, 2, 0, 
	0, 2, 0, 9, 0, 10, 0, 2, 
	11, 0, 0, 2, 0, 0, 2, 0, 
	12, 0, 13, 14, 0, 15, 0, 2, 
	0, 0, 0, 16, 0, 2, 0, 17, 
	0, 18, 19, 0, 2, 20, 0, 0, 
	21, 0, 2, 0, 0, 2, 0, 24, 
	25, 0, 2, 26, 0, 26, 26, 0, 
	27, 0, 28, 29, 0, 30, 31, 32, 
	33, 34, 0, 0, 0, 0, 35, 36, 
	37, 0, 2, 26, 26, 26, 38, 39, 
	26, 26, 0, 0, 40, 41, 0, 42, 
	43, 44, 45, 46, 0, 47, 48, 49, 
	50, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 51, 52, 0, 
	0, 0, 53, 0, 54, 0, 0, 0, 
	0, 55, 0, 56, 0, 0, 57, 0, 
	0, 58, 0, 0, 59, 60, 0, 61, 
	0, 0, 0, 62, 63, 64, 0, 2, 
	26, 26, 65, 66, 26, 26, 0, 0, 
	67, 68, 0, 69, 70, 71, 72, 73, 
	0, 74, 75, 76, 77, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 78, 79, 0, 0, 0, 80, 0, 
	81, 0, 0, 0, 0, 82, 0, 83, 
	0, 0, 84, 0, 0, 85, 0, 0, 
	86, 87, 0, 88, 0, 0, 0, 89, 
	90, 91, 92, 0, 93, 94, 95, 96, 
	97, 98, 99, 100, 101, 102, 103, 104, 
	105, 106, 107, 108, 0, 0, 0, 109, 
	110, 111, 112, 113, 114, 115, 116, 117, 
	118, 119, 120, 121, 122, 123, 124, 0, 
	0, 125, 0, 126, 127, 128, 129, 0, 
	130, 26, 26, 0, 0, 26, 0, 0, 
	0, 131, 26, 0, 0, 0, 0, 0, 
	0, 132, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 133, 
	0, 26, 134, 135, 136, 137, 138, 139, 
	140, 141, 142, 143, 144, 145, 146, 147, 
	148, 149, 150, 151, 152, 153, 154, 0, 
	155, 156, 157, 0, 158, 159, 160, 161, 
	162, 163, 0, 164, 165, 166, 167, 168, 
	169, 170, 171, 172, 173, 174, 175, 176, 
	177, 178, 179, 180, 181, 182, 183, 184, 
	185, 186, 187, 0, 0, 0, 0, 0, 
	0, 188, 0, 0, 189, 0, 0, 0, 
	0, 0, 190, 0, 0, 0, 191, 192, 
	0, 0, 0, 193, 0, 0, 194, 0, 
	0, 0, 0, 195, 0, 0, 0, 0, 
	0, 196, 0, 0, 0, 0, 0, 197, 
	0, 0, 198, 0, 0, 199, 0, 0, 
	0, 0, 0, 200, 0, 0, 0, 0, 
	0, 201, 0, 0, 0, 0, 0, 0, 
	202, 0, 0, 0, 0, 203, 204, 0, 
	0, 0, 0, 0, 0, 205, 0, 0, 
	0, 206, 0, 0, 207, 208, 0, 209, 
	210, 0, 2, 26, 26, 0, 26, 26, 
	0, 0, 211, 212, 0, 213, 214, 215, 
	216
};

static const short _rlscan_to_state_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 22, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 22, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 22, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 22, 0, 22, 0, 0, 0, 22, 
	0, 0, 22, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 22, 0, 0, 
	0, 0, 0, 0, 0, 0, 0
};

static const short _rlscan_from_state_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 23, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 23, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 23, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 23, 0, 23, 0, 0, 0, 23, 
	0, 0, 23, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 23, 0, 0, 
	0, 0, 0, 0, 0, 0, 0
};

static const short _rlscan_eof_trans[] = {
	0, 1, 1, 1, 1, 1, 1, 1, 
	15, 15, 15, 15, 15, 15, 15, 28, 
	30, 30, 30, 30, 30, 30, 30, 44, 
	46, 46, 46, 46, 46, 56, 46, 59, 
	59, 59, 59, 59, 59, 59, 0, 83, 
	84, 84, 86, 84, 84, 88, 89, 90, 
	90, 90, 90, 90, 0, 114, 115, 115, 
	115, 115, 116, 116, 118, 115, 120, 121, 
	121, 121, 121, 121, 135, 121, 121, 121, 
	121, 121, 121, 121, 121, 121, 121, 121, 
	121, 121, 121, 121, 121, 121, 121, 121, 
	121, 121, 121, 121, 121, 121, 121, 0, 
	182, 183, 183, 183, 184, 184, 186, 183, 
	188, 189, 189, 189, 189, 189, 203, 189, 
	189, 189, 189, 189, 189, 189, 189, 189, 
	189, 189, 189, 189, 189, 189, 189, 189, 
	189, 189, 189, 189, 189, 189, 189, 189, 
	189, 0, 238, 0, 256, 258, 260, 0, 
	276, 277, 0, 315, 316, 317, 316, 316, 
	316, 316, 316, 316, 316, 335, 335, 337, 
	316, 341, 316, 350, 316, 316, 316, 368, 
	369, 371, 371, 371, 371, 371, 371, 371, 
	371, 371, 371, 371, 371, 371, 371, 371, 
	371, 371, 371, 371, 371, 371, 371, 371, 
	371, 371, 371, 371, 371, 371, 371, 371, 
	371, 371, 371, 371, 371, 371, 371, 371, 
	371, 371, 371, 371, 371, 371, 371, 371, 
	371, 371, 371, 371, 371, 371, 371, 371, 
	371, 371, 371, 371, 371, 371, 371, 371, 
	371, 371, 371, 371, 371, 371, 371, 371, 
	371, 371, 371, 371, 371, 371, 371, 371, 
	371, 371, 371, 316, 316, 0, 475, 476, 
	476, 476, 478, 476, 476, 480, 481
};

enum {rlscan_start = 38};
enum {rlscan_first_final = 38};
enum {rlscan_error = 0};

enum {rlscan_en_inline_code_ruby = 52};
enum {rlscan_en_inline_code = 95};
enum {rlscan_en_or_literal = 137};
enum {rlscan_en_ragel_re_literal = 139};
enum {rlscan_en_write_statement = 143};
enum {rlscan_en_parser_def = 146};
enum {rlscan_en_main_ruby = 253};
enum {rlscan_en_main = 38};


#line 1172 "./rlscan.rl"

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
	
#line 3360 "rlscan.fast_build.cpp"
	{
	cs = rlscan_start;
	top = 0;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 1192 "./rlscan.rl"

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

		
#line 3415 "rlscan.fast_build.cpp"
	{
	int _slen;
	int _trans;
	const char *_keys;
	const short *_inds;
	if ( p == pe )
		goto _test_eof;
	if ( cs == 0 )
		goto _out;
_resume:
	switch ( _rlscan_from_state_actions[cs] ) {
	case 23:
#line 1 "NONE"
	{ts = p;}
	break;
#line 3431 "rlscan.fast_build.cpp"
	}

	_keys = _rlscan_trans_keys + (cs<<1);
	_inds = _rlscan_indicies + _rlscan_index_offsets[cs];

	_slen = _rlscan_key_spans[cs];
	_trans = _inds[ _slen > 0 && _keys[0] <=(*p) &&
		(*p) <= _keys[1] ?
		(*p) - _keys[0] : _slen ];

_eof_trans:
	cs = _rlscan_trans_targs[_trans];

	if ( _rlscan_trans_actions[_trans] == 0 )
		goto _again;

	switch ( _rlscan_trans_actions[_trans] ) {
	case 2:
#line 637 "./rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
	break;
	case 26:
#line 1 "NONE"
	{te = p+1;}
	break;
	case 7:
#line 712 "./rlscan.rl"
	{te = p+1;{ token( IL_Literal, ts, te ); }}
	break;
	case 48:
#line 721 "./rlscan.rl"
	{te = p+1;{ token( TK_NameSep, ts, te ); }}
	break;
	case 40:
#line 729 "./rlscan.rl"
	{te = p+1;{
			whitespaceOn = true;
			token( *ts, ts, te );
			if ( inlineBlockType == SemiTerminated )
				{cs = stack[--top];goto _again;}
		}}
	break;
	case 39:
#line 736 "./rlscan.rl"
	{te = p+1;{ 
			whitespaceOn = true;
			token( *ts, ts, te );
		}}
	break;
	case 38:
#line 741 "./rlscan.rl"
	{te = p+1;{ token( *ts, ts, te ); }}
	break;
	case 42:
#line 743 "./rlscan.rl"
	{te = p+1;{ 
			token( IL_Symbol, ts, te );
			curly_count += 1; 
		}}
	break;
	case 43:
#line 748 "./rlscan.rl"
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
	break;
	case 37:
#line 761 "./rlscan.rl"
	{te = p+1;{
			scan_error() << "unterminated code block" << endl;
		}}
	break;
	case 36:
#line 766 "./rlscan.rl"
	{te = p+1;{ token( IL_Symbol, ts, te ); }}
	break;
	case 52:
#line 671 "./rlscan.rl"
	{te = p;p--;{ token( KW_Char ); }}
	break;
	case 50:
#line 706 "./rlscan.rl"
	{te = p;p--;{ token( TK_Word, ts, te ); }}
	break;
	case 46:
#line 708 "./rlscan.rl"
	{te = p;p--;{ token( TK_UInt, ts, te ); }}
	break;
	case 47:
#line 709 "./rlscan.rl"
	{te = p;p--;{ token( TK_Hex, ts, te ); }}
	break;
	case 44:
#line 714 "./rlscan.rl"
	{te = p;p--;{ 
			if ( whitespaceOn ) 
				token( IL_WhiteSpace, ts, te );
		}}
	break;
	case 45:
#line 766 "./rlscan.rl"
	{te = p;p--;{ token( IL_Symbol, ts, te ); }}
	break;
	case 9:
#line 708 "./rlscan.rl"
	{{p = ((te))-1;}{ token( TK_UInt, ts, te ); }}
	break;
	case 6:
#line 766 "./rlscan.rl"
	{{p = ((te))-1;}{ token( IL_Symbol, ts, te ); }}
	break;
	case 49:
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
	break;
	case 11:
#line 815 "./rlscan.rl"
	{te = p+1;{ token( IL_Literal, ts, te ); }}
	break;
	case 12:
#line 822 "./rlscan.rl"
	{te = p+1;{ token( IL_Comment, ts, te ); }}
	break;
	case 75:
#line 824 "./rlscan.rl"
	{te = p+1;{ token( TK_NameSep, ts, te ); }}
	break;
	case 67:
#line 832 "./rlscan.rl"
	{te = p+1;{
			whitespaceOn = true;
			token( *ts, ts, te );
			if ( inlineBlockType == SemiTerminated )
				{cs = stack[--top];goto _again;}
		}}
	break;
	case 66:
#line 839 "./rlscan.rl"
	{te = p+1;{ 
			whitespaceOn = true;
			token( *ts, ts, te );
		}}
	break;
	case 65:
#line 844 "./rlscan.rl"
	{te = p+1;{ token( *ts, ts, te ); }}
	break;
	case 69:
#line 846 "./rlscan.rl"
	{te = p+1;{ 
			token( IL_Symbol, ts, te );
			curly_count += 1; 
		}}
	break;
	case 70:
#line 851 "./rlscan.rl"
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
	break;
	case 64:
#line 864 "./rlscan.rl"
	{te = p+1;{
			scan_error() << "unterminated code block" << endl;
		}}
	break;
	case 63:
#line 869 "./rlscan.rl"
	{te = p+1;{ token( IL_Symbol, ts, te ); }}
	break;
	case 79:
#line 774 "./rlscan.rl"
	{te = p;p--;{ token( KW_Char ); }}
	break;
	case 77:
#line 809 "./rlscan.rl"
	{te = p;p--;{ token( TK_Word, ts, te ); }}
	break;
	case 73:
#line 811 "./rlscan.rl"
	{te = p;p--;{ token( TK_UInt, ts, te ); }}
	break;
	case 74:
#line 812 "./rlscan.rl"
	{te = p;p--;{ token( TK_Hex, ts, te ); }}
	break;
	case 71:
#line 817 "./rlscan.rl"
	{te = p;p--;{ 
			if ( whitespaceOn ) 
				token( IL_WhiteSpace, ts, te );
		}}
	break;
	case 72:
#line 869 "./rlscan.rl"
	{te = p;p--;{ token( IL_Symbol, ts, te ); }}
	break;
	case 14:
#line 811 "./rlscan.rl"
	{{p = ((te))-1;}{ token( TK_UInt, ts, te ); }}
	break;
	case 10:
#line 869 "./rlscan.rl"
	{{p = ((te))-1;}{ token( IL_Symbol, ts, te ); }}
	break;
	case 76:
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
	break;
	case 97:
#line 874 "./rlscan.rl"
	{te = p+1;{ token( RE_Char, '\0' ); }}
	break;
	case 98:
#line 875 "./rlscan.rl"
	{te = p+1;{ token( RE_Char, '\a' ); }}
	break;
	case 99:
#line 876 "./rlscan.rl"
	{te = p+1;{ token( RE_Char, '\b' ); }}
	break;
	case 103:
#line 877 "./rlscan.rl"
	{te = p+1;{ token( RE_Char, '\t' ); }}
	break;
	case 101:
#line 878 "./rlscan.rl"
	{te = p+1;{ token( RE_Char, '\n' ); }}
	break;
	case 104:
#line 879 "./rlscan.rl"
	{te = p+1;{ token( RE_Char, '\v' ); }}
	break;
	case 100:
#line 880 "./rlscan.rl"
	{te = p+1;{ token( RE_Char, '\f' ); }}
	break;
	case 102:
#line 881 "./rlscan.rl"
	{te = p+1;{ token( RE_Char, '\r' ); }}
	break;
	case 96:
#line 882 "./rlscan.rl"
	{te = p+1;{ updateCol(); }}
	break;
	case 95:
#line 883 "./rlscan.rl"
	{te = p+1;{ token( RE_Char, ts+1, te ); }}
	break;
	case 92:
#line 886 "./rlscan.rl"
	{te = p+1;{ token( RE_Dash, 0, 0 ); }}
	break;
	case 93:
#line 889 "./rlscan.rl"
	{te = p+1;{ token( RE_SqClose ); {cs = stack[--top];goto _again;} }}
	break;
	case 91:
#line 891 "./rlscan.rl"
	{te = p+1;{
			scan_error() << "unterminated OR literal" << endl;
		}}
	break;
	case 90:
#line 896 "./rlscan.rl"
	{te = p+1;{ token( RE_Char, ts, te ); }}
	break;
	case 94:
#line 896 "./rlscan.rl"
	{te = p;p--;{ token( RE_Char, ts, te ); }}
	break;
	case 116:
#line 902 "./rlscan.rl"
	{te = p+1;{ token( RE_Char, '\0' ); }}
	break;
	case 117:
#line 903 "./rlscan.rl"
	{te = p+1;{ token( RE_Char, '\a' ); }}
	break;
	case 118:
#line 904 "./rlscan.rl"
	{te = p+1;{ token( RE_Char, '\b' ); }}
	break;
	case 122:
#line 905 "./rlscan.rl"
	{te = p+1;{ token( RE_Char, '\t' ); }}
	break;
	case 120:
#line 906 "./rlscan.rl"
	{te = p+1;{ token( RE_Char, '\n' ); }}
	break;
	case 123:
#line 907 "./rlscan.rl"
	{te = p+1;{ token( RE_Char, '\v' ); }}
	break;
	case 119:
#line 908 "./rlscan.rl"
	{te = p+1;{ token( RE_Char, '\f' ); }}
	break;
	case 121:
#line 909 "./rlscan.rl"
	{te = p+1;{ token( RE_Char, '\r' ); }}
	break;
	case 115:
#line 910 "./rlscan.rl"
	{te = p+1;{ updateCol(); }}
	break;
	case 114:
#line 911 "./rlscan.rl"
	{te = p+1;{ token( RE_Char, ts+1, te ); }}
	break;
	case 110:
#line 914 "./rlscan.rl"
	{te = p+1;{ 
			token( RE_Slash, ts, te ); 
			{cs = 146; goto _again;}
		}}
	break;
	case 108:
#line 920 "./rlscan.rl"
	{te = p+1;{ token( RE_Dot ); }}
	break;
	case 107:
#line 921 "./rlscan.rl"
	{te = p+1;{ token( RE_Star ); }}
	break;
	case 112:
#line 924 "./rlscan.rl"
	{te = p+1;{ token( RE_SqOpenNeg ); {stack[top++] = cs; cs = 137; goto _again;} }}
	break;
	case 106:
#line 926 "./rlscan.rl"
	{te = p+1;{
			scan_error() << "unterminated regular expression" << endl;
		}}
	break;
	case 105:
#line 931 "./rlscan.rl"
	{te = p+1;{ token( RE_Char, ts, te ); }}
	break;
	case 109:
#line 914 "./rlscan.rl"
	{te = p;p--;{ 
			token( RE_Slash, ts, te ); 
			{cs = 146; goto _again;}
		}}
	break;
	case 111:
#line 923 "./rlscan.rl"
	{te = p;p--;{ token( RE_SqOpen ); {stack[top++] = cs; cs = 137; goto _again;} }}
	break;
	case 113:
#line 931 "./rlscan.rl"
	{te = p;p--;{ token( RE_Char, ts, te ); }}
	break;
	case 125:
#line 938 "./rlscan.rl"
	{te = p+1;{ token( ';' ); {cs = 146; goto _again;} }}
	break;
	case 124:
#line 940 "./rlscan.rl"
	{te = p+1;{
			scan_error() << "unterminated write statement" << endl;
		}}
	break;
	case 127:
#line 936 "./rlscan.rl"
	{te = p;p--;{ token( TK_Word, ts, te ); }}
	break;
	case 126:
#line 937 "./rlscan.rl"
	{te = p;p--;{ updateCol(); }}
	break;
	case 137:
#line 1006 "./rlscan.rl"
	{te = p+1;{ token( TK_Literal, ts, te ); }}
	break;
	case 186:
#line 1009 "./rlscan.rl"
	{te = p+1;{ token( RE_SqOpenNeg ); {stack[top++] = cs; cs = 137; goto _again;} }}
	break;
	case 131:
#line 1011 "./rlscan.rl"
	{te = p+1;{ token( RE_Slash ); {cs = 139; goto _again;} }}
	break;
	case 157:
#line 1016 "./rlscan.rl"
	{te = p+1;{ token( TK_ColonEquals ); }}
	break;
	case 178:
#line 1019 "./rlscan.rl"
	{te = p+1;{ token( TK_StartToState ); }}
	break;
	case 143:
#line 1020 "./rlscan.rl"
	{te = p+1;{ token( TK_AllToState ); }}
	break;
	case 149:
#line 1021 "./rlscan.rl"
	{te = p+1;{ token( TK_FinalToState ); }}
	break;
	case 165:
#line 1022 "./rlscan.rl"
	{te = p+1;{ token( TK_NotStartToState ); }}
	break;
	case 183:
#line 1023 "./rlscan.rl"
	{te = p+1;{ token( TK_NotFinalToState ); }}
	break;
	case 171:
#line 1024 "./rlscan.rl"
	{te = p+1;{ token( TK_MiddleToState ); }}
	break;
	case 174:
#line 1027 "./rlscan.rl"
	{te = p+1;{ token( TK_StartFromState ); }}
	break;
	case 139:
#line 1028 "./rlscan.rl"
	{te = p+1;{ token( TK_AllFromState ); }}
	break;
	case 145:
#line 1029 "./rlscan.rl"
	{te = p+1;{ token( TK_FinalFromState ); }}
	break;
	case 161:
#line 1030 "./rlscan.rl"
	{te = p+1;{ token( TK_NotStartFromState ); }}
	break;
	case 180:
#line 1031 "./rlscan.rl"
	{te = p+1;{ token( TK_NotFinalFromState ); }}
	break;
	case 168:
#line 1032 "./rlscan.rl"
	{te = p+1;{ token( TK_MiddleFromState ); }}
	break;
	case 175:
#line 1035 "./rlscan.rl"
	{te = p+1;{ token( TK_StartEOF ); }}
	break;
	case 140:
#line 1036 "./rlscan.rl"
	{te = p+1;{ token( TK_AllEOF ); }}
	break;
	case 146:
#line 1037 "./rlscan.rl"
	{te = p+1;{ token( TK_FinalEOF ); }}
	break;
	case 162:
#line 1038 "./rlscan.rl"
	{te = p+1;{ token( TK_NotStartEOF ); }}
	break;
	case 181:
#line 1039 "./rlscan.rl"
	{te = p+1;{ token( TK_NotFinalEOF ); }}
	break;
	case 169:
#line 1040 "./rlscan.rl"
	{te = p+1;{ token( TK_MiddleEOF ); }}
	break;
	case 173:
#line 1043 "./rlscan.rl"
	{te = p+1;{ token( TK_StartGblError ); }}
	break;
	case 138:
#line 1044 "./rlscan.rl"
	{te = p+1;{ token( TK_AllGblError ); }}
	break;
	case 144:
#line 1045 "./rlscan.rl"
	{te = p+1;{ token( TK_FinalGblError ); }}
	break;
	case 160:
#line 1046 "./rlscan.rl"
	{te = p+1;{ token( TK_NotStartGblError ); }}
	break;
	case 179:
#line 1047 "./rlscan.rl"
	{te = p+1;{ token( TK_NotFinalGblError ); }}
	break;
	case 167:
#line 1048 "./rlscan.rl"
	{te = p+1;{ token( TK_MiddleGblError ); }}
	break;
	case 177:
#line 1051 "./rlscan.rl"
	{te = p+1;{ token( TK_StartLocalError ); }}
	break;
	case 142:
#line 1052 "./rlscan.rl"
	{te = p+1;{ token( TK_AllLocalError ); }}
	break;
	case 148:
#line 1053 "./rlscan.rl"
	{te = p+1;{ token( TK_FinalLocalError ); }}
	break;
	case 164:
#line 1054 "./rlscan.rl"
	{te = p+1;{ token( TK_NotStartLocalError ); }}
	break;
	case 182:
#line 1055 "./rlscan.rl"
	{te = p+1;{ token( TK_NotFinalLocalError ); }}
	break;
	case 170:
#line 1056 "./rlscan.rl"
	{te = p+1;{ token( TK_MiddleLocalError ); }}
	break;
	case 176:
#line 1062 "./rlscan.rl"
	{te = p+1;{ token( TK_StartCond ); }}
	break;
	case 141:
#line 1063 "./rlscan.rl"
	{te = p+1;{ token( TK_AllCond ); }}
	break;
	case 147:
#line 1064 "./rlscan.rl"
	{te = p+1;{ token( TK_LeavingCond ); }}
	break;
	case 153:
#line 1066 "./rlscan.rl"
	{te = p+1;{ token( TK_DotDot ); }}
	break;
	case 150:
#line 1067 "./rlscan.rl"
	{te = p+1;{ token( TK_StarStar ); }}
	break;
	case 151:
#line 1068 "./rlscan.rl"
	{te = p+1;{ token( TK_DashDash ); }}
	break;
	case 152:
#line 1069 "./rlscan.rl"
	{te = p+1;{ token( TK_Arrow ); }}
	break;
	case 172:
#line 1070 "./rlscan.rl"
	{te = p+1;{ token( TK_DoubleArrow ); }}
	break;
	case 159:
#line 1073 "./rlscan.rl"
	{te = p+1;{ token( TK_ColonGtGt ); }}
	break;
	case 163:
#line 1074 "./rlscan.rl"
	{te = p+1;{ token( TK_LtColon ); }}
	break;
	case 208:
#line 1077 "./rlscan.rl"
	{te = p+1;{ token( TK_BarStar ); }}
	break;
	case 156:
#line 1080 "./rlscan.rl"
	{te = p+1;{ token( TK_NameSep, ts, te ); }}
	break;
	case 18:
#line 1082 "./rlscan.rl"
	{te = p+1;{ 
			updateCol();
			endSection();
			{cs = stack[--top];goto _again;}
		}}
	break;
	case 133:
#line 1099 "./rlscan.rl"
	{te = p+1;{ 
			if ( lastToken == KW_Export || lastToken == KW_Entry )
				token( '{' );
			else {
				token( '{' );
				curly_count = 1; 
				inlineBlockType = CurlyDelimited;
				if ( hostLang->lang == HostLang::Ruby )
					{stack[top++] = cs; cs = 52; goto _again;}
				else
					{stack[top++] = cs; cs = 95; goto _again;}
			}
		}}
	break;
	case 129:
#line 1113 "./rlscan.rl"
	{te = p+1;{
			scan_error() << "unterminated ragel section" << endl;
		}}
	break;
	case 128:
#line 1117 "./rlscan.rl"
	{te = p+1;{ token( *ts ); }}
	break;
	case 187:
#line 998 "./rlscan.rl"
	{te = p;p--;{ token( TK_Word, ts, te ); }}
	break;
	case 154:
#line 1001 "./rlscan.rl"
	{te = p;p--;{ token( TK_UInt, ts, te ); }}
	break;
	case 155:
#line 1002 "./rlscan.rl"
	{te = p;p--;{ token( TK_Hex, ts, te ); }}
	break;
	case 136:
#line 1006 "./rlscan.rl"
	{te = p;p--;{ token( TK_Literal, ts, te ); }}
	break;
	case 185:
#line 1008 "./rlscan.rl"
	{te = p;p--;{ token( RE_SqOpen ); {stack[top++] = cs; cs = 137; goto _again;} }}
	break;
	case 166:
#line 1059 "./rlscan.rl"
	{te = p;p--;{ token( TK_Middle ); }}
	break;
	case 158:
#line 1072 "./rlscan.rl"
	{te = p;p--;{ token( TK_ColonGt ); }}
	break;
	case 134:
#line 1088 "./rlscan.rl"
	{te = p;p--;{ updateCol(); }}
	break;
	case 135:
#line 1117 "./rlscan.rl"
	{te = p;p--;{ token( *ts ); }}
	break;
	case 17:
#line 1001 "./rlscan.rl"
	{{p = ((te))-1;}{ token( TK_UInt, ts, te ); }}
	break;
	case 15:
#line 1117 "./rlscan.rl"
	{{p = ((te))-1;}{ token( *ts ); }}
	break;
	case 184:
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
			{cs = 143; goto _again;}
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
				{stack[top++] = cs; cs = 52; goto _again;}
			else
				{stack[top++] = cs; cs = 95; goto _again;}
		}
	break;
	case 97:
	{{p = ((te))-1;} 
			token( KW_Access );
			inlineBlockType = SemiTerminated;
			if ( hostLang->lang == HostLang::Ruby )
				{stack[top++] = cs; cs = 52; goto _again;}
			else
				{stack[top++] = cs; cs = 95; goto _again;}
		}
	break;
	case 98:
	{{p = ((te))-1;} 
			token( KW_Variable );
			inlineBlockType = SemiTerminated;
			if ( hostLang->lang == HostLang::Ruby )
				{stack[top++] = cs; cs = 52; goto _again;}
			else
				{stack[top++] = cs; cs = 95; goto _again;}
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
	break;
	case 20:
#line 1126 "./rlscan.rl"
	{te = p+1;{ pass( IMP_Literal, ts, te ); }}
	break;
	case 214:
#line 1128 "./rlscan.rl"
	{te = p+1;{ 
			updateCol();
			singleLineSpec = false;
			startSection();
			{stack[top++] = cs; cs = 146; goto _again;}
		}}
	break;
	case 210:
#line 1141 "./rlscan.rl"
	{te = p+1;}
	break;
	case 209:
#line 1142 "./rlscan.rl"
	{te = p+1;{ pass( *ts, 0, 0 ); }}
	break;
	case 216:
#line 1122 "./rlscan.rl"
	{te = p;p--;{ pass( IMP_Word, ts, te ); }}
	break;
	case 215:
#line 1123 "./rlscan.rl"
	{te = p;p--;{ pass( IMP_UInt, ts, te ); }}
	break;
	case 213:
#line 1134 "./rlscan.rl"
	{te = p;p--;{ 
			updateCol();
			singleLineSpec = true;
			startSection();
			{stack[top++] = cs; cs = 146; goto _again;}
		}}
	break;
	case 211:
#line 1140 "./rlscan.rl"
	{te = p;p--;{ pass(); }}
	break;
	case 212:
#line 1142 "./rlscan.rl"
	{te = p;p--;{ pass( *ts, 0, 0 ); }}
	break;
	case 19:
#line 1142 "./rlscan.rl"
	{{p = ((te))-1;}{ pass( *ts, 0, 0 ); }}
	break;
	case 4:
#line 1150 "./rlscan.rl"
	{te = p+1;{ pass(); }}
	break;
	case 3:
#line 1151 "./rlscan.rl"
	{te = p+1;{ pass( IMP_Literal, ts, te ); }}
	break;
	case 31:
#line 1153 "./rlscan.rl"
	{te = p+1;{ 
			updateCol();
			singleLineSpec = false;
			startSection();
			{stack[top++] = cs; cs = 146; goto _again;}
		}}
	break;
	case 25:
#line 1166 "./rlscan.rl"
	{te = p+1;}
	break;
	case 24:
#line 1167 "./rlscan.rl"
	{te = p+1;{ pass( *ts, 0, 0 ); }}
	break;
	case 34:
#line 1148 "./rlscan.rl"
	{te = p;p--;{ pass( IMP_Word, ts, te ); }}
	break;
	case 32:
#line 1149 "./rlscan.rl"
	{te = p;p--;{ pass( IMP_UInt, ts, te ); }}
	break;
	case 30:
#line 1159 "./rlscan.rl"
	{te = p;p--;{ 
			updateCol();
			singleLineSpec = true;
			startSection();
			{stack[top++] = cs; cs = 146; goto _again;}
		}}
	break;
	case 28:
#line 1165 "./rlscan.rl"
	{te = p;p--;{ pass(); }}
	break;
	case 29:
#line 1167 "./rlscan.rl"
	{te = p;p--;{ pass( *ts, 0, 0 ); }}
	break;
	case 1:
#line 1167 "./rlscan.rl"
	{{p = ((te))-1;}{ pass( *ts, 0, 0 ); }}
	break;
	case 33:
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
	break;
	case 8:
#line 637 "./rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
#line 719 "./rlscan.rl"
	{te = p+1;{ token( IL_Comment, ts, te ); }}
	break;
	case 13:
#line 637 "./rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
#line 822 "./rlscan.rl"
	{te = p+1;{ token( IL_Comment, ts, te ); }}
	break;
	case 16:
#line 637 "./rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
#line 1014 "./rlscan.rl"
	{te = p+1;{ updateCol(); }}
	break;
	case 130:
#line 637 "./rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
#line 1091 "./rlscan.rl"
	{te = p+1;{
			updateCol();
			if ( singleLineSpec ) {
				endSection();
				{cs = stack[--top];goto _again;}
			}
		}}
	break;
	case 21:
#line 637 "./rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
#line 1124 "./rlscan.rl"
	{te = p+1;{ pass(); }}
	break;
	case 5:
#line 637 "./rlscan.rl"
	{ 
		lastnl = p; 
		column = 0;
		line++;
	}
#line 1150 "./rlscan.rl"
	{te = p+1;{ pass(); }}
	break;
	case 60:
#line 1 "NONE"
	{te = p+1;}
#line 670 "./rlscan.rl"
	{act = 1;}
	break;
	case 54:
#line 1 "NONE"
	{te = p+1;}
#line 672 "./rlscan.rl"
	{act = 3;}
	break;
	case 62:
#line 1 "NONE"
	{te = p+1;}
#line 673 "./rlscan.rl"
	{act = 4;}
	break;
	case 55:
#line 1 "NONE"
	{te = p+1;}
#line 674 "./rlscan.rl"
	{act = 5;}
	break;
	case 58:
#line 1 "NONE"
	{te = p+1;}
#line 680 "./rlscan.rl"
	{act = 6;}
	break;
	case 56:
#line 1 "NONE"
	{te = p+1;}
#line 684 "./rlscan.rl"
	{act = 7;}
	break;
	case 57:
#line 1 "NONE"
	{te = p+1;}
#line 685 "./rlscan.rl"
	{act = 8;}
	break;
	case 59:
#line 1 "NONE"
	{te = p+1;}
#line 689 "./rlscan.rl"
	{act = 9;}
	break;
	case 53:
#line 1 "NONE"
	{te = p+1;}
#line 693 "./rlscan.rl"
	{act = 10;}
	break;
	case 61:
#line 1 "NONE"
	{te = p+1;}
#line 697 "./rlscan.rl"
	{act = 11;}
	break;
	case 51:
#line 1 "NONE"
	{te = p+1;}
#line 701 "./rlscan.rl"
	{act = 12;}
	break;
	case 41:
#line 1 "NONE"
	{te = p+1;}
#line 706 "./rlscan.rl"
	{act = 13;}
	break;
	case 87:
#line 1 "NONE"
	{te = p+1;}
#line 773 "./rlscan.rl"
	{act = 27;}
	break;
	case 81:
#line 1 "NONE"
	{te = p+1;}
#line 775 "./rlscan.rl"
	{act = 29;}
	break;
	case 89:
#line 1 "NONE"
	{te = p+1;}
#line 776 "./rlscan.rl"
	{act = 30;}
	break;
	case 82:
#line 1 "NONE"
	{te = p+1;}
#line 777 "./rlscan.rl"
	{act = 31;}
	break;
	case 85:
#line 1 "NONE"
	{te = p+1;}
#line 783 "./rlscan.rl"
	{act = 32;}
	break;
	case 83:
#line 1 "NONE"
	{te = p+1;}
#line 787 "./rlscan.rl"
	{act = 33;}
	break;
	case 84:
#line 1 "NONE"
	{te = p+1;}
#line 788 "./rlscan.rl"
	{act = 34;}
	break;
	case 86:
#line 1 "NONE"
	{te = p+1;}
#line 792 "./rlscan.rl"
	{act = 35;}
	break;
	case 80:
#line 1 "NONE"
	{te = p+1;}
#line 796 "./rlscan.rl"
	{act = 36;}
	break;
	case 88:
#line 1 "NONE"
	{te = p+1;}
#line 800 "./rlscan.rl"
	{act = 37;}
	break;
	case 78:
#line 1 "NONE"
	{te = p+1;}
#line 804 "./rlscan.rl"
	{act = 38;}
	break;
	case 68:
#line 1 "NONE"
	{te = p+1;}
#line 809 "./rlscan.rl"
	{act = 39;}
	break;
	case 200:
#line 1 "NONE"
	{te = p+1;}
#line 948 "./rlscan.rl"
	{act = 88;}
	break;
	case 197:
#line 1 "NONE"
	{te = p+1;}
#line 949 "./rlscan.rl"
	{act = 89;}
	break;
	case 196:
#line 1 "NONE"
	{te = p+1;}
#line 950 "./rlscan.rl"
	{act = 90;}
	break;
	case 207:
#line 1 "NONE"
	{te = p+1;}
#line 951 "./rlscan.rl"
	{act = 91;}
	break;
	case 189:
#line 1 "NONE"
	{te = p+1;}
#line 955 "./rlscan.rl"
	{act = 92;}
	break;
	case 190:
#line 1 "NONE"
	{te = p+1;}
#line 956 "./rlscan.rl"
	{act = 93;}
	break;
	case 203:
#line 1 "NONE"
	{te = p+1;}
#line 957 "./rlscan.rl"
	{act = 94;}
	break;
	case 202:
#line 1 "NONE"
	{te = p+1;}
#line 958 "./rlscan.rl"
	{act = 95;}
	break;
	case 195:
#line 1 "NONE"
	{te = p+1;}
#line 963 "./rlscan.rl"
	{act = 96;}
	break;
	case 188:
#line 1 "NONE"
	{te = p+1;}
#line 971 "./rlscan.rl"
	{act = 97;}
	break;
	case 205:
#line 1 "NONE"
	{te = p+1;}
#line 979 "./rlscan.rl"
	{act = 98;}
	break;
	case 206:
#line 1 "NONE"
	{te = p+1;}
#line 987 "./rlscan.rl"
	{act = 99;}
	break;
	case 198:
#line 1 "NONE"
	{te = p+1;}
#line 988 "./rlscan.rl"
	{act = 100;}
	break;
	case 201:
#line 1 "NONE"
	{te = p+1;}
#line 989 "./rlscan.rl"
	{act = 101;}
	break;
	case 191:
#line 1 "NONE"
	{te = p+1;}
#line 990 "./rlscan.rl"
	{act = 102;}
	break;
	case 192:
#line 1 "NONE"
	{te = p+1;}
#line 991 "./rlscan.rl"
	{act = 103;}
	break;
	case 199:
#line 1 "NONE"
	{te = p+1;}
#line 992 "./rlscan.rl"
	{act = 104;}
	break;
	case 204:
#line 1 "NONE"
	{te = p+1;}
#line 993 "./rlscan.rl"
	{act = 105;}
	break;
	case 194:
#line 1 "NONE"
	{te = p+1;}
#line 994 "./rlscan.rl"
	{act = 106;}
	break;
	case 193:
#line 1 "NONE"
	{te = p+1;}
#line 995 "./rlscan.rl"
	{act = 107;}
	break;
	case 132:
#line 1 "NONE"
	{te = p+1;}
#line 998 "./rlscan.rl"
	{act = 108;}
	break;
	case 35:
#line 1 "NONE"
	{te = p+1;}
#line 1147 "./rlscan.rl"
	{act = 176;}
	break;
	case 27:
#line 1 "NONE"
	{te = p+1;}
#line 1148 "./rlscan.rl"
	{act = 177;}
	break;
#line 4764 "rlscan.fast_build.cpp"
	}

_again:
	switch ( _rlscan_to_state_actions[cs] ) {
	case 22:
#line 1 "NONE"
	{ts = 0;}
	break;
#line 4773 "rlscan.fast_build.cpp"
	}

	if ( cs == 0 )
		goto _out;
	if ( ++p != pe )
		goto _resume;
	_test_eof: {}
	if ( p == eof )
	{
	if ( _rlscan_eof_trans[cs] > 0 ) {
		_trans = _rlscan_eof_trans[cs] - 1;
		goto _eof_trans;
	}
	}

	_out: {}
	}

#line 1237 "./rlscan.rl"

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
