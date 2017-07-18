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

%%{
	machine inline_token_scan;
	alphtype int;
	access tok_;

	# Import scanner tokens.
	import "rlparse.h"; 

	main := |*
		# Define of number.
		IMP_Define IMP_Word IMP_UInt => { 
			int base = tok_ts - token_data;
			int nameOff = 1;
			int numOff = 2;

			directToParser( inclToParser, fileName, line, column, TK_Word, 
					token_strings[base+nameOff], token_lens[base+nameOff] );
			directToParser( inclToParser, fileName, line, column, '=', 0, 0 );
			directToParser( inclToParser, fileName, line, column, TK_UInt,
					token_strings[base+numOff], token_lens[base+numOff] );
			directToParser( inclToParser, fileName, line, column, ';', 0, 0 );
		};

		# Assignment of number.
		IMP_Word '=' IMP_UInt => { 
			int base = tok_ts - token_data;
			int nameOff = 0;
			int numOff = 2;

			directToParser( inclToParser, fileName, line, column, TK_Word, 
					token_strings[base+nameOff], token_lens[base+nameOff] );
			directToParser( inclToParser, fileName, line, column, '=', 0, 0 );
			directToParser( inclToParser, fileName, line, column, TK_UInt,
					token_strings[base+numOff], token_lens[base+numOff] );
			directToParser( inclToParser, fileName, line, column, ';', 0, 0 );
		};

		# Define of literal.
		IMP_Define IMP_Word IMP_Literal => { 
			int base = tok_ts - token_data;
			int nameOff = 1;
			int litOff = 2;

			directToParser( inclToParser, fileName, line, column, TK_Word, 
					token_strings[base+nameOff], token_lens[base+nameOff] );
			directToParser( inclToParser, fileName, line, column, '=', 0, 0 );
			directToParser( inclToParser, fileName, line, column, TK_Literal,
					token_strings[base+litOff], token_lens[base+litOff] );
			directToParser( inclToParser, fileName, line, column, ';', 0, 0 );
		};

		# Assignment of literal.
		IMP_Word '=' IMP_Literal => { 
			int base = tok_ts - token_data;
			int nameOff = 0;
			int litOff = 2;

			directToParser( inclToParser, fileName, line, column, TK_Word, 
					token_strings[base+nameOff], token_lens[base+nameOff] );
			directToParser( inclToParser, fileName, line, column, '=', 0, 0 );
			directToParser( inclToParser, fileName, line, column, TK_Literal,
					token_strings[base+litOff], token_lens[base+litOff] );
			directToParser( inclToParser, fileName, line, column, ';', 0, 0 );
		};

		# Catch everything else.
		any;
	*|;
}%%

%% write data;

void Scanner::flushImport()
{
	int *p = token_data;
	int *pe = token_data + cur_token;
	int *eof = 0;

	%%{
		machine inline_token_scan;
		write init;
		write exec;
	}%%

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

%%{
	machine section_parse;
	alphtype int;
	write data;
}%%


void Scanner::init( )
{
	%% write init;
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

%%{
	machine section_parse;

	# Need the defines representing tokens.
	import "rlparse.h"; 

	action clear_words { word = lit = 0; word_len = lit_len = 0; }
	action store_word { word = tokdata; word_len = toklen; }
	action store_lit { lit = tokdata; lit_len = toklen; }

	action mach_err { scan_error() << "bad machine statement" << endl; }
	action incl_err { scan_error() << "bad include statement" << endl; }
	action import_err { scan_error() << "bad import statement" << endl; }
	action write_err { scan_error() << "bad write statement" << endl; }

	action handle_machine { handleMachine(); }
	action handle_include { handleInclude(); }
	action handle_import { handleImport(); }

	machine_stmt =
		( KW_Machine TK_Word @store_word ';' ) @handle_machine
		<>err mach_err <>eof mach_err;

	include_names = (
		TK_Word @store_word ( TK_Literal @store_lit )? |
		TK_Literal @store_lit
	) >clear_words;

	include_stmt =
		( KW_Include include_names ';' ) @handle_include
		<>err incl_err <>eof incl_err;

	import_stmt =
		( KW_Import TK_Literal @store_lit ';' ) @handle_import
		<>err import_err <>eof import_err;

	action write_command
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

	action write_arg
	{
		if ( active() && machineSpec == 0 && machineName == 0 )
			id.inputItems.tail->writeArgs.append( strdup(tokdata) );
	}

	action write_close
	{
		if ( active() && machineSpec == 0 && machineName == 0 )
			id.inputItems.tail->writeArgs.append( 0 );
	}

	write_stmt =
		( KW_Write @write_command 
		( TK_Word @write_arg )+ ';' @write_close )
		<>err write_err <>eof write_err;

	action handle_token
	{
		/* Send the token off to the parser. */
		if ( active() )
			directToParser( parser, fileName, line, column, type, tokdata, toklen );
	}

	# Catch everything else.
	everything_else = 
		^( KW_Machine | KW_Include | KW_Import | KW_Write ) @handle_token;

	main := ( 
		machine_stmt |
		include_stmt |
		import_stmt |
		write_stmt |
		everything_else
	)*;
}%%

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

	%%{
		machine section_parse;
		write exec;
	}%%

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

%%{
	machine rlscan;

	# This is sent by the driver code.
	EOF = 0;
	
	action inc_nl { 
		lastnl = p; 
		column = 0;
		line++;
	}
	NL = '\n' @inc_nl;

	# Identifiers, numbers, commetns, and other common things.
	ident = ( alpha | '_' ) ( alpha |digit |'_' )*;
	number = digit+;
	hex_number = '0x' [0-9a-fA-F]+;

	c_comment = 
		'/*' ( any | NL )* :>> '*/';

	cpp_comment =
		'//' [^\n]* NL;

	c_cpp_comment = c_comment | cpp_comment;

	ruby_comment = '#' [^\n]* NL;

	# These literal forms are common to host code and ragel.
	s_literal = "'" ([^'\\] | NL | '\\' (any | NL))* "'";
	d_literal = '"' ([^"\\] | NL | '\\' (any | NL))* '"';
	host_re_literal = '/' ([^/\\] | NL | '\\' (any | NL))* '/';

	whitespace = [ \t] | NL;
	pound_comment = '#' [^\n]* NL;

	# An inline block of code for Ruby.
	inline_code_ruby := |*
		# Inline expression keywords.
		"fpc" => { token( KW_PChar ); };
		"fc" => { token( KW_Char ); };
		"fcurs" => { token( KW_CurState ); };
		"ftargs" => { token( KW_TargState ); };
		"fentry" => { 
			whitespaceOn = false; 
			token( KW_Entry );
		};

		# Inline statement keywords.
		"fhold" => { 
			whitespaceOn = false; 
			token( KW_Hold );
		};
		"fexec" => { token( KW_Exec, 0, 0 ); };
		"fgoto" => { 
			whitespaceOn = false; 
			token( KW_Goto );
		};
		"fnext" => { 
			whitespaceOn = false; 
			token( KW_Next );
		};
		"fcall" => { 
			whitespaceOn = false; 
			token( KW_Call );
		};
		"fret" => { 
			whitespaceOn = false; 
			token( KW_Ret );
		};
		"fbreak" => { 
			whitespaceOn = false; 
			token( KW_Break );
		};

		ident => { token( TK_Word, ts, te ); };

		number => { token( TK_UInt, ts, te ); };
		hex_number => { token( TK_Hex, ts, te ); };

		( s_literal | d_literal | host_re_literal ) 
			=> { token( IL_Literal, ts, te ); };

		whitespace+ => { 
			if ( whitespaceOn ) 
				token( IL_WhiteSpace, ts, te );
		};

		ruby_comment => { token( IL_Comment, ts, te ); };

		"::" => { token( TK_NameSep, ts, te ); };

		# Some symbols need to go to the parser as with their cardinal value as
		# the token type (as opposed to being sent as anonymous symbols)
		# because they are part of the sequences which we interpret. The * ) ;
		# symbols cause whitespace parsing to come back on. This gets turned
		# off by some keywords.

		";" => {
			whitespaceOn = true;
			token( *ts, ts, te );
			if ( inlineBlockType == SemiTerminated )
				fret;
		};

		[*)] => { 
			whitespaceOn = true;
			token( *ts, ts, te );
		};

		[,(] => { token( *ts, ts, te ); };

		'{' => { 
			token( IL_Symbol, ts, te );
			curly_count += 1; 
		};

		'}' => { 
			if ( --curly_count == 0 && inlineBlockType == CurlyDelimited ) {
				/* Inline code block ends. */
				token( '}' );
				fret;
			}
			else {
				/* Either a semi terminated inline block or only the closing
				 * brace of some inner scope, not the block's closing brace. */
				token( IL_Symbol, ts, te );
			}
		};

		EOF => {
			scan_error() << "unterminated code block" << endl;
		};

		# Send every other character as a symbol.
		any => { token( IL_Symbol, ts, te ); };
	*|;


	# An inline block of code for languages other than Ruby.
	inline_code := |*
		# Inline expression keywords.
		"fpc" => { token( KW_PChar ); };
		"fc" => { token( KW_Char ); };
		"fcurs" => { token( KW_CurState ); };
		"ftargs" => { token( KW_TargState ); };
		"fentry" => { 
			whitespaceOn = false; 
			token( KW_Entry );
		};

		# Inline statement keywords.
		"fhold" => { 
			whitespaceOn = false; 
			token( KW_Hold );
		};
		"fexec" => { token( KW_Exec, 0, 0 ); };
		"fgoto" => { 
			whitespaceOn = false; 
			token( KW_Goto );
		};
		"fnext" => { 
			whitespaceOn = false; 
			token( KW_Next );
		};
		"fcall" => { 
			whitespaceOn = false; 
			token( KW_Call );
		};
		"fret" => { 
			whitespaceOn = false; 
			token( KW_Ret );
		};
		"fbreak" => { 
			whitespaceOn = false; 
			token( KW_Break );
		};

		ident => { token( TK_Word, ts, te ); };

		number => { token( TK_UInt, ts, te ); };
		hex_number => { token( TK_Hex, ts, te ); };

		( s_literal | d_literal ) 
			=> { token( IL_Literal, ts, te ); };

		whitespace+ => { 
			if ( whitespaceOn ) 
				token( IL_WhiteSpace, ts, te );
		};

		c_cpp_comment => { token( IL_Comment, ts, te ); };

		"::" => { token( TK_NameSep, ts, te ); };

		# Some symbols need to go to the parser as with their cardinal value as
		# the token type (as opposed to being sent as anonymous symbols)
		# because they are part of the sequences which we interpret. The * ) ;
		# symbols cause whitespace parsing to come back on. This gets turned
		# off by some keywords.

		";" => {
			whitespaceOn = true;
			token( *ts, ts, te );
			if ( inlineBlockType == SemiTerminated )
				fret;
		};

		[*)] => { 
			whitespaceOn = true;
			token( *ts, ts, te );
		};

		[,(] => { token( *ts, ts, te ); };

		'{' => { 
			token( IL_Symbol, ts, te );
			curly_count += 1; 
		};

		'}' => { 
			if ( --curly_count == 0 && inlineBlockType == CurlyDelimited ) {
				/* Inline code block ends. */
				token( '}' );
				fret;
			}
			else {
				/* Either a semi terminated inline block or only the closing
				 * brace of some inner scope, not the block's closing brace. */
				token( IL_Symbol, ts, te );
			}
		};

		EOF => {
			scan_error() << "unterminated code block" << endl;
		};

		# Send every other character as a symbol.
		any => { token( IL_Symbol, ts, te ); };
	*|;

	or_literal := |*
		# Escape sequences in OR expressions.
		'\\0' => { token( RE_Char, '\0' ); };
		'\\a' => { token( RE_Char, '\a' ); };
		'\\b' => { token( RE_Char, '\b' ); };
		'\\t' => { token( RE_Char, '\t' ); };
		'\\n' => { token( RE_Char, '\n' ); };
		'\\v' => { token( RE_Char, '\v' ); };
		'\\f' => { token( RE_Char, '\f' ); };
		'\\r' => { token( RE_Char, '\r' ); };
		'\\\n' => { updateCol(); };
		'\\' any => { token( RE_Char, ts+1, te ); };

		# Range dash in an OR expression.
		'-' => { token( RE_Dash, 0, 0 ); };

		# Terminate an OR expression.
		']'	=> { token( RE_SqClose ); fret; };

		EOF => {
			scan_error() << "unterminated OR literal" << endl;
		};

		# Characters in an OR expression.
		[^\]] => { token( RE_Char, ts, te ); };

	*|;

	ragel_re_literal := |*
		# Escape sequences in regular expressions.
		'\\0' => { token( RE_Char, '\0' ); };
		'\\a' => { token( RE_Char, '\a' ); };
		'\\b' => { token( RE_Char, '\b' ); };
		'\\t' => { token( RE_Char, '\t' ); };
		'\\n' => { token( RE_Char, '\n' ); };
		'\\v' => { token( RE_Char, '\v' ); };
		'\\f' => { token( RE_Char, '\f' ); };
		'\\r' => { token( RE_Char, '\r' ); };
		'\\\n' => { updateCol(); };
		'\\' any => { token( RE_Char, ts+1, te ); };

		# Terminate an OR expression.
		'/' [i]? => { 
			token( RE_Slash, ts, te ); 
			fgoto parser_def;
		};

		# Special characters.
		'.' => { token( RE_Dot ); };
		'*' => { token( RE_Star ); };

		'[' => { token( RE_SqOpen ); fcall or_literal; };
		'[^' => { token( RE_SqOpenNeg ); fcall or_literal; };

		EOF => {
			scan_error() << "unterminated regular expression" << endl;
		};

		# Characters in an OR expression.
		[^\/] => { token( RE_Char, ts, te ); };
	*|;

	# We need a separate token space here to avoid the ragel keywords.
	write_statement := |*
		ident => { token( TK_Word, ts, te ); } ;
		[ \t\n]+ => { updateCol(); };
		';' => { token( ';' ); fgoto parser_def; };

		EOF => {
			scan_error() << "unterminated write statement" << endl;
		};
	*|;

	# Parser definitions. 
	parser_def := |*
		#'length_cond' => { token( KW_Length ); };
		'machine' => { token( KW_Machine ); };
		'include' => { token( KW_Include ); };
		'import' => { token( KW_Import ); };
		'write' => { 
			token( KW_Write );
			fgoto write_statement;
		};
		'action' => { token( KW_Action ); };
		'alphtype' => { token( KW_AlphType ); };
		'prepush' => { token( KW_PrePush ); };
		'postpop' => { token( KW_PostPop ); };

		# FIXME: Enable this post 5.17.
		# 'range' => { token( KW_Range ); };

		'getkey' => { 
			token( KW_GetKey );
			inlineBlockType = SemiTerminated;
			if ( hostLang->lang == HostLang::Ruby )
				fcall inline_code_ruby;
			else
				fcall inline_code;
		};
		'access' => { 
			token( KW_Access );
			inlineBlockType = SemiTerminated;
			if ( hostLang->lang == HostLang::Ruby )
				fcall inline_code_ruby;
			else
				fcall inline_code;
		};
		'variable' => { 
			token( KW_Variable );
			inlineBlockType = SemiTerminated;
			if ( hostLang->lang == HostLang::Ruby )
				fcall inline_code_ruby;
			else
				fcall inline_code;
		};
		'when' => { token( KW_When ); };
		'inwhen' => { token( KW_InWhen ); };
		'outwhen' => { token( KW_OutWhen ); };
		'eof' => { token( KW_Eof ); };
		'err' => { token( KW_Err ); };
		'lerr' => { token( KW_Lerr ); };
		'to' => { token( KW_To ); };
		'from' => { token( KW_From ); };
		'export' => { token( KW_Export ); };

		# Identifiers.
		ident => { token( TK_Word, ts, te ); } ;

		# Numbers
		number => { token( TK_UInt, ts, te ); };
		hex_number => { token( TK_Hex, ts, te ); };

		# Literals, with optionals.
		( s_literal | d_literal ) [i]? 
			=> { token( TK_Literal, ts, te ); };

		'[' => { token( RE_SqOpen ); fcall or_literal; };
		'[^' => { token( RE_SqOpenNeg ); fcall or_literal; };

		'/' => { token( RE_Slash ); fgoto ragel_re_literal; };

		# Ignore.
		pound_comment => { updateCol(); };

		':=' => { token( TK_ColonEquals ); };

		# To State Actions.
		">~" => { token( TK_StartToState ); };
		"$~" => { token( TK_AllToState ); };
		"%~" => { token( TK_FinalToState ); };
		"<~" => { token( TK_NotStartToState ); };
		"@~" => { token( TK_NotFinalToState ); };
		"<>~" => { token( TK_MiddleToState ); };

		# From State actions
		">*" => { token( TK_StartFromState ); };
		"$*" => { token( TK_AllFromState ); };
		"%*" => { token( TK_FinalFromState ); };
		"<*" => { token( TK_NotStartFromState ); };
		"@*" => { token( TK_NotFinalFromState ); };
		"<>*" => { token( TK_MiddleFromState ); };

		# EOF Actions.
		">/" => { token( TK_StartEOF ); };
		"$/" => { token( TK_AllEOF ); };
		"%/" => { token( TK_FinalEOF ); };
		"</" => { token( TK_NotStartEOF ); };
		"@/" => { token( TK_NotFinalEOF ); };
		"<>/" => { token( TK_MiddleEOF ); };

		# Global Error actions.
		">!" => { token( TK_StartGblError ); };
		"$!" => { token( TK_AllGblError ); };
		"%!" => { token( TK_FinalGblError ); };
		"<!" => { token( TK_NotStartGblError ); };
		"@!" => { token( TK_NotFinalGblError ); };
		"<>!" => { token( TK_MiddleGblError ); };

		# Local error actions.
		">^" => { token( TK_StartLocalError ); };
		"$^" => { token( TK_AllLocalError ); };
		"%^" => { token( TK_FinalLocalError ); };
		"<^" => { token( TK_NotStartLocalError ); };
		"@^" => { token( TK_NotFinalLocalError ); };
		"<>^" => { token( TK_MiddleLocalError ); };

		# Middle.
		"<>" => { token( TK_Middle ); };

		# Conditions. 
		'>?' => { token( TK_StartCond ); };
		'$?' => { token( TK_AllCond ); };
		'%?' => { token( TK_LeavingCond ); };

		'..' => { token( TK_DotDot ); };
		'**' => { token( TK_StarStar ); };
		'--' => { token( TK_DashDash ); };
		'->' => { token( TK_Arrow ); };
		'=>' => { token( TK_DoubleArrow ); };

		":>"  => { token( TK_ColonGt ); };
		":>>" => { token( TK_ColonGtGt ); };
		"<:"  => { token( TK_LtColon ); };

		# Opening of longest match.
		"|*" => { token( TK_BarStar ); };

		# Separater for name references.
		"::" => { token( TK_NameSep, ts, te ); };

		'}%%' => { 
			updateCol();
			endSection();
			fret;
		};

		[ \t\r]+ => { updateCol(); };

		# If we are in a single line machine then newline may end the spec.
		NL => {
			updateCol();
			if ( singleLineSpec ) {
				endSection();
				fret;
			}
		};

		'{' => { 
			if ( lastToken == KW_Export || lastToken == KW_Entry )
				token( '{' );
			else {
				token( '{' );
				curly_count = 1; 
				inlineBlockType = CurlyDelimited;
				if ( hostLang->lang == HostLang::Ruby )
					fcall inline_code_ruby;
				else
					fcall inline_code;
			}
		};

		EOF => {
			scan_error() << "unterminated ragel section" << endl;
		};

		any => { token( *ts ); } ;
	*|;

	# Outside code scanner. These tokens get passed through.
	main_ruby := |*
		ident => { pass( IMP_Word, ts, te ); };
		number => { pass( IMP_UInt, ts, te ); };
		ruby_comment => { pass(); };
		( s_literal | d_literal | host_re_literal ) 
			=> { pass( IMP_Literal, ts, te ); };

		'%%{' => { 
			updateCol();
			singleLineSpec = false;
			startSection();
			fcall parser_def;
		};
		'%%' => { 
			updateCol();
			singleLineSpec = true;
			startSection();
			fcall parser_def;
		};
		whitespace+ => { pass(); };
		EOF;
		any => { pass( *ts, 0, 0 ); };
	*|;

	# Outside code scanner. These tokens get passed through.
	main := |*
		'define' => { pass( IMP_Define, 0, 0 ); };
		ident => { pass( IMP_Word, ts, te ); };
		number => { pass( IMP_UInt, ts, te ); };
		c_cpp_comment => { pass(); };
		( s_literal | d_literal ) => { pass( IMP_Literal, ts, te ); };

		'%%{' => { 
			updateCol();
			singleLineSpec = false;
			startSection();
			fcall parser_def;
		};
		'%%' => { 
			updateCol();
			singleLineSpec = true;
			startSection();
			fcall parser_def;
		};
		whitespace+ => { pass(); };
		EOF;
		any => { pass( *ts, 0, 0 ); };
	*|;
}%%

%% write data;

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
	%% write init;

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

		%% write exec;

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
