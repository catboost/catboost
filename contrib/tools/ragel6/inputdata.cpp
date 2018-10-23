/*
 *  Copyright 2008 Adrian Thurston <thurston@complang.org>
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
#include "common.h"
#include "inputdata.h"
#include "parsedata.h"
#include "rlparse.h"
#include <iostream>
#include "dotcodegen.h"

using std::cout;
using std::cerr;
using std::endl;
using std::ios;

/* Invoked by the parser when the root element is opened. */
void InputData::cdDefaultFileName( const char *inputFile )
{
	/* If the output format is code and no output file name is given, then
	 * make a default. */
	if ( outputFileName == 0 ) {
		const char *ext = findFileExtension( inputFile );
		if ( ext != 0 && strcmp( ext, ".rh" ) == 0 )
			outputFileName = fileNameFromStem( inputFile, ".h" );
		else {
			const char *defExtension = 0;
			switch ( hostLang->lang ) {
				case HostLang::C: defExtension = ".c"; break;
				case HostLang::D: defExtension = ".d"; break;
				case HostLang::D2: defExtension = ".d"; break;
				default: break;
			}
			outputFileName = fileNameFromStem( inputFile, defExtension );
		}
	}
}

/* Invoked by the parser when the root element is opened. */
void InputData::goDefaultFileName( const char *inputFile )
{
	/* If the output format is code and no output file name is given, then
	 * make a default. */
	if ( outputFileName == 0 )
		outputFileName = fileNameFromStem( inputFile, ".go" );
}

/* Invoked by the parser when the root element is opened. */
void InputData::javaDefaultFileName( const char *inputFile )
{
	/* If the output format is code and no output file name is given, then
	 * make a default. */
	if ( outputFileName == 0 )
		outputFileName = fileNameFromStem( inputFile, ".java" );
}

/* Invoked by the parser when the root element is opened. */
void InputData::rubyDefaultFileName( const char *inputFile )
{
	/* If the output format is code and no output file name is given, then
	 * make a default. */
	if ( outputFileName == 0 )
		outputFileName = fileNameFromStem( inputFile, ".rb" );
}

/* Invoked by the parser when the root element is opened. */
void InputData::csharpDefaultFileName( const char *inputFile )
{
	/* If the output format is code and no output file name is given, then
	 * make a default. */
	if ( outputFileName == 0 ) {
		const char *ext = findFileExtension( inputFile );
		if ( ext != 0 && strcmp( ext, ".rh" ) == 0 )
			outputFileName = fileNameFromStem( inputFile, ".h" );
		else
			outputFileName = fileNameFromStem( inputFile, ".cs" );
	}
}

/* Invoked by the parser when the root element is opened. */
void InputData::ocamlDefaultFileName( const char *inputFile )
{
	/* If the output format is code and no output file name is given, then
	 * make a default. */
	if ( outputFileName == 0 )
		outputFileName = fileNameFromStem( inputFile, ".ml" );
}

void InputData::makeOutputStream()
{
	if ( ! generateDot && ! generateXML ) {
		switch ( hostLang->lang ) {
			case HostLang::C:
			case HostLang::D:
			case HostLang::D2:
				cdDefaultFileName( inputFileName );
				break;
			case HostLang::Java:
				javaDefaultFileName( inputFileName );
				break;
			case HostLang::Go:
				goDefaultFileName( inputFileName );
				break;
			case HostLang::Ruby:
				rubyDefaultFileName( inputFileName );
				break;
			case HostLang::CSharp:
				csharpDefaultFileName( inputFileName );
				break;
			case HostLang::OCaml:
				ocamlDefaultFileName( inputFileName );
				break;
		}
	}

	/* Make sure we are not writing to the same file as the input file. */
	if ( outputFileName != 0 ) {
		if ( strcmp( inputFileName, outputFileName  ) == 0 ) {
			error() << "output file \"" << outputFileName  << 
					"\" is the same as the input file" << endl;
		}

		/* Create the filter on the output and open it. */
		outFilter = new output_filter( outputFileName );

		/* Open the output stream, attaching it to the filter. */
		outStream = new ostream( outFilter );
	}
	else {
		/* Writing out ot std out. */
		outStream = &cout;
	}
}

void InputData::openOutput()
{
	if ( outFilter != 0 ) {
		outFilter->open( outputFileName, ios::out|ios::trunc );
		if ( !outFilter->is_open() ) {
			error() << "error opening " << outputFileName << " for writing" << endl;
			exit(1);
		}
	}
}

void InputData::prepareMachineGen()
{
	if ( generateDot ) {
		/* Locate a machine spec to generate dot output for. We can only emit.
		 * Dot takes one graph at a time. */
		if ( machineSpec != 0 ) {
			/* Machine specified. */
			ParserDictEl *pdEl = parserDict.find( machineSpec );
			if ( pdEl == 0 )
				error() << "could not locate machine specified with -S and/or -M" << endp;
			dotGenParser = pdEl->value;
		}
		else { 
			/* No machine spec given, just use the first one. */
			if ( parserList.length() == 0 )
				error() << "no machine specification to generate graphviz output" << endp;

			dotGenParser = parserList.head;
		}

		GraphDictEl *gdEl = 0;

		if ( machineName != 0 ) {
			gdEl = dotGenParser->pd->graphDict.find( machineName );
			if ( gdEl == 0 )
				error() << "machine definition/instantiation not found" << endp;
		}
		else {
			/* We are using the whole machine spec. Need to make sure there
			 * are instances in the spec. */
			if ( dotGenParser->pd->instanceList.length() == 0 )
				error() << "no machine instantiations to generate graphviz output" << endp;
		}

		dotGenParser->pd->prepareMachineGen( gdEl );
	}
	else {
		/* No machine spec or machine name given. Generate everything. */
		for ( ParserDict::Iter parser = parserDict; parser.lte(); parser++ ) {
			ParseData *pd = parser->value->pd;
			if ( pd->instanceList.length() > 0 )
				pd->prepareMachineGen( 0 );
		}
	}
}

void InputData::generateReduced()
{
	if ( generateDot )
		dotGenParser->pd->generateReduced( *this );
	else {
		for ( ParserDict::Iter parser = parserDict; parser.lte(); parser++ ) {
			ParseData *pd = parser->value->pd;
			if ( pd->instanceList.length() > 0 )
				pd->generateReduced( *this );
		}
	}
}

/* Send eof to all parsers. */
void InputData::terminateAllParsers( )
{
	/* FIXME: a proper token is needed here. Suppose we should use the
	 * location of EOF in the last file that the parser was referenced in. */
	InputLoc loc;
	loc.fileName = "<EOF>";
	loc.line = 0;
	loc.col = 0;
	for ( ParserDict::Iter pdel = parserDict; pdel.lte(); pdel++ )
		pdel->value->token( loc, Parser_tk_eof, 0, 0 );
}

void InputData::verifyWritesHaveData()
{
	if ( !generateXML && !generateDot ) {
		for ( InputItemList::Iter ii = inputItems; ii.lte(); ii++ ) {
			if ( ii->type == InputItem::Write ) {
				if ( ii->pd->cgd == 0 )
					error( ii->loc ) << "no machine instantiations to write" << endl;
			}
		}
	}
}

void InputData::writeOutput()
{
	if ( generateXML )
		writeXML( *outStream );
	else if ( generateDot )
		static_cast<GraphvizDotGen*>(dotGenParser->pd->cgd)->writeDotFile();
	else {
		bool hostLineDirective = true;
		for ( InputItemList::Iter ii = inputItems; ii.lte(); ii++ ) {
			if ( ii->type == InputItem::Write ) {
				CodeGenData *cgd = ii->pd->cgd;
				::keyOps = &cgd->thisKeyOps;

				hostLineDirective = cgd->writeStatement( ii->loc,
						ii->writeArgs.length()-1, ii->writeArgs.data );
			}
			else {
				if ( hostLineDirective ) {
					/* Write statements can turn off host line directives for
					 * host sections that follow them. */
					*outStream << '\n';
					lineDirective( *outStream, inputFileName, ii->loc.line );
				}
				*outStream << ii->data.str();
				hostLineDirective = true;
			}
		}
	}
}

