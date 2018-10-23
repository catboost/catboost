/*
 *  Copyright 2001-2008 Adrian Thurston <thurston@complang.org>
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
#include <iomanip>
#include <errno.h>
#include <stdlib.h>
#include <limits.h>

#include "ragel.h"
#include "rlparse.h"
#include "parsedata.h"
#include "parsetree.h"
#include "mergesort.h"
#include "xmlcodegen.h"
#include "version.h"
#include "inputdata.h"

using namespace std;

char mainMachine[] = "main";

void Token::set( const char *str, int len )
{
	length = len;
	data = new char[len+1];
	memcpy( data, str, len );
	data[len] = 0;
}

void Token::append( const Token &other )
{
	int newLength = length + other.length;
	char *newString = new char[newLength+1];
	memcpy( newString, data, length );
	memcpy( newString + length, other.data, other.length );
	newString[newLength] = 0;
	data = newString;
	length = newLength;
}

/* Perform minimization after an operation according 
 * to the command line args. */
void afterOpMinimize( FsmAp *fsm, bool lastInSeq )
{
	/* Switch on the prefered minimization algorithm. */
	if ( minimizeOpt == MinimizeEveryOp || ( minimizeOpt == MinimizeMostOps && lastInSeq ) ) {
		/* First clean up the graph. FsmAp operations may leave these
		 * lying around. There should be no dead end states. The subtract
		 * intersection operators are the only places where they may be
		 * created and those operators clean them up. */
		fsm->removeUnreachableStates();

		switch ( minimizeLevel ) {
			case MinimizeApprox:
				fsm->minimizeApproximate();
				break;
			case MinimizePartition1:
				fsm->minimizePartition1();
				break;
			case MinimizePartition2:
				fsm->minimizePartition2();
				break;
			case MinimizeStable:
				fsm->minimizeStable();
				break;
		}
	}
}

/* Count the transitions in the fsm by walking the state list. */
int countTransitions( FsmAp *fsm )
{
	int numTrans = 0;
	StateAp *state = fsm->stateList.head;
	while ( state != 0 ) {
		numTrans += state->outList.length();
		state = state->next;
	}
	return numTrans;
}

Key makeFsmKeyHex( char *str, const InputLoc &loc, ParseData *pd )
{
	/* Reset errno so we can check for overflow or underflow. In the event of
	 * an error, sets the return val to the upper or lower bound being tested
	 * against. */
	errno = 0;
	unsigned int size = keyOps->alphType->size;
	bool unusedBits = size < sizeof(unsigned long);

	unsigned long ul = strtoul( str, 0, 16 );

	if ( errno == ERANGE || ( unusedBits && ul >> (size * 8) ) ) {
		error(loc) << "literal " << str << " overflows the alphabet type" << endl;
		ul = 1 << (size * 8);
	}

	if ( unusedBits && keyOps->alphType->isSigned && ul >> (size * 8 - 1) )
		ul |= ( (unsigned long)(-1L) >> (size*8) ) << (size*8);

	return Key( (long)ul );
}

#ifdef _MSC_VER
#   define strtoll _strtoi64
#endif

Key makeFsmKeyDec( char *str, const InputLoc &loc, ParseData *pd )
{
	if ( keyOps->alphType->isSigned ) {
		/* Convert the number to a decimal. First reset errno so we can check
		 * for overflow or underflow. */
		errno = 0;
		long long minVal = keyOps->alphType->sMinVal;
		long long maxVal = keyOps->alphType->sMaxVal;
	
		long long ll = strtoll( str, 0, 10 );
	
		/* Check for underflow. */
		if ( ( errno == ERANGE && ll < 0 ) || ll < minVal) {
			error(loc) << "literal " << str << " underflows the alphabet type" << endl;
			ll = minVal;
		}
		/* Check for overflow. */
		else if ( ( errno == ERANGE && ll > 0 ) || ll > maxVal ) {
			error(loc) << "literal " << str << " overflows the alphabet type" << endl;
			ll = maxVal;
		}
	
		return Key( (long)ll );
	}
	else {
		/* Convert the number to a decimal. First reset errno so we can check
		 * for overflow or underflow. */
		errno = 0;
		unsigned long long minVal = keyOps->alphType->uMinVal;
		unsigned long long maxVal = keyOps->alphType->uMaxVal;
	
		unsigned long long ull = strtoull( str, 0, 10 );
	
		/* Check for underflow. */
		if ( ( errno == ERANGE && ull < 0 ) || ull < minVal) {
			error(loc) << "literal " << str << " underflows the alphabet type" << endl;
			ull = minVal;
		}
		/* Check for overflow. */
		else if ( ( errno == ERANGE && ull > 0 ) || ull > maxVal ) {
			error(loc) << "literal " << str << " overflows the alphabet type" << endl;
			ull = maxVal;
		}
	
		return Key( (unsigned long)ull );
	}
}

/* Make an fsm key in int format (what the fsm graph uses) from an alphabet
 * number returned by the parser. Validates that the number doesn't overflow
 * the alphabet type. */
Key makeFsmKeyNum( char *str, const InputLoc &loc, ParseData *pd )
{
	/* Switch on hex/decimal format. */
	if ( str[0] == '0' && str[1] == 'x' )
		return makeFsmKeyHex( str, loc, pd );
	else
		return makeFsmKeyDec( str, loc, pd );
}

/* Make an fsm int format (what the fsm graph uses) from a single character.
 * Performs proper conversion depending on signed/unsigned property of the
 * alphabet. */
Key makeFsmKeyChar( char c, ParseData *pd )
{
	if ( keyOps->isSigned ) {
		/* Copy from a char type. */
		return Key( c );
	}
	else {
		/* Copy from an unsigned byte type. */
		return Key( (unsigned char)c );
	}
}

/* Make an fsm key array in int format (what the fsm graph uses) from a string
 * of characters. Performs proper conversion depending on signed/unsigned
 * property of the alphabet. */
void makeFsmKeyArray( Key *result, char *data, int len, ParseData *pd )
{
	if ( keyOps->isSigned ) {
		/* Copy from a char star type. */
		char *src = data;
		for ( int i = 0; i < len; i++ )
			result[i] = Key(src[i]);
	}
	else {
		/* Copy from an unsigned byte ptr type. */
		unsigned char *src = (unsigned char*) data;
		for ( int i = 0; i < len; i++ )
			result[i] = Key(src[i]);
	}
}

/* Like makeFsmKeyArray except the result has only unique keys. They ordering
 * will be changed. */
void makeFsmUniqueKeyArray( KeySet &result, char *data, int len, 
		bool caseInsensitive, ParseData *pd )
{
	/* Use a transitions list for getting unique keys. */
	if ( keyOps->isSigned ) {
		/* Copy from a char star type. */
		char *src = data;
		for ( int si = 0; si < len; si++ ) {
			Key key( src[si] );
			result.insert( key );
			if ( caseInsensitive ) {
				if ( key.isLower() )
					result.insert( key.toUpper() );
				else if ( key.isUpper() )
					result.insert( key.toLower() );
			}
		}
	}
	else {
		/* Copy from an unsigned byte ptr type. */
		unsigned char *src = (unsigned char*) data;
		for ( int si = 0; si < len; si++ ) {
			Key key( src[si] );
			result.insert( key );
			if ( caseInsensitive ) {
				if ( key.isLower() )
					result.insert( key.toUpper() );
				else if ( key.isUpper() )
					result.insert( key.toLower() );
			}
		}
	}
}

FsmAp *dotFsm( ParseData *pd )
{
	FsmAp *retFsm = new FsmAp();
	retFsm->rangeFsm( keyOps->minKey, keyOps->maxKey );
	return retFsm;
}

FsmAp *dotStarFsm( ParseData *pd )
{
	FsmAp *retFsm = new FsmAp();
	retFsm->rangeStarFsm( keyOps->minKey, keyOps->maxKey );
	return retFsm;
}

/* Make a builtin type. Depends on the signed nature of the alphabet type. */
FsmAp *makeBuiltin( BuiltinMachine builtin, ParseData *pd )
{
	/* FsmAp created to return. */
	FsmAp *retFsm = 0;
	bool isSigned = keyOps->isSigned;

	switch ( builtin ) {
	case BT_Any: {
		/* All characters. */
		retFsm = dotFsm( pd );
		break;
	}
	case BT_Ascii: {
		/* Ascii characters 0 to 127. */
		retFsm = new FsmAp();
		retFsm->rangeFsm( 0, 127 );
		break;
	}
	case BT_Extend: {
		/* Ascii extended characters. This is the full byte range. Dependent
		 * on signed, vs no signed. If the alphabet is one byte then just use
		 * dot fsm. */
		if ( isSigned ) {
			retFsm = new FsmAp();
			retFsm->rangeFsm( -128, 127 );
		}
		else {
			retFsm = new FsmAp();
			retFsm->rangeFsm( 0, 255 );
		}
		break;
	}
	case BT_Alpha: {
		/* Alpha [A-Za-z]. */
		FsmAp *upper = new FsmAp(), *lower = new FsmAp();
		upper->rangeFsm( 'A', 'Z' );
		lower->rangeFsm( 'a', 'z' );
		upper->unionOp( lower );
		upper->minimizePartition2();
		retFsm = upper;
		break;
	}
	case BT_Digit: {
		/* Digits [0-9]. */
		retFsm = new FsmAp();
		retFsm->rangeFsm( '0', '9' );
		break;
	}
	case BT_Alnum: {
		/* Alpha numerics [0-9A-Za-z]. */
		FsmAp *digit = new FsmAp(), *lower = new FsmAp();
		FsmAp *upper = new FsmAp();
		digit->rangeFsm( '0', '9' );
		upper->rangeFsm( 'A', 'Z' );
		lower->rangeFsm( 'a', 'z' );
		digit->unionOp( upper );
		digit->unionOp( lower );
		digit->minimizePartition2();
		retFsm = digit;
		break;
	}
	case BT_Lower: {
		/* Lower case characters. */
		retFsm = new FsmAp();
		retFsm->rangeFsm( 'a', 'z' );
		break;
	}
	case BT_Upper: {
		/* Upper case characters. */
		retFsm = new FsmAp();
		retFsm->rangeFsm( 'A', 'Z' );
		break;
	}
	case BT_Cntrl: {
		/* Control characters. */
		FsmAp *cntrl = new FsmAp();
		FsmAp *highChar = new FsmAp();
		cntrl->rangeFsm( 0, 31 );
		highChar->concatFsm( 127 );
		cntrl->unionOp( highChar );
		cntrl->minimizePartition2();
		retFsm = cntrl;
		break;
	}
	case BT_Graph: {
		/* Graphical ascii characters [!-~]. */
		retFsm = new FsmAp();
		retFsm->rangeFsm( '!', '~' );
		break;
	}
	case BT_Print: {
		/* Printable characters. Same as graph except includes space. */
		retFsm = new FsmAp();
		retFsm->rangeFsm( ' ', '~' );
		break;
	}
	case BT_Punct: {
		/* Punctuation. */
		FsmAp *range1 = new FsmAp();
		FsmAp *range2 = new FsmAp();
		FsmAp *range3 = new FsmAp(); 
		FsmAp *range4 = new FsmAp();
		range1->rangeFsm( '!', '/' );
		range2->rangeFsm( ':', '@' );
		range3->rangeFsm( '[', '`' );
		range4->rangeFsm( '{', '~' );
		range1->unionOp( range2 );
		range1->unionOp( range3 );
		range1->unionOp( range4 );
		range1->minimizePartition2();
		retFsm = range1;
		break;
	}
	case BT_Space: {
		/* Whitespace: [\t\v\f\n\r ]. */
		FsmAp *cntrl = new FsmAp();
		FsmAp *space = new FsmAp();
		cntrl->rangeFsm( '\t', '\r' );
		space->concatFsm( ' ' );
		cntrl->unionOp( space );
		cntrl->minimizePartition2();
		retFsm = cntrl;
		break;
	}
	case BT_Xdigit: {
		/* Hex digits [0-9A-Fa-f]. */
		FsmAp *digit = new FsmAp();
		FsmAp *upper = new FsmAp();
		FsmAp *lower = new FsmAp();
		digit->rangeFsm( '0', '9' );
		upper->rangeFsm( 'A', 'F' );
		lower->rangeFsm( 'a', 'f' );
		digit->unionOp( upper );
		digit->unionOp( lower );
		digit->minimizePartition2();
		retFsm = digit;
		break;
	}
	case BT_Lambda: {
		retFsm = new FsmAp();
		retFsm->lambdaFsm();
		break;
	}
	case BT_Empty: {
		retFsm = new FsmAp();
		retFsm->emptyFsm();
		break;
	}}

	return retFsm;
}

/* Check if this name inst or any name inst below is referenced. */
bool NameInst::anyRefsRec()
{
	if ( numRefs > 0 )
		return true;

	/* Recurse on children until true. */
	for ( NameVect::Iter ch = childVect; ch.lte(); ch++ ) {
		if ( (*ch)->anyRefsRec() )
			return true;
	}

	return false;
}

/*
 * ParseData
 */

/* Initialize the structure that will collect info during the parse of a
 * machine. */
ParseData::ParseData( const char *fileName, char *sectionName, 
		const InputLoc &sectionLoc )
:	
	sectionGraph(0),
	generatingSectionSubset(false),
	nextPriorKey(0),
	/* 0 is reserved for global error actions. */
	nextLocalErrKey(1),
	nextNameId(0),
	nextCondId(0),
	alphTypeSet(false),
	getKeyExpr(0),
	accessExpr(0),
	prePushExpr(0),
	postPopExpr(0),
	pExpr(0),
	peExpr(0),
	eofExpr(0),
	csExpr(0),
	topExpr(0),
	stackExpr(0),
	actExpr(0),
	tokstartExpr(0),
	tokendExpr(0),
	dataExpr(0),
	lowerNum(0),
	upperNum(0),
	fileName(fileName),
	sectionName(sectionName),
	sectionLoc(sectionLoc),
	curActionOrd(0),
	curPriorOrd(0),
	rootName(0),
	exportsRootName(0),
	nextEpsilonResolvedLink(0),
	nextLongestMatchId(1),
	lmRequiresErrorState(false),
	cgd(0)
{
	/* Initialize the dictionary of graphs. This is our symbol table. The
	 * initialization needs to be done on construction which happens at the
	 * beginning of a machine spec so any assignment operators can reference
	 * the builtins. */
	initGraphDict();
}

/* Clean up the data collected during a parse. */
ParseData::~ParseData()
{
	/* Delete all the nodes in the action list. Will cause all the
	 * string data that represents the actions to be deallocated. */
	actionList.empty();
}

/* Make a name id in the current name instantiation scope if it is not
 * already there. */
NameInst *ParseData::addNameInst( const InputLoc &loc, const char *data, bool isLabel )
{
	/* Create the name instantitaion object and insert it. */
	NameInst *newNameInst = new NameInst( loc, curNameInst, data, nextNameId++, isLabel );
	curNameInst->childVect.append( newNameInst );
	if ( data != 0 )
		curNameInst->children.insertMulti( data, newNameInst );
	return newNameInst;
}

void ParseData::initNameWalk()
{
	curNameInst = rootName;
	curNameChild = 0;
}

void ParseData::initExportsNameWalk()
{
	curNameInst = exportsRootName;
	curNameChild = 0;
}

/* Goes into the next child scope. The number of the child is already set up.
 * We need this for the syncronous name tree and parse tree walk to work
 * properly. It is reset on entry into a scope and advanced on poping of a
 * scope. A call to enterNameScope should be accompanied by a corresponding
 * popNameScope. */
NameFrame ParseData::enterNameScope( bool isLocal, int numScopes )
{
	/* Save off the current data. */
	NameFrame retFrame;
	retFrame.prevNameInst = curNameInst;
	retFrame.prevNameChild = curNameChild;
	retFrame.prevLocalScope = localNameScope;

	/* Enter into the new name scope. */
	for ( int i = 0; i < numScopes; i++ ) {
		curNameInst = curNameInst->childVect[curNameChild];
		curNameChild = 0;
	}

	if ( isLocal )
		localNameScope = curNameInst;

	return retFrame;
}

/* Return from a child scope to a parent. The parent info must be specified as
 * an argument and is obtained from the corresponding call to enterNameScope.
 * */
void ParseData::popNameScope( const NameFrame &frame )
{
	/* Pop the name scope. */
	curNameInst = frame.prevNameInst;
	curNameChild = frame.prevNameChild+1;
	localNameScope = frame.prevLocalScope;
}

void ParseData::resetNameScope( const NameFrame &frame )
{
	/* Pop the name scope. */
	curNameInst = frame.prevNameInst;
	curNameChild = frame.prevNameChild;
	localNameScope = frame.prevLocalScope;
}


void ParseData::unsetObsoleteEntries( FsmAp *graph )
{
	/* Loop the reference names and increment the usage. Names that are no
	 * longer needed will be unset in graph. */
	for ( NameVect::Iter ref = curNameInst->referencedNames; ref.lte(); ref++ ) {
		/* Get the name. */
		NameInst *name = *ref;
		name->numUses += 1;

		/* If the name is no longer needed unset its corresponding entry. */
		if ( name->numUses == name->numRefs ) {
			assert( graph->entryPoints.find( name->id ) != 0 );
			graph->unsetEntry( name->id );
			assert( graph->entryPoints.find( name->id ) == 0 );
		}
	}
}

NameSet ParseData::resolvePart( NameInst *refFrom, const char *data, bool recLabelsOnly )
{
	/* Queue needed for breadth-first search, load it with the start node. */
	NameInstList nameQueue;
	nameQueue.append( refFrom );

	NameSet result;
	while ( nameQueue.length() > 0 ) {
		/* Pull the next from location off the queue. */
		NameInst *from = nameQueue.detachFirst();

		/* Look for the name. */
		NameMapEl *low, *high;
		if ( from->children.findMulti( data, low, high ) ) {
			/* Record all instances of the name. */
			for ( ; low <= high; low++ )
				result.insert( low->value );
		}

		/* Name not there, do breadth-first operation of appending all
		 * childrent to the processing queue. */
		for ( NameVect::Iter name = from->childVect; name.lte(); name++ ) {
			if ( !recLabelsOnly || (*name)->isLabel )
				nameQueue.append( *name );
		}
	}

	/* Queue exhausted and name never found. */
	return result;
}

void ParseData::resolveFrom( NameSet &result, NameInst *refFrom, 
		const NameRef &nameRef, int namePos )
{
	/* Look for the name in the owning scope of the factor with aug. */
	NameSet partResult = resolvePart( refFrom, nameRef[namePos], false );
	
	/* If there are more parts to the name then continue on. */
	if ( ++namePos < nameRef.length() ) {
		/* There are more components to the name, search using all the part
		 * results as the base. */
		for ( NameSet::Iter name = partResult; name.lte(); name++ )
			resolveFrom( result, *name, nameRef, namePos );
	}
	else {
		/* This is the last component, append the part results to the final
		 * results. */
		result.insert( partResult );
	}
}

/* Write out a name reference. */
ostream &operator<<( ostream &out, const NameRef &nameRef )
{
	int pos = 0;
	if ( nameRef[pos] == 0 ) {
		out << "::";
		pos += 1;
	}
	out << nameRef[pos++];
	for ( ; pos < nameRef.length(); pos++ )
		out << "::" << nameRef[pos];
	return out;
}

ostream &operator<<( ostream &out, const NameInst &nameInst )
{
	/* Count the number fully qualified name parts. */
	int numParents = 0;
	NameInst *curParent = nameInst.parent;
	while ( curParent != 0 ) {
		numParents += 1;
		curParent = curParent->parent;
	}

	/* Make an array and fill it in. */
	curParent = nameInst.parent;
	NameInst **parents = new NameInst*[numParents];
	for ( int p = numParents-1; p >= 0; p-- ) {
		parents[p] = curParent;
		curParent = curParent->parent;
	}
		
	/* Write the parents out, skip the root. */
	for ( int p = 1; p < numParents; p++ )
		out << "::" << ( parents[p]->name != 0 ? parents[p]->name : "<ANON>" );

	/* Write the name and cleanup. */
	out << "::" << ( nameInst.name != 0 ? nameInst.name : "<ANON>" );
	delete[] parents;
	return out;
}

struct CmpNameInstLoc
{
	static int compare( const NameInst *ni1, const NameInst *ni2 )
	{
		if ( ni1->loc.line < ni2->loc.line )
			return -1;
		else if ( ni1->loc.line > ni2->loc.line )
			return 1;
		else if ( ni1->loc.col < ni2->loc.col )
			return -1;
		else if ( ni1->loc.col > ni2->loc.col )
			return 1;
		return 0;
	}
};

void errorStateLabels( const NameSet &resolved )
{
	MergeSort<NameInst*, CmpNameInstLoc> mergeSort;
	mergeSort.sort( resolved.data, resolved.length() );
	for ( NameSet::Iter res = resolved; res.lte(); res++ )
		error((*res)->loc) << "  -> " << **res << endl;
}


NameInst *ParseData::resolveStateRef( const NameRef &nameRef, InputLoc &loc, Action *action )
{
	NameInst *nameInst = 0;

	/* Do the local search if the name is not strictly a root level name
	 * search. */
	if ( nameRef[0] != 0 ) {
		/* If the action is referenced, resolve all of them. */
		if ( action != 0 && action->actionRefs.length() > 0 ) {
			/* Look for the name in all referencing scopes. */
			NameSet resolved;
			for ( ActionRefs::Iter actRef = action->actionRefs; actRef.lte(); actRef++ )
				resolveFrom( resolved, *actRef, nameRef, 0 );

			if ( resolved.length() > 0 ) {
				/* Take the first one. */
				nameInst = resolved[0];
				if ( resolved.length() > 1 ) {
					/* Complain about the multiple references. */
					error(loc) << "state reference " << nameRef << 
							" resolves to multiple entry points" << endl;
					errorStateLabels( resolved );
				}
			}
		}
	}

	/* If not found in the local scope, look in global. */
	if ( nameInst == 0 ) {
		NameSet resolved;
		int fromPos = nameRef[0] != 0 ? 0 : 1;
		resolveFrom( resolved, rootName, nameRef, fromPos );

		if ( resolved.length() > 0 ) {
			/* Take the first. */
			nameInst = resolved[0];
			if ( resolved.length() > 1 ) {
				/* Complain about the multiple references. */
				error(loc) << "state reference " << nameRef << 
						" resolves to multiple entry points" << endl;
				errorStateLabels( resolved );
			}
		}
	}

	if ( nameInst == 0 ) {
		/* If not found then complain. */
		error(loc) << "could not resolve state reference " << nameRef << endl;
	}
	return nameInst;
}

void ParseData::resolveNameRefs( InlineList *inlineList, Action *action )
{
	for ( InlineList::Iter item = *inlineList; item.lte(); item++ ) {
		switch ( item->type ) {
			case InlineItem::Entry: case InlineItem::Goto:
			case InlineItem::Call: case InlineItem::Next: {
				/* Resolve, pass action for local search. */
				NameInst *target = resolveStateRef( *item->nameRef, item->loc, action );

				/* Name lookup error reporting is handled by resolveStateRef. */
				if ( target != 0 ) {
					/* Check if the target goes into a longest match. */
					NameInst *search = target->parent;
					while ( search != 0 ) {
						if ( search->isLongestMatch ) {
							error(item->loc) << "cannot enter inside a longest "
									"match construction as an entry point" << endl;
							break;
						}
						search = search->parent;
					}

					/* Record the reference in the name. This will cause the
					 * entry point to survive to the end of the graph
					 * generating walk. */
					target->numRefs += 1;
				}

				item->nameTarg = target;
				break;
			}
			default:
				break;
		}

		/* Some of the item types may have children. */
		if ( item->children != 0 )
			resolveNameRefs( item->children, action );
	}
}

/* Resolve references to labels in actions. */
void ParseData::resolveActionNameRefs()
{
	for ( ActionList::Iter act = actionList; act.lte(); act++ ) {
		/* Only care about the actions that are referenced. */
		if ( act->actionRefs.length() > 0 )
			resolveNameRefs( act->inlineList, act );
	}
}

/* Walk a name tree starting at from and fill the name index. */
void ParseData::fillNameIndex( NameInst *from )
{
	/* Fill the value for from in the name index. */
	nameIndex[from->id] = from;

	/* Recurse on the implicit final state and then all children. */
	if ( from->final != 0 )
		fillNameIndex( from->final );
	for ( NameVect::Iter name = from->childVect; name.lte(); name++ )
		fillNameIndex( *name );
}

void ParseData::makeRootNames()
{
	/* Create the root name. */
	rootName = new NameInst( InputLoc(), 0, 0, nextNameId++, false );
	exportsRootName = new NameInst( InputLoc(), 0, 0, nextNameId++, false );
}

/* Build the name tree and supporting data structures. */
void ParseData::makeNameTree( GraphDictEl *dictEl )
{
	/* Set up curNameInst for the walk. */
	initNameWalk();

	if ( dictEl != 0 ) {
		/* A start location has been specified. */
		dictEl->value->makeNameTree( dictEl->loc, this );
	}
	else {
		/* First make the name tree. */
		for ( GraphList::Iter glel = instanceList; glel.lte(); glel++ ) {
			/* Recurse on the instance. */
			glel->value->makeNameTree( glel->loc, this );
		}
	}
	
	/* The number of nodes in the tree can now be given by nextNameId */
	nameIndex = new NameInst*[nextNameId];
	memset( nameIndex, 0, sizeof(NameInst*)*nextNameId );
	fillNameIndex( rootName );
	fillNameIndex( exportsRootName );
}


void ParseData::createBuiltin( const char *name, BuiltinMachine builtin )
{
	Expression *expression = new Expression( builtin );
	Join *join = new Join( expression );
	MachineDef *machineDef = new MachineDef( join );
	VarDef *varDef = new VarDef( name, machineDef );
	GraphDictEl *graphDictEl = new GraphDictEl( name, varDef );
	graphDict.insert( graphDictEl );
}

/* Initialize the graph dict with builtin types. */
void ParseData::initGraphDict( )
{
	createBuiltin( "any", BT_Any );
	createBuiltin( "ascii", BT_Ascii );
	createBuiltin( "extend", BT_Extend );
	createBuiltin( "alpha", BT_Alpha );
	createBuiltin( "digit", BT_Digit );
	createBuiltin( "alnum", BT_Alnum );
	createBuiltin( "lower", BT_Lower );
	createBuiltin( "upper", BT_Upper );
	createBuiltin( "cntrl", BT_Cntrl );
	createBuiltin( "graph", BT_Graph );
	createBuiltin( "print", BT_Print );
	createBuiltin( "punct", BT_Punct );
	createBuiltin( "space", BT_Space );
	createBuiltin( "xdigit", BT_Xdigit );
	createBuiltin( "null", BT_Lambda );
	createBuiltin( "zlen", BT_Lambda );
	createBuiltin( "empty", BT_Empty );
}

/* Set the alphabet type. If the types are not valid returns false. */
bool ParseData::setAlphType( const InputLoc &loc, char *s1, char *s2 )
{
	alphTypeLoc = loc;
	userAlphType = findAlphType( s1, s2 );
	alphTypeSet = true;
	return userAlphType != 0;
}

/* Set the alphabet type. If the types are not valid returns false. */
bool ParseData::setAlphType( const InputLoc &loc, char *s1 )
{
	alphTypeLoc = loc;
	userAlphType = findAlphType( s1 );
	alphTypeSet = true;
	return userAlphType != 0;
}

bool ParseData::setVariable( char *var, InlineList *inlineList )
{
	bool set = true;

	if ( strcmp( var, "p" ) == 0 )
		pExpr = inlineList;
	else if ( strcmp( var, "pe" ) == 0 )
		peExpr = inlineList;
	else if ( strcmp( var, "eof" ) == 0 )
		eofExpr = inlineList;
	else if ( strcmp( var, "cs" ) == 0 )
		csExpr = inlineList;
	else if ( strcmp( var, "data" ) == 0 )
		dataExpr = inlineList;
	else if ( strcmp( var, "top" ) == 0 )
		topExpr = inlineList;
	else if ( strcmp( var, "stack" ) == 0 )
		stackExpr = inlineList;
	else if ( strcmp( var, "act" ) == 0 )
		actExpr = inlineList;
	else if ( strcmp( var, "ts" ) == 0 )
		tokstartExpr = inlineList;
	else if ( strcmp( var, "te" ) == 0 )
		tokendExpr = inlineList;
	else
		set = false;

	return set;
}

/* Initialize the key operators object that will be referenced by all fsms
 * created. */
void ParseData::initKeyOps( )
{
	/* Signedness and bounds. */
	HostType *alphType = alphTypeSet ? userAlphType : hostLang->defaultAlphType;
	thisKeyOps.setAlphType( alphType );

	if ( lowerNum != 0 ) {
		/* If ranges are given then interpret the alphabet type. */
		thisKeyOps.minKey = makeFsmKeyNum( lowerNum, rangeLowLoc, this );
		thisKeyOps.maxKey = makeFsmKeyNum( upperNum, rangeHighLoc, this );
	}

	thisCondData.lastCondKey = thisKeyOps.maxKey;
}

void ParseData::printNameInst( NameInst *nameInst, int level )
{
	for ( int i = 0; i < level; i++ )
		cerr << "  ";
	cerr << (nameInst->name != 0 ? nameInst->name : "<ANON>") << 
			"  id: " << nameInst->id << 
			"  refs: " << nameInst->numRefs <<
			"  uses: " << nameInst->numUses << endl;
	for ( NameVect::Iter name = nameInst->childVect; name.lte(); name++ )
		printNameInst( *name, level+1 );
}

/* Remove duplicates of unique actions from an action table. */
void ParseData::removeDups( ActionTable &table )
{
	/* Scan through the table looking for unique actions to 
	 * remove duplicates of. */
	for ( int i = 0; i < table.length(); i++ ) {
		/* Remove any duplicates ahead of i. */
		for ( int r = i+1; r < table.length(); ) {
			if ( table[r].value == table[i].value )
				table.vremove(r);
			else
				r += 1;
		}
	}
}

/* Remove duplicates from action lists. This operates only on transition and
 * eof action lists and so should be called once all actions have been
 * transfered to their final resting place. */
void ParseData::removeActionDups( FsmAp *graph )
{
	/* Loop all states. */
	for ( StateList::Iter state = graph->stateList; state.lte(); state++ ) {
		/* Loop all transitions. */
		for ( TransList::Iter trans = state->outList; trans.lte(); trans++ )
			removeDups( trans->actionTable );
		removeDups( state->toStateActionTable );
		removeDups( state->fromStateActionTable );
		removeDups( state->eofActionTable );
	}
}

Action *ParseData::newAction( const char *name, InlineList *inlineList )
{
	InputLoc loc;
	loc.line = 1;
	loc.col = 1;
	loc.fileName = "NONE";

	Action *action = new Action( loc, name, inlineList, nextCondId++ );
	action->actionRefs.append( rootName );
	actionList.append( action );
	return action;
}

void ParseData::initLongestMatchData()
{
	if ( lmList.length() > 0 ) {
		/* The initTokStart action resets the token start. */
		InlineList *il1 = new InlineList;
		il1->append( new InlineItem( InputLoc(), InlineItem::LmInitTokStart ) );
		initTokStart = newAction( "initts", il1 );
		initTokStart->isLmAction = true;

		/* The initActId action gives act a default value. */
		InlineList *il4 = new InlineList;
		il4->append( new InlineItem( InputLoc(), InlineItem::LmInitAct ) );
		initActId = newAction( "initact", il4 );
		initActId->isLmAction = true;

		/* The setTokStart action sets tokstart. */
		InlineList *il5 = new InlineList;
		il5->append( new InlineItem( InputLoc(), InlineItem::LmSetTokStart ) );
		setTokStart = newAction( "ts", il5 );
		setTokStart->isLmAction = true;

		/* The setTokEnd action sets tokend. */
		InlineList *il3 = new InlineList;
		il3->append( new InlineItem( InputLoc(), InlineItem::LmSetTokEnd ) );
		setTokEnd = newAction( "te", il3 );
		setTokEnd->isLmAction = true;

		/* The action will also need an ordering: ahead of all user action
		 * embeddings. */
		initTokStartOrd = curActionOrd++;
		initActIdOrd = curActionOrd++;
		setTokStartOrd = curActionOrd++;
		setTokEndOrd = curActionOrd++;
	}
}

/* After building the graph, do some extra processing to ensure the runtime
 * data of the longest mactch operators is consistent. */
void ParseData::setLongestMatchData( FsmAp *graph )
{
	if ( lmList.length() > 0 ) {
		/* Make sure all entry points (targets of fgoto, fcall, fnext, fentry)
		 * init the tokstart. */
		for ( EntryMap::Iter en = graph->entryPoints; en.lte(); en++ ) {
			/* This is run after duplicates are removed, we must guard against
			 * inserting a duplicate. */
			ActionTable &actionTable = en->value->toStateActionTable;
			if ( ! actionTable.hasAction( initTokStart ) )
				actionTable.setAction( initTokStartOrd, initTokStart );
		}

		/* Find the set of states that are the target of transitions with
		 * actions that have calls. These states will be targeted by fret
		 * statements. */
		StateSet states;
		for ( StateList::Iter state = graph->stateList; state.lte(); state++ ) {
			for ( TransList::Iter trans = state->outList; trans.lte(); trans++ ) {
				for ( ActionTable::Iter ati = trans->actionTable; ati.lte(); ati++ ) {
					if ( ati->value->anyCall && trans->toState != 0 )
						states.insert( trans->toState );
				}
			}
		}


		/* Init tokstart upon entering the above collected states. */
		for ( StateSet::Iter ps = states; ps.lte(); ps++ ) {
			/* This is run after duplicates are removed, we must guard against
			 * inserting a duplicate. */
			ActionTable &actionTable = (*ps)->toStateActionTable;
			if ( ! actionTable.hasAction( initTokStart ) )
				actionTable.setAction( initTokStartOrd, initTokStart );
		}
	}
}

/* Make the graph from a graph dict node. Does minimization and state sorting. */
FsmAp *ParseData::makeInstance( GraphDictEl *gdNode )
{
	/* Build the graph from a walk of the parse tree. */
	FsmAp *graph = gdNode->value->walk( this );

	/* Resolve any labels that point to multiple states. Any labels that are
	 * still around are referenced only by gotos and calls and they need to be
	 * made into deterministic entry points. */
	graph->deterministicEntry();

	/*
	 * All state construction is now complete.
	 */

	/* Transfer actions from the out action tables to eof action tables. */
	for ( StateSet::Iter state = graph->finStateSet; state.lte(); state++ )
		graph->transferOutActions( *state );

	/* Transfer global error actions. */
	for ( StateList::Iter state = graph->stateList; state.lte(); state++ )
		graph->transferErrorActions( state, 0 );
	
	if ( ::wantDupsRemoved )
		removeActionDups( graph );

	/* Remove unreachable states. There should be no dead end states. The
	 * subtract and intersection operators are the only places where they may
	 * be created and those operators clean them up. */
	graph->removeUnreachableStates();

	/* No more fsm operations are to be done. Action ordering numbers are
	 * no longer of use and will just hinder minimization. Clear them. */
	graph->nullActionKeys();

	/* Transition priorities are no longer of use. We can clear them
	 * because they will just hinder minimization as well. Clear them. */
	graph->clearAllPriorities();

	if ( minimizeOpt != MinimizeNone ) {
		/* Minimize here even if we minimized at every op. Now that function
		 * keys have been cleared we may get a more minimal fsm. */
		switch ( minimizeLevel ) {
			case MinimizeApprox:
				graph->minimizeApproximate();
				break;
			case MinimizeStable:
				graph->minimizeStable();
				break;
			case MinimizePartition1:
				graph->minimizePartition1();
				break;
			case MinimizePartition2:
				graph->minimizePartition2();
				break;
		}
	}

	graph->compressTransitions();

	return graph;
}

void ParseData::printNameTree()
{
	/* Print the name instance map. */
	for ( NameVect::Iter name = rootName->childVect; name.lte(); name++ )
		printNameInst( *name, 0 );
	
	cerr << "name index:" << endl;
	/* Show that the name index is correct. */
	for ( int ni = 0; ni < nextNameId; ni++ ) {
		cerr << ni << ": ";
		const char *name = nameIndex[ni]->name;
		cerr << ( name != 0 ? name : "<ANON>" ) << endl;
	}
}

FsmAp *ParseData::makeSpecific( GraphDictEl *gdNode )
{
	/* Build the name tree and supporting data structures. */
	makeNameTree( gdNode );

	/* Resove name references from gdNode. */
	initNameWalk();
	gdNode->value->resolveNameRefs( this );

	/* Do not resolve action references. Since we are not building the entire
	 * graph there's a good chance that many name references will fail. This
	 * is okay since generating part of the graph is usually only done when
	 * inspecting the compiled machine. */

	/* Same story for extern entry point references. */

	/* Flag this case so that the XML code generator is aware that we haven't
	 * looked up name references in actions. It can then avoid segfaulting. */
	generatingSectionSubset = true;

	/* Just building the specified graph. */
	initNameWalk();
	FsmAp *mainGraph = makeInstance( gdNode );

	return mainGraph;
}

FsmAp *ParseData::makeAll()
{
	/* Build the name tree and supporting data structures. */
	makeNameTree( 0 );

	/* Resove name references in the tree. */
	initNameWalk();
	for ( GraphList::Iter glel = instanceList; glel.lte(); glel++ )
		glel->value->resolveNameRefs( this );

	/* Resolve action code name references. */
	resolveActionNameRefs();

	/* Force name references to the top level instantiations. */
	for ( NameVect::Iter inst = rootName->childVect; inst.lte(); inst++ )
		(*inst)->numRefs += 1;

	FsmAp *mainGraph = 0;
	FsmAp **graphs = new FsmAp*[instanceList.length()];
	int numOthers = 0;

	/* Make all the instantiations, we know that main exists in this list. */
	initNameWalk();
	for ( GraphList::Iter glel = instanceList; glel.lte();  glel++ ) {
		if ( strcmp( glel->key, mainMachine ) == 0 ) {
			/* Main graph is always instantiated. */
			mainGraph = makeInstance( glel );
		}
		else {
			/* Instantiate and store in others array. */
			graphs[numOthers++] = makeInstance( glel );
		}
	}

	if ( mainGraph == 0 )
		mainGraph = graphs[--numOthers];

	if ( numOthers > 0 ) {
		/* Add all the other graphs into main. */
		mainGraph->globOp( graphs, numOthers );
	}

	delete[] graphs;
	return mainGraph;
}

void ParseData::analyzeAction( Action *action, InlineList *inlineList )
{
	/* FIXME: Actions used as conditions should be very constrained. */
	for ( InlineList::Iter item = *inlineList; item.lte(); item++ ) {
		if ( item->type == InlineItem::Call || item->type == InlineItem::CallExpr )
			action->anyCall = true;

		/* Need to recurse into longest match items. */
		if ( item->type == InlineItem::LmSwitch ) {
			LongestMatch *lm = item->longestMatch;
			for ( LmPartList::Iter lmi = *lm->longestMatchList; lmi.lte(); lmi++ ) {
				if ( lmi->action != 0 )
					analyzeAction( action, lmi->action->inlineList );
			}
		}

		if ( item->type == InlineItem::LmOnLast || 
				item->type == InlineItem::LmOnNext ||
				item->type == InlineItem::LmOnLagBehind )
		{
			LongestMatchPart *lmi = item->longestMatchPart;
			if ( lmi->action != 0 )
				analyzeAction( action, lmi->action->inlineList );
		}

		if ( item->children != 0 )
			analyzeAction( action, item->children );
	}
}


/* Check actions for bad uses of fsm directives. We don't go inside longest
 * match items in actions created by ragel, since we just want the user
 * actions. */
void ParseData::checkInlineList( Action *act, InlineList *inlineList )
{
	for ( InlineList::Iter item = *inlineList; item.lte(); item++ ) {
		/* EOF checks. */
		if ( act->numEofRefs > 0 ) {
			switch ( item->type ) {
				/* Currently no checks. */
				default:
					break;
			}
		}

		/* Recurse. */
		if ( item->children != 0 )
			checkInlineList( act, item->children );
	}
}

void ParseData::checkAction( Action *action )
{
	/* Check for actions with calls that are embedded within a longest match
	 * machine. */
	if ( !action->isLmAction && action->numRefs() > 0 && action->anyCall ) {
		for ( ActionRefs::Iter ar = action->actionRefs; ar.lte(); ar++ ) {
			NameInst *check = *ar;
			while ( check != 0 ) {
				if ( check->isLongestMatch ) {
					error(action->loc) << "within a scanner, fcall is permitted"
						" only in pattern actions" << endl;
					break;
				}
				check = check->parent;
			}
		}
	}

	checkInlineList( action, action->inlineList );
}


void ParseData::analyzeGraph( FsmAp *graph )
{
	for ( ActionList::Iter act = actionList; act.lte(); act++ )
		analyzeAction( act, act->inlineList );

	for ( StateList::Iter st = graph->stateList; st.lte(); st++ ) {
		/* The transition list. */
		for ( TransList::Iter trans = st->outList; trans.lte(); trans++ ) {
			for ( ActionTable::Iter at = trans->actionTable; at.lte(); at++ )
				at->value->numTransRefs += 1;
		}

		for ( ActionTable::Iter at = st->toStateActionTable; at.lte(); at++ )
			at->value->numToStateRefs += 1;

		for ( ActionTable::Iter at = st->fromStateActionTable; at.lte(); at++ )
			at->value->numFromStateRefs += 1;

		for ( ActionTable::Iter at = st->eofActionTable; at.lte(); at++ )
			at->value->numEofRefs += 1;

		for ( StateCondList::Iter sc = st->stateCondList; sc.lte(); sc++ ) {
			for ( CondSet::Iter sci = sc->condSpace->condSet; sci.lte(); sci++ )
				(*sci)->numCondRefs += 1;
		}
	}

	/* Checks for bad usage of directives in action code. */
	for ( ActionList::Iter act = actionList; act.lte(); act++ )
		checkAction( act );
}

void ParseData::makeExportsNameTree()
{
	/* Make a name tree for the exports. */
	initExportsNameWalk();

	/* First make the name tree. */
	for ( GraphDict::Iter gdel = graphDict; gdel.lte(); gdel++ ) {
		if ( gdel->value->isExport ) {
			/* Recurse on the instance. */
			gdel->value->makeNameTree( gdel->loc, this );
		}
	}
}

void ParseData::makeExports()
{
	makeExportsNameTree();

	/* Resove name references in the tree. */
	initExportsNameWalk();
	for ( GraphDict::Iter gdel = graphDict; gdel.lte(); gdel++ ) {
		if ( gdel->value->isExport )
			gdel->value->resolveNameRefs( this );
	}

	/* Make all the instantiations, we know that main exists in this list. */
	initExportsNameWalk();
	for ( GraphDict::Iter gdel = graphDict; gdel.lte();  gdel++ ) {
		/* Check if this var def is an export. */
		if ( gdel->value->isExport ) {
			/* Build the graph from a walk of the parse tree. */
			FsmAp *graph = gdel->value->walk( this );

			/* Build the graph from a walk of the parse tree. */
			if ( !graph->checkSingleCharMachine() ) {
				error(gdel->loc) << "bad export machine, must define "
						"a single character" << endl;
			}
			else {
				/* Safe to extract the key and declare the export. */
				Key exportKey = graph->startState->outList.head->lowKey;
				exportList.append( new Export( gdel->value->name, exportKey ) );
			}
		}
	}

}

/* Construct the machine and catch failures which can occur during
 * construction. */
void ParseData::prepareMachineGen( GraphDictEl *graphDictEl )
{
	try {
		/* This machine construction can fail. */
		prepareMachineGenTBWrapped( graphDictEl );
	}
	catch ( FsmConstructFail fail ) {
		switch ( fail.reason ) {
			case FsmConstructFail::CondNoKeySpace: {
				InputLoc &loc = alphTypeSet ? alphTypeLoc : sectionLoc;
				error(loc) << "sorry, no more characters are "
						"available in the alphabet space" << endl;
				error(loc) << "  for conditions, please use a "
						"smaller alphtype or reduce" << endl;
				error(loc) << "  the span of characters on which "
						"conditions are embedded" << endl;
				break;
			}
		}
	}
}

void ParseData::prepareMachineGenTBWrapped( GraphDictEl *graphDictEl )
{
	beginProcessing();
	initKeyOps();
	makeRootNames();
	initLongestMatchData();

	/* Make the graph, do minimization. */
	if ( graphDictEl == 0 )
		sectionGraph = makeAll();
	else
		sectionGraph = makeSpecific( graphDictEl );
	
	/* Compute exports from the export definitions. */
	makeExports();

	/* If any errors have occured in the input file then don't write anything. */
	if ( gblErrorCount > 0 )
		return;

	analyzeGraph( sectionGraph );

	/* Depends on the graph analysis. */
	setLongestMatchData( sectionGraph );

	/* Decide if an error state is necessary.
	 *  1. There is an error transition
	 *  2. There is a gap in the transitions
	 *  3. The longest match operator requires it. */
	if ( lmRequiresErrorState || sectionGraph->hasErrorTrans() )
		sectionGraph->errState = sectionGraph->addState();

	/* State numbers need to be assigned such that all final states have a
	 * larger state id number than all non-final states. This enables the
	 * first_final mechanism to function correctly. We also want states to be
	 * ordered in a predictable fashion. So we first apply a depth-first
	 * search, then do a stable sort by final state status, then assign
	 * numbers. */

	sectionGraph->depthFirstOrdering();
	sectionGraph->sortStatesByFinal();
	sectionGraph->setStateNumbers( 0 );
}

void ParseData::generateReduced( InputData &inputData )
{
	beginProcessing();

	cgd = makeCodeGen( inputData.inputFileName, sectionName, *inputData.outStream );

	/* Make the generator. */
	BackendGen backendGen( sectionName, this, sectionGraph, cgd );

	/* Write out with it. */
	backendGen.makeBackend();

	if ( printStatistics ) {
		cerr << "fsm name  : " << sectionName << endl;
		cerr << "num states: " << sectionGraph->stateList.length() << endl;
		cerr << endl;
	}
}

void ParseData::generateXML( ostream &out )
{
	beginProcessing();

	/* Make the generator. */
	XMLCodeGen codeGen( sectionName, this, sectionGraph, out );

	/* Write out with it. */
	codeGen.writeXML();

	if ( printStatistics ) {
		cerr << "fsm name  : " << sectionName << endl;
		cerr << "num states: " << sectionGraph->stateList.length() << endl;
		cerr << endl;
	}
}

