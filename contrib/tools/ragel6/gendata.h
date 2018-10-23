/*
 *  Copyright 2005-2007 Adrian Thurston <thurston@complang.org>
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

#ifndef _GENDATA_H
#define _GENDATA_H

#include <iostream>
#include "config.h"
#include "redfsm.h"
#include "common.h"

using std::ostream;

extern bool generateDot;

struct NameInst;
typedef DList<GenAction> GenActionList;

typedef unsigned long ulong;

extern int gblErrorCount;

struct CodeGenData;

typedef AvlMap<char *, CodeGenData*, CmpStr> CodeGenMap;
typedef AvlMapEl<char *, CodeGenData*> CodeGenMapEl;

void cdLineDirective( ostream &out, const char *fileName, int line );
void javaLineDirective( ostream &out, const char *fileName, int line );
void goLineDirective( ostream &out, const char *fileName, int line );
void rubyLineDirective( ostream &out, const char *fileName, int line );
void csharpLineDirective( ostream &out, const char *fileName, int line );
void ocamlLineDirective( ostream &out, const char *fileName, int line );
void genLineDirective( ostream &out );
void lineDirective( ostream &out, const char *fileName, int line );

string itoa( int i );

/*********************************/

struct CodeGenData
{
	/*
	 * The interface to the code generator.
	 */
	virtual void finishRagelDef() {}

	/* These are invoked by the corresponding write statements. */
	virtual void writeData() {};
	virtual void writeInit() {};
	virtual void writeExec() {};
	virtual void writeExports() {};
	virtual void writeStart() {};
	virtual void writeFirstFinal() {};
	virtual void writeError() {};

	/* This can also be overwridden to modify the processing of write
	 * statements. */
	virtual bool writeStatement( InputLoc &loc, int nargs, char **args );

	/********************/

	CodeGenData( ostream &out );
	virtual ~CodeGenData() {}

	/* 
	 * Collecting the machine.
	 */

	const char *sourceFileName;
	const char *fsmName;
	ostream &out;
	RedFsmAp *redFsm;
	GenAction *allActions;
	RedAction *allActionTables;
	Condition *allConditions;
	GenCondSpace *allCondSpaces;
	RedStateAp *allStates;
	NameInst **nameIndex;
	int startState;
	int errState;
	GenActionList actionList;
	ConditionList conditionList;
	CondSpaceList condSpaceList;
	GenInlineList *getKeyExpr;
	GenInlineList *accessExpr;
	GenInlineList *prePushExpr;
	GenInlineList *postPopExpr;

	/* Overriding variables. */
	GenInlineList *pExpr;
	GenInlineList *peExpr;
	GenInlineList *eofExpr;
	GenInlineList *csExpr;
	GenInlineList *topExpr;
	GenInlineList *stackExpr;
	GenInlineList *actExpr;
	GenInlineList *tokstartExpr;
	GenInlineList *tokendExpr;
	GenInlineList *dataExpr;

	KeyOps thisKeyOps;
	bool wantComplete;
	EntryIdVect entryPointIds;
	EntryNameVect entryPointNames;
	bool hasLongestMatch;
	ExportList exportList;

	/* Write options. */
	bool noEnd;
	bool noPrefix;
	bool noFinal;
	bool noError;
	bool noEntry;
	bool noCS;

	void createMachine();
	void initActionList( unsigned long length );
	void newAction( int anum, const char *name, const InputLoc &loc, GenInlineList *inlineList );
	void initActionTableList( unsigned long length );
	void initStateList( unsigned long length );
	void setStartState( unsigned long startState );
	void setErrorState( unsigned long errState );
	void addEntryPoint( char *name, unsigned long entryState );
	void setId( int snum, int id );
	void setFinal( int snum );
	void initTransList( int snum, unsigned long length );
	void newTrans( int snum, int tnum, Key lowKey, Key highKey, 
			long targ, long act );
	void finishTransList( int snum );
	void setStateActions( int snum, long toStateAction, 
			long fromStateAction, long eofAction );
	void setEofTrans( int snum, long targ, long eofAction );
	void setForcedErrorState()
		{ redFsm->forcedErrorState = true; }

	
	void initCondSpaceList( ulong length );
	void condSpaceItem( int cnum, long condActionId );
	void newCondSpace( int cnum, int condSpaceId, Key baseKey );

	void initStateCondList( int snum, ulong length );
	void addStateCond( int snum, Key lowKey, Key highKey, long condNum );

	GenCondSpace *findCondSpace( Key lowKey, Key highKey );
	Condition *findCondition( Key key );

	bool setAlphType( const char *data );

	void resolveTargetStates( GenInlineList *inlineList );
	Key findMaxKey();

	/* Gather various info on the machine. */
	void analyzeActionList( RedAction *redAct, GenInlineList *inlineList );
	void analyzeAction( GenAction *act, GenInlineList *inlineList );
	void findFinalActionRefs();
	void analyzeMachine();

	void closeMachine();
	void setValueLimits();
	void assignActionIds();

	ostream &source_warning( const InputLoc &loc );
	ostream &source_error( const InputLoc &loc );
	void write_option_error( InputLoc &loc, char *arg );
};

CodeGenData *makeCodeGen( const char *sourceFileName, 
		const char *fsmName, ostream &out );

#endif
