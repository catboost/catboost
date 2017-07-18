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

#ifndef _XMLCODEGEN_H
#define _XMLCODEGEN_H

#include <iostream>
#include "avltree.h"
#include "fsmgraph.h"
#include "parsedata.h"
#include "redfsm.h"

/* Forwards. */
struct TransAp;
struct FsmAp;
struct ParseData;
struct GenInlineList;
struct CodeGenData;

struct RedActionTable
:
	public AvlTreeEl<RedActionTable>
{
	RedActionTable( const ActionTable &key )
	:	
		key(key), 
		id(0)
	{ }

	const ActionTable &getKey() 
		{ return key; }

	ActionTable key;
	int id;
};

typedef AvlTree<RedActionTable, ActionTable, CmpActionTable> ActionTableMap;

struct NextRedTrans
{
	Key lowKey, highKey;
	TransAp *trans;
	TransAp *next;

	void load() {
		if ( trans != 0 ) {
			next = trans->next;
			lowKey = trans->lowKey;
			highKey = trans->highKey;
		}
	}

	NextRedTrans( TransAp *t ) {
		trans = t;
		load();
	}

	void increment() {
		trans = next;
		load();
	}
};

struct GenBase
{
	GenBase( char *fsmName, ParseData *pd, FsmAp *fsm );

	void appendTrans( TransListVect &outList, Key lowKey, Key highKey, TransAp *trans );
	void reduceActionTables();

	char *fsmName;
	ParseData *pd;
	FsmAp *fsm;

	ActionTableMap actionTableMap;
	int nextActionTableId;
};

class XMLCodeGen : protected GenBase
{
public:
	XMLCodeGen( char *fsmName, ParseData *pd, FsmAp *fsm, std::ostream &out );

	void writeXML( );

private:
	void writeStateActions( StateAp *state );
	void writeStateList();
	void writeStateConditions( StateAp *state );

	void writeKey( Key key );
	void writeText( InlineItem *item );
	void writeGoto( InlineItem *item );
	void writeGotoExpr( InlineItem *item );
	void writeCall( InlineItem *item );
	void writeCallExpr( InlineItem *item );
	void writeNext( InlineItem *item );
	void writeNextExpr( InlineItem *item );
	void writeEntry( InlineItem *item );
	void writeLmOnLast( InlineItem *item );
	void writeLmOnNext( InlineItem *item );
	void writeLmOnLagBehind( InlineItem *item );

	void writeExports();
	bool writeNameInst( NameInst *nameInst );
	void writeEntryPoints();
	void writeConditions();
	void writeInlineList( InlineList *inlineList );
	void writeActionList();
	void writeActionTableList();
	void reduceTrans( TransAp *trans );
	void writeTransList( StateAp *state );
	void writeEofTrans( StateAp *state );
	void writeTrans( Key lowKey, Key highKey, TransAp *defTrans );
	void writeAction( Action *action );
	void writeLmSwitch( InlineItem *item );
	void writeMachine();
	void writeActionExec( InlineItem *item );

	std::ostream &out;
};

class BackendGen : protected GenBase
{
public:
	BackendGen( char *fsmName, ParseData *pd, FsmAp *fsm, CodeGenData *cgd );
	void makeBackend( );

private:
	void makeGenInlineList( GenInlineList *outList, InlineList *inList );
	void makeKey( GenInlineList *outList, Key key );
	void makeText( GenInlineList *outList, InlineItem *item );
	void makeLmOnLast( GenInlineList *outList, InlineItem *item );
	void makeLmOnNext( GenInlineList *outList, InlineItem *item );
	void makeLmOnLagBehind( GenInlineList *outList, InlineItem *item );
	void makeActionExec( GenInlineList *outList, InlineItem *item );
	void makeLmSwitch( GenInlineList *outList, InlineItem *item );
	void makeSetTokend( GenInlineList *outList, long offset );
	void makeSetAct( GenInlineList *outList, long lmId );
	void makeSubList( GenInlineList *outList, InlineList *inlineList, 
			GenInlineItem::Type type );
	void makeTargetItem( GenInlineList *outList, NameInst *nameTarg, GenInlineItem::Type type );
	void makeExecGetTokend( GenInlineList *outList );
	void makeExports();
	void makeMachine();
	void makeActionList();
	void makeAction( Action *action );
	void makeActionTableList();
	void makeConditions();
	void makeEntryPoints();
	bool makeNameInst( std::string &out, NameInst *nameInst );
	void makeStateList();

	void makeStateActions( StateAp *state );
	void makeEofTrans( StateAp *state );
	void makeStateConditions( StateAp *state );
	void makeTransList( StateAp *state );
	void makeTrans( Key lowKey, Key highKey, TransAp *trans );

	void close_ragel_def();

	CodeGenData *cgd;

	/* Collected during parsing. */
	int curAction;
	int curActionTable;
	int curTrans;
	int curState;
	int curCondSpace;
	int curStateCond;

};

#endif
