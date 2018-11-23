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

#ifndef _GOCODEGEN_H
#define _GOCODEGEN_H

#include <iostream>
#include <string>
#include <stdio.h>
#include "common.h"
#include "gendata.h"

using std::string;
using std::ostream;

/* Integer array line length. */
#define IALL 8

/* Forwards. */
struct RedFsmAp;
struct RedStateAp;
struct CodeGenData;
struct GenAction;
struct NameInst;
struct GenInlineItem;
struct GenInlineList;
struct RedAction;
struct LongestMatch;
struct LongestMatchPart;

class GoCodeGen : public CodeGenData
{
public:
	GoCodeGen( ostream &out )
		: CodeGenData(out) {}

	virtual ~GoCodeGen() {}

	virtual void finishRagelDef();
	virtual void writeInit();
	virtual void writeStart();
	virtual void writeFirstFinal();
	virtual void writeError();
	virtual void writeExports();
protected:
	string FSM_NAME();
	string START_STATE_ID();
	ostream &ACTIONS_ARRAY();
	string GET_WIDE_KEY();
	string GET_WIDE_KEY( RedStateAp *state );
	string TABS( int level );
	string KEY( Key key );
	string WIDE_KEY( RedStateAp *state, Key key );
	string LDIR_PATH( char *path );
	virtual void ACTION( ostream &ret, GenAction *action, int targState,
			bool inFinish, bool csForced );
	void CONDITION( ostream &ret, GenAction *condition );
	string ALPH_TYPE();
	string WIDE_ALPH_TYPE();
	string ARRAY_TYPE( unsigned long maxVal );

	bool isAlphTypeSigned();
	bool isWideAlphTypeSigned();

	virtual string CAST( string type, string expr );
	virtual string UINT();
	virtual string INT();
	virtual string NULL_ITEM();
	virtual string GET_KEY();

	string P();
	string PE();
	string vEOF();

	string ACCESS();
	string vCS();
	string STACK();
	string TOP();
	string TOKSTART();
	string TOKEND();
	string ACT();
	string DATA();

	string DATA_PREFIX();
	string PM() { return "_" + DATA_PREFIX() + "partition_map"; }
	string C() { return "_" + DATA_PREFIX() + "cond_spaces"; }
	string CK() { return "_" + DATA_PREFIX() + "cond_keys"; }
	string K() { return "_" + DATA_PREFIX() + "trans_keys"; }
	string I() { return "_" + DATA_PREFIX() + "indicies"; }
	string CO() { return "_" + DATA_PREFIX() + "cond_offsets"; }
	string KO() { return "_" + DATA_PREFIX() + "key_offsets"; }
	string IO() { return "_" + DATA_PREFIX() + "index_offsets"; }
	string CL() { return "_" + DATA_PREFIX() + "cond_lengths"; }
	string SL() { return "_" + DATA_PREFIX() + "single_lengths"; }
	string RL() { return "_" + DATA_PREFIX() + "range_lengths"; }
	string A() { return "_" + DATA_PREFIX() + "actions"; }
	string TA() { return "_" + DATA_PREFIX() + "trans_actions"; }
	string TT() { return "_" + DATA_PREFIX() + "trans_targs"; }
	string TSA() { return "_" + DATA_PREFIX() + "to_state_actions"; }
	string FSA() { return "_" + DATA_PREFIX() + "from_state_actions"; }
	string EA() { return "_" + DATA_PREFIX() + "eof_actions"; }
	string ET() { return "_" + DATA_PREFIX() + "eof_trans"; }
	string SP() { return "_" + DATA_PREFIX() + "key_spans"; }
	string CSP() { return "_" + DATA_PREFIX() + "cond_key_spans"; }
	string START() { return DATA_PREFIX() + "start"; }
	string ERROR() { return DATA_PREFIX() + "error"; }
	string FIRST_FINAL() { return DATA_PREFIX() + "first_final"; }
	string CTXDATA() { return DATA_PREFIX() + "ctxdata"; }

	void INLINE_LIST( ostream &ret, GenInlineList *inlineList,
			int targState, bool inFinish, bool csForced );
	virtual void GOTO( ostream &ret, int gotoDest, bool inFinish ) = 0;
	virtual void CALL( ostream &ret, int callDest, int targState, bool inFinish ) = 0;
	virtual void NEXT( ostream &ret, int nextDest, bool inFinish ) = 0;
	virtual void GOTO_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish ) = 0;
	virtual void NEXT_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish ) = 0;
	virtual void CALL_EXPR( ostream &ret, GenInlineItem *ilItem,
			int targState, bool inFinish ) = 0;
	virtual void RET( ostream &ret, bool inFinish ) = 0;
	virtual void BREAK( ostream &ret, int targState, bool csForced ) = 0;
	virtual void CURS( ostream &ret, bool inFinish ) = 0;
	virtual void TARGS( ostream &ret, bool inFinish, int targState ) = 0;
	void EXEC( ostream &ret, GenInlineItem *item, int targState, int inFinish );
	void LM_SWITCH( ostream &ret, GenInlineItem *item, int targState,
			int inFinish, bool csForced );
	void SET_ACT( ostream &ret, GenInlineItem *item );
	void INIT_TOKSTART( ostream &ret, GenInlineItem *item );
	void INIT_ACT( ostream &ret, GenInlineItem *item );
	void SET_TOKSTART( ostream &ret, GenInlineItem *item );
	void SET_TOKEND( ostream &ret, GenInlineItem *item );
	void GET_TOKEND( ostream &ret, GenInlineItem *item );
	virtual void SUB_ACTION( ostream &ret, GenInlineItem *item,
			int targState, bool inFinish, bool csForced );
	void STATE_IDS();

	string ERROR_STATE();
	string FIRST_FINAL_STATE();

	virtual ostream &OPEN_ARRAY( string type, string name );
	virtual ostream &CLOSE_ARRAY();
	virtual ostream &STATIC_VAR( string type, string name );
	virtual ostream &CONST( string type, string name );

	ostream &source_warning(const InputLoc &loc);
	ostream &source_error(const InputLoc &loc);

	unsigned int arrayTypeSize( unsigned long maxVal );

	bool outLabelUsed;
	bool testEofUsed;
	bool againLabelUsed;
	bool useIndicies;

	void genLineDirective( ostream &out );

public:
	/* Determine if we should use indicies. */
	virtual void calcIndexSize() {}
};

#endif
