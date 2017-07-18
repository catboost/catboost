/*
 *  Copyright 2006-2007 Adrian Thurston <thurston@complang.org>
 *            2007 Colin Fleming <colin.fleming@caverock.com>
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

#ifndef _JAVACODEGEN_H
#define _JAVACODEGEN_H

#include <iostream>
#include <string>
#include <stdio.h>
#include "common.h"
#include "gendata.h"

using std::string;
using std::ostream;

/*
 * JavaTabCodeGen
 */
struct JavaTabCodeGen : public CodeGenData
{
	JavaTabCodeGen( ostream &out ) : 
		CodeGenData(out) {}

	std::ostream &TO_STATE_ACTION_SWITCH();
	std::ostream &FROM_STATE_ACTION_SWITCH();
	std::ostream &EOF_ACTION_SWITCH();
	std::ostream &ACTION_SWITCH();

	std::ostream &COND_KEYS();
	std::ostream &COND_SPACES();
	std::ostream &KEYS();
	std::ostream &INDICIES();
	std::ostream &COND_OFFSETS();
	std::ostream &KEY_OFFSETS();
	std::ostream &INDEX_OFFSETS();
	std::ostream &COND_LENS();
	std::ostream &SINGLE_LENS();
	std::ostream &RANGE_LENS();
	std::ostream &TO_STATE_ACTIONS();
	std::ostream &FROM_STATE_ACTIONS();
	std::ostream &EOF_ACTIONS();
	std::ostream &EOF_TRANS();
	std::ostream &TRANS_TARGS();
	std::ostream &TRANS_ACTIONS();
	std::ostream &TRANS_TARGS_WI();
	std::ostream &TRANS_ACTIONS_WI();

	void BREAK( ostream &ret, int targState );
	void GOTO( ostream &ret, int gotoDest, bool inFinish );
	void GOTO_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish );
	void CALL( ostream &ret, int callDest, int targState, bool inFinish );
	void CALL_EXPR( ostream &ret, GenInlineItem *ilItem, int targState, bool inFinish );
	void RET( ostream &ret, bool inFinish );

	void COND_TRANSLATE();
	void LOCATE_TRANS();

	virtual void writeExec();
	virtual void writeData();
	virtual void writeInit();
	virtual void writeExports();
	virtual void writeStart();
	virtual void writeFirstFinal();
	virtual void writeError();
	virtual void finishRagelDef();

	void NEXT( ostream &ret, int nextDest, bool inFinish );
	void NEXT_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish );

	int TO_STATE_ACTION( RedStateAp *state );
	int FROM_STATE_ACTION( RedStateAp *state );
	int EOF_ACTION( RedStateAp *state );
	int TRANS_ACTION( RedTransAp *trans );

	/* Determine if we should use indicies. */
	void calcIndexSize();

private:
	string array_type;
	string array_name;
	int item_count;
	int div_count;

public:

	virtual string NULL_ITEM();
	virtual ostream &OPEN_ARRAY( string type, string name );
	virtual ostream &ARRAY_ITEM( string item, bool last );
	virtual ostream &CLOSE_ARRAY();
	virtual ostream &STATIC_VAR( string type, string name );
	virtual string ARR_OFF( string ptr, string offset );
	virtual string CAST( string type );
	virtual string GET_KEY();
	virtual string CTRL_FLOW();

	string FSM_NAME();
	string START_STATE_ID();
	ostream &ACTIONS_ARRAY();
	string GET_WIDE_KEY();
	string GET_WIDE_KEY( RedStateAp *state );
	string TABS( int level );
	string KEY( Key key );
	string INT( int i );
	void ACTION( ostream &ret, GenAction *action, int targState, bool inFinish );
	void CONDITION( ostream &ret, GenAction *condition );
	string ALPH_TYPE();
	string WIDE_ALPH_TYPE();
	string ARRAY_TYPE( unsigned long maxVal );

	string ACCESS();

	string P();
	string PE();
	string vEOF();

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

	void INLINE_LIST( ostream &ret, GenInlineList *inlineList, int targState, bool inFinish );
	void EXEC( ostream &ret, GenInlineItem *item, int targState, int inFinish );
	void EXECTE( ostream &ret, GenInlineItem *item, int targState, int inFinish );
	void LM_SWITCH( ostream &ret, GenInlineItem *item, int targState, int inFinish );
	void SET_ACT( ostream &ret, GenInlineItem *item );
	void INIT_TOKSTART( ostream &ret, GenInlineItem *item );
	void INIT_ACT( ostream &ret, GenInlineItem *item );
	void SET_TOKSTART( ostream &ret, GenInlineItem *item );
	void SET_TOKEND( ostream &ret, GenInlineItem *item );
	void GET_TOKEND( ostream &ret, GenInlineItem *item );
	void SUB_ACTION( ostream &ret, GenInlineItem *item, 
			int targState, bool inFinish );

	string ERROR_STATE();
	string FIRST_FINAL_STATE();

	ostream &source_warning(const InputLoc &loc);
	ostream &source_error(const InputLoc &loc);

	unsigned int arrayTypeSize( unsigned long maxVal );

	bool outLabelUsed;
	bool againLabelUsed;
	bool useIndicies;

	void genLineDirective( ostream &out );
};

#endif
