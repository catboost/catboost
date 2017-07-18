/*
 *  2007 Victor Hugo Borja <vic@rubyforge.org>
 *  Copyright 2001-2007 Adrian Thurston <thurston@complang.org>
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

#ifndef _RUBY_CODEGEN_H
#define _RUBY_CODEGEN_H

#include "common.h"
#include "gendata.h"

/* Integer array line length. */
#define IALL 8


class RubyCodeGen : public CodeGenData
{
public:
   RubyCodeGen( ostream &out ) : CodeGenData(out) { }
   virtual ~RubyCodeGen() {}
protected:
	ostream &START_ARRAY_LINE();
	ostream &ARRAY_ITEM( string item, int count, bool last );
	ostream &END_ARRAY_LINE();
  

	string FSM_NAME();

        string START_STATE_ID();
	string ERROR_STATE();
	string FIRST_FINAL_STATE();
        void INLINE_LIST(ostream &ret, GenInlineList *inlineList, int targState, bool inFinish);
        string ACCESS();

        void ACTION( ostream &ret, GenAction *action, int targState, bool inFinish );
	string GET_KEY();
        string GET_WIDE_KEY();
	string GET_WIDE_KEY( RedStateAp *state );
	string KEY( Key key );
	string TABS( int level );
	string INT( int i );
	void CONDITION( ostream &ret, GenAction *condition );
	string ALPH_TYPE();
	string WIDE_ALPH_TYPE();
  	string ARRAY_TYPE( unsigned long maxVal );
	ostream &ACTIONS_ARRAY();
	void STATE_IDS();


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

public:
	string NULL_ITEM();
	ostream &OPEN_ARRAY( string type, string name );
	ostream &CLOSE_ARRAY();
	ostream &STATIC_VAR( string type, string name );
	string ARR_OFF( string ptr, string offset );

	string P();
	string PE();
	string vEOF();

	string vCS();
	string TOP();
	string STACK();
	string ACT();
	string TOKSTART();
	string TOKEND();
	string DATA();


	void finishRagelDef();
	unsigned int arrayTypeSize( unsigned long maxVal );

protected:
	virtual void writeExports();
	virtual void writeInit();
	virtual void writeStart();
	virtual void writeFirstFinal();
	virtual void writeError();

	/* Determine if we should use indicies. */
	virtual void calcIndexSize();

	virtual void BREAK( ostream &ret, int targState ) = 0;
	virtual void GOTO( ostream &ret, int gotoDest, bool inFinish ) = 0;
	virtual void GOTO_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish ) = 0;
	virtual void CALL( ostream &ret, int callDest, int targState, bool inFinish ) = 0;
	virtual void CALL_EXPR( ostream &ret, GenInlineItem *ilItem, int targState, bool inFinish ) = 0;
	virtual void RET( ostream &ret, bool inFinish ) = 0;


	virtual void NEXT( ostream &ret, int nextDest, bool inFinish ) = 0;
	virtual void NEXT_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish ) = 0;

	virtual int TO_STATE_ACTION( RedStateAp *state ) = 0;
	virtual int FROM_STATE_ACTION( RedStateAp *state ) = 0;
	virtual int EOF_ACTION( RedStateAp *state ) = 0;

	virtual int TRANS_ACTION( RedTransAp *trans );

        void EXEC( ostream &ret, GenInlineItem *item, int targState, int inFinish );
	void LM_SWITCH( ostream &ret, GenInlineItem *item, int targState, int inFinish );
	void SET_ACT( ostream &ret, GenInlineItem *item );
	void INIT_TOKSTART( ostream &ret, GenInlineItem *item );
	void INIT_ACT( ostream &ret, GenInlineItem *item );
	void SET_TOKSTART( ostream &ret, GenInlineItem *item );
	void SET_TOKEND( ostream &ret, GenInlineItem *item );
	void GET_TOKEND( ostream &ret, GenInlineItem *item );
	void SUB_ACTION( ostream &ret, GenInlineItem *item, int targState, bool inFinish );

protected:
	ostream &source_warning(const InputLoc &loc);
	ostream &source_error(const InputLoc &loc);


        /* fields */
	bool outLabelUsed;
	bool againLabelUsed;
	bool useIndicies;

	void genLineDirective( ostream &out );
};

/*
 * Local Variables:
 * mode: c++
 * indent-tabs-mode: 1
 * c-file-style: "bsd"
 * End:
 */

#endif
