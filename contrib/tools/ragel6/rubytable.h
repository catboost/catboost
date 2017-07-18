/*
 *  Copyright 2007 Victor Hugo Borja <vic@rubyforge.org>
 *            2006-2007 Adrian Thurston <thurston@complang.org>
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

#ifndef _RUBY_TABCODEGEN_H
#define _RUBY_TABCODEGEN_H

#include <iostream>
#include <string>
#include <stdio.h>
#include "common.h"
#include "gendata.h"
#include "rubycodegen.h"


using std::string;
using std::ostream;

/*
 * RubyCodeGen
 */
class RubyTabCodeGen : public RubyCodeGen
{
public:
	RubyTabCodeGen( ostream &out ) : 
          RubyCodeGen(out) {}
        virtual ~RubyTabCodeGen() {}

public:
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

 protected:
	virtual std::ostream &TO_STATE_ACTION_SWITCH();
	virtual std::ostream &FROM_STATE_ACTION_SWITCH();
	virtual std::ostream &EOF_ACTION_SWITCH();
	virtual std::ostream &ACTION_SWITCH();

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


	void NEXT( ostream &ret, int nextDest, bool inFinish );
	void NEXT_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish );

	virtual int TO_STATE_ACTION( RedStateAp *state );
	virtual int FROM_STATE_ACTION( RedStateAp *state );
	virtual int EOF_ACTION( RedStateAp *state );

private:
	string array_type;
	string array_name;

public:

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


};


#endif

/*
 * Local Variables:
 * mode: c++
 * indent-tabs-mode: 1
 * c-file-style: "bsd"
 * End:
 */
