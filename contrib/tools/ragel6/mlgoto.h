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

#ifndef _MLGOTO_H
#define _MLGOTO_H

#include <iostream>
#include "mlcodegen.h"

/* Forwards. */
//struct CodeGenData;
//struct NameInst;
//struct RedTransAp;
//struct RedStateAp;
//struct GenStateCond;

/*
 * OCamlGotoCodeGen
 */
class OCamlGotoCodeGen : virtual public OCamlCodeGen
{
public:
	OCamlGotoCodeGen( ostream &out ) : OCamlCodeGen(out) {}
	std::ostream &TO_STATE_ACTION_SWITCH();
	std::ostream &FROM_STATE_ACTION_SWITCH();
	std::ostream &EOF_ACTION_SWITCH();
	std::ostream &ACTION_SWITCH();
	std::ostream &STATE_GOTOS();
	std::ostream &TRANSITIONS();
	std::ostream &EXEC_FUNCS();
	std::ostream &FINISH_CASES();

	void GOTO( ostream &ret, int gotoDest, bool inFinish );
	void CALL( ostream &ret, int callDest, int targState, bool inFinish );
	void NEXT( ostream &ret, int nextDest, bool inFinish );
	void GOTO_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish );
	void NEXT_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish );
	void CALL_EXPR( ostream &ret, GenInlineItem *ilItem, int targState, bool inFinish );
	void CURS( ostream &ret, bool inFinish );
	void TARGS( ostream &ret, bool inFinish, int targState );
	void RET( ostream &ret, bool inFinish );
	void BREAK( ostream &ret, int targState );

	virtual unsigned int TO_STATE_ACTION( RedStateAp *state );
	virtual unsigned int FROM_STATE_ACTION( RedStateAp *state );
	virtual unsigned int EOF_ACTION( RedStateAp *state );

	std::ostream &TO_STATE_ACTIONS();
	std::ostream &FROM_STATE_ACTIONS();
	std::ostream &EOF_ACTIONS();

	void COND_TRANSLATE( GenStateCond *stateCond, int level );
	void emitCondBSearch( RedStateAp *state, int level, int low, int high );
	void STATE_CONDS( RedStateAp *state, bool genDefault ); 

	virtual std::ostream &TRANS_GOTO( RedTransAp *trans, int level );

	void emitSingleSwitch( RedStateAp *state );
	void emitRangeBSearch( RedStateAp *state, int level, int low, int high, RedTransAp* def );

	/* Called from STATE_GOTOS just before writing the gotos */
	virtual void GOTO_HEADER( RedStateAp *state );
	virtual void STATE_GOTO_ERROR();

	virtual void writeData();
	virtual void writeExec();
};

#endif
