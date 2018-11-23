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

#ifndef _CDTABLE_H
#define _CDTABLE_H

#include <iostream>
#include "cdcodegen.h"

/* Forwards. */
struct CodeGenData;
struct NameInst;
struct RedTransAp;
struct RedStateAp;

/*
 * TabCodeGen
 */
class TabCodeGen : virtual public FsmCodeGen
{
public:
	TabCodeGen( ostream &out ) : FsmCodeGen(out) {}
	virtual ~TabCodeGen() { }
	virtual void writeData();
	virtual void writeExec();

protected:
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
	void LOCATE_TRANS();

	void COND_TRANSLATE();

	void GOTO( ostream &ret, int gotoDest, bool inFinish );
	void CALL( ostream &ret, int callDest, int targState, bool inFinish );
	void NEXT( ostream &ret, int nextDest, bool inFinish );
	void GOTO_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish );
	void NEXT_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish );
	void CALL_EXPR( ostream &ret, GenInlineItem *ilItem, int targState, bool inFinish );
	void CURS( ostream &ret, bool inFinish );
	void TARGS( ostream &ret, bool inFinish, int targState );
	void RET( ostream &ret, bool inFinish );
	void BREAK( ostream &ret, int targState, bool csForced );

	virtual std::ostream &TO_STATE_ACTION( RedStateAp *state );
	virtual std::ostream &FROM_STATE_ACTION( RedStateAp *state );
	virtual std::ostream &EOF_ACTION( RedStateAp *state );
	virtual std::ostream &TRANS_ACTION( RedTransAp *trans );
	virtual void calcIndexSize();
};


/*
 * CTabCodeGen
 */
struct CTabCodeGen
	: public TabCodeGen, public CCodeGen
{
	CTabCodeGen( ostream &out ) : 
		FsmCodeGen(out), TabCodeGen(out), CCodeGen(out) {}
};

/*
 * DTabCodeGen
 */
struct DTabCodeGen
	: public TabCodeGen, public DCodeGen
{
	DTabCodeGen( ostream &out ) : 
		FsmCodeGen(out), TabCodeGen(out), DCodeGen(out) {}
};

/*
 * D2TabCodeGen
 */
struct D2TabCodeGen
	: public TabCodeGen, public D2CodeGen
{
	D2TabCodeGen( ostream &out ) : 
		FsmCodeGen(out), TabCodeGen(out), D2CodeGen(out) {}
};

#endif
