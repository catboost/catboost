/*
 *  Copyright 2004-2006 Adrian Thurston <thurston@complang.org>
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

#ifndef _CDFLAT_H
#define _CDFLAT_H

#include <iostream>
#include "cdcodegen.h"

/* Forwards. */
struct CodeGenData;
struct NameInst;
struct RedTransAp;
struct RedStateAp;

/*
 * FlatCodeGen
 */
class FlatCodeGen : virtual public FsmCodeGen
{
public:
	FlatCodeGen( ostream &out ) : FsmCodeGen(out) {}
	virtual ~FlatCodeGen() { }

protected:
	std::ostream &TO_STATE_ACTION_SWITCH();
	std::ostream &FROM_STATE_ACTION_SWITCH();
	std::ostream &EOF_ACTION_SWITCH();
	std::ostream &ACTION_SWITCH();
	std::ostream &KEYS();
	std::ostream &INDICIES();
	std::ostream &FLAT_INDEX_OFFSET();
	std::ostream &KEY_SPANS();
	std::ostream &TO_STATE_ACTIONS();
	std::ostream &FROM_STATE_ACTIONS();
	std::ostream &EOF_ACTIONS();
	std::ostream &EOF_TRANS();
	std::ostream &TRANS_TARGS();
	std::ostream &TRANS_ACTIONS();
	void LOCATE_TRANS();

	std::ostream &COND_INDEX_OFFSET();
	void COND_TRANSLATE();
	std::ostream &CONDS();
	std::ostream &COND_KEYS();
	std::ostream &COND_KEY_SPANS();

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

	virtual void writeData();
	virtual void writeExec();
};

/*
 * CFlatCodeGen
 */
struct CFlatCodeGen
	: public FlatCodeGen, public CCodeGen
{
	CFlatCodeGen( ostream &out ) : 
		FsmCodeGen(out), FlatCodeGen(out), CCodeGen(out) {}
};

/*
 * DFlatCodeGen
 */
struct DFlatCodeGen
	: public FlatCodeGen, public DCodeGen
{
	DFlatCodeGen( ostream &out ) : 
		FsmCodeGen(out), FlatCodeGen(out), DCodeGen(out) {}
};

/*
 * D2FlatCodeGen
 */
struct D2FlatCodeGen
	: public FlatCodeGen, public D2CodeGen
{
	D2FlatCodeGen( ostream &out ) : 
		FsmCodeGen(out), FlatCodeGen(out), D2CodeGen(out) {}
};

#endif
