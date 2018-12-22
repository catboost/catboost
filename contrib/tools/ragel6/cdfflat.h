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

#ifndef _CDFFLAT_H
#define _CDFFLAT_H

#include <iostream>
#include "cdflat.h"

/* Forwards. */
struct CodeGenData;

/*
 * FFlatCodeGen
 */
class FFlatCodeGen : public FlatCodeGen
{
protected:
	FFlatCodeGen( ostream &out ) : FsmCodeGen(out), FlatCodeGen(out) {}

	std::ostream &TO_STATE_ACTION_SWITCH();
	std::ostream &FROM_STATE_ACTION_SWITCH();
	std::ostream &EOF_ACTION_SWITCH();
	std::ostream &ACTION_SWITCH();

	virtual std::ostream &TO_STATE_ACTION( RedStateAp *state );
	virtual std::ostream &FROM_STATE_ACTION( RedStateAp *state );
	virtual std::ostream &EOF_ACTION( RedStateAp *state );
	virtual std::ostream &TRANS_ACTION( RedTransAp *trans );

	virtual void writeData();
	virtual void writeExec();
};

/*
 * CFFlatCodeGen
 */
struct CFFlatCodeGen
	: public FFlatCodeGen, public CCodeGen
{
	CFFlatCodeGen( ostream &out ) : 
		FsmCodeGen(out), FFlatCodeGen(out), CCodeGen(out) {}
};

/*
 * DFFlatCodeGen
 */
struct DFFlatCodeGen
	: public FFlatCodeGen, public DCodeGen
{
	DFFlatCodeGen( ostream &out ) : 
		FsmCodeGen(out), FFlatCodeGen(out), DCodeGen(out) {}
};

/*
 * D2FFlatCodeGen
 */
struct D2FFlatCodeGen
	: public FFlatCodeGen, public D2CodeGen
{
	D2FFlatCodeGen( ostream &out ) : 
		FsmCodeGen(out), FFlatCodeGen(out), D2CodeGen(out) {}
};

#endif
