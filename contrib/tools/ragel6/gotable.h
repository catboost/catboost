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

#ifndef _GOTABLE_H
#define _GOTABLE_H

#include <iostream>
#include "gotablish.h"

class GoTabCodeGen
    : public GoTablishCodeGen
{
public:
	GoTabCodeGen( ostream &out )
        : GoTablishCodeGen(out) {}

    virtual ~GoTabCodeGen() { }
    virtual void writeData();
    virtual void writeExec();

protected:
    std::ostream &TO_STATE_ACTION_SWITCH( int level );
    std::ostream &FROM_STATE_ACTION_SWITCH( int level );
    std::ostream &EOF_ACTION_SWITCH( int level );
    std::ostream &ACTION_SWITCH( int level );

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

    virtual std::ostream &TO_STATE_ACTION( RedStateAp *state );
    virtual std::ostream &FROM_STATE_ACTION( RedStateAp *state );
    virtual std::ostream &EOF_ACTION( RedStateAp *state );
    virtual std::ostream &TRANS_ACTION( RedTransAp *trans );
    virtual void calcIndexSize();
};

#endif
