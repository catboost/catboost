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

#include "ragel.h"
#include "gotablish.h"

using std::endl;

void GoTablishCodeGen::GOTO( ostream &ret, int gotoDest, bool inFinish )
{
    ret << vCS() << " = " << gotoDest << endl <<
            "goto _again" << endl;
}

void GoTablishCodeGen::GOTO_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish )
{
    ret << vCS() << " = (";
    INLINE_LIST( ret, ilItem->children, 0, inFinish, false );
    ret << ")" << endl << "goto _again" << endl;
}

void GoTablishCodeGen::CURS( ostream &ret, bool inFinish )
{
    ret << "(_ps)";
}

void GoTablishCodeGen::TARGS( ostream &ret, bool inFinish, int targState )
{
    ret << "(" << vCS() << ")";
}

void GoTablishCodeGen::NEXT( ostream &ret, int nextDest, bool inFinish )
{
    ret << vCS() << " = " << nextDest << endl;
}

void GoTablishCodeGen::NEXT_EXPR( ostream &ret, GenInlineItem *ilItem, bool inFinish )
{
    ret << vCS() << " = (";
    INLINE_LIST( ret, ilItem->children, 0, inFinish, false );
    ret << ")" << endl;
}

void GoTablishCodeGen::CALL( ostream &ret, int callDest, int targState, bool inFinish )
{
    if ( prePushExpr != 0 ) {
        ret << "{ ";
        INLINE_LIST( ret, prePushExpr, 0, false, false );
    }

    ret << STACK() << "[" << TOP() << "] = " << vCS() << "; " << TOP() << "++; " <<
            vCS() << " = " << callDest << "; " << "goto _again" << endl;

    if ( prePushExpr != 0 )
        ret << " }";
}

void GoTablishCodeGen::CALL_EXPR( ostream &ret, GenInlineItem *ilItem, int targState, bool inFinish )
{
    if ( prePushExpr != 0 ) {
        ret << "{";
        INLINE_LIST( ret, prePushExpr, 0, false, false );
    }

    ret << STACK() << "[" << TOP() << "] = " << vCS() << "; " << TOP() << "++; " << vCS() << " = (";
    INLINE_LIST( ret, ilItem->children, targState, inFinish, false );
    ret << "); " << "goto _again" << endl;

    if ( prePushExpr != 0 )
        ret << "}";
}

void GoTablishCodeGen::RET( ostream &ret, bool inFinish )
{
    ret << TOP() << "--; " << vCS() << " = " << STACK() << "[" <<
            TOP() << "]" << endl;

    if ( postPopExpr != 0 ) {
        ret << "{ ";
        INLINE_LIST( ret, postPopExpr, 0, false, false );
        ret << " }" << endl;
    }

    ret << "goto _again" << endl;
}

void GoTablishCodeGen::BREAK( ostream &ret, int targState, bool csForced )
{
    outLabelUsed = true;
    ret << P() << "++; " << "goto _out" << endl;
}
