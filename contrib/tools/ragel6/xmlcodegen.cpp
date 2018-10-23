/*
 *  Copyright 2005-2007 Adrian Thurston <thurston@complang.org>
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
#include "xmlcodegen.h"
#include "parsedata.h"
#include "fsmgraph.h"
#include "gendata.h"
#include "inputdata.h"
#include <string.h>
#include "rlparse.h"
#include "version.h"

using namespace std;

GenBase::GenBase( char *fsmName, ParseData *pd, FsmAp *fsm )
:
	fsmName(fsmName),
	pd(pd),
	fsm(fsm),
	nextActionTableId(0)
{
}

void GenBase::appendTrans( TransListVect &outList, Key lowKey, 
		Key highKey, TransAp *trans )
{
	if ( trans->toState != 0 || trans->actionTable.length() > 0 )
		outList.append( TransEl( lowKey, highKey, trans ) );
}

void GenBase::reduceActionTables()
{
	/* Reduce the actions tables to a set. */
	for ( StateList::Iter st = fsm->stateList; st.lte(); st++ ) {
		RedActionTable *actionTable = 0;

		/* Reduce To State Actions. */
		if ( st->toStateActionTable.length() > 0 ) {
			if ( actionTableMap.insert( st->toStateActionTable, &actionTable ) )
				actionTable->id = nextActionTableId++;
		}

		/* Reduce From State Actions. */
		if ( st->fromStateActionTable.length() > 0 ) {
			if ( actionTableMap.insert( st->fromStateActionTable, &actionTable ) )
				actionTable->id = nextActionTableId++;
		}

		/* Reduce EOF actions. */
		if ( st->eofActionTable.length() > 0 ) {
			if ( actionTableMap.insert( st->eofActionTable, &actionTable ) )
				actionTable->id = nextActionTableId++;
		}

		/* Loop the transitions and reduce their actions. */
		for ( TransList::Iter trans = st->outList; trans.lte(); trans++ ) {
			if ( trans->actionTable.length() > 0 ) {
				if ( actionTableMap.insert( trans->actionTable, &actionTable ) )
					actionTable->id = nextActionTableId++;
			}
		}
	}
}

XMLCodeGen::XMLCodeGen( char *fsmName, ParseData *pd, FsmAp *fsm, std::ostream &out )
:
	GenBase(fsmName, pd, fsm),
	out(out)
{
}


void XMLCodeGen::writeActionList()
{
	/* Determine which actions to write. */
	int nextActionId = 0;
	for ( ActionList::Iter act = pd->actionList; act.lte(); act++ ) {
		if ( act->numRefs() > 0 || act->numCondRefs > 0 )
			act->actionId = nextActionId++;
	}

	/* Write the list. */
	out << "    <action_list length=\"" << nextActionId << "\">\n";
	for ( ActionList::Iter act = pd->actionList; act.lte(); act++ ) {
		if ( act->actionId >= 0 )
			writeAction( act );
	}
	out << "    </action_list>\n";
}

void XMLCodeGen::writeActionTableList()
{
	/* Must first order the action tables based on their id. */
	int numTables = nextActionTableId;
	RedActionTable **tables = new RedActionTable*[numTables];
	for ( ActionTableMap::Iter at = actionTableMap; at.lte(); at++ )
		tables[at->id] = at;

	out << "    <action_table_list length=\"" << numTables << "\">\n";
	for ( int t = 0; t < numTables; t++ ) {
		out << "      <action_table id=\"" << t << "\" length=\"" << 
				tables[t]->key.length() << "\">";
		for ( ActionTable::Iter atel = tables[t]->key; atel.lte(); atel++ ) {
			out << atel->value->actionId;
			if ( ! atel.last() )
				out << " ";
		}
		out << "</action_table>\n";
	}
	out << "    </action_table_list>\n";

	delete[] tables;
}

void XMLCodeGen::writeKey( Key key )
{
	if ( keyOps->isSigned )
		out << key.getVal();
	else
		out << (unsigned long) key.getVal();
}

void XMLCodeGen::writeTrans( Key lowKey, Key highKey, TransAp *trans )
{
	/* First reduce the action. */
	RedActionTable *actionTable = 0;
	if ( trans->actionTable.length() > 0 )
		actionTable = actionTableMap.find( trans->actionTable );

	/* Write the transition. */
	out << "        <t>";
	writeKey( lowKey );
	out << " ";
	writeKey( highKey );

	if ( trans->toState != 0 )
		out << " " << trans->toState->alg.stateNum;
	else
		out << " x";

	if ( actionTable != 0 )
		out << " " << actionTable->id;
	else
		out << " x";
	out << "</t>\n";
}

void XMLCodeGen::writeTransList( StateAp *state )
{
	TransListVect outList;

	/* If there is only are no ranges the task is simple. */
	if ( state->outList.length() > 0 ) {
		/* Loop each source range. */
		for ( TransList::Iter trans = state->outList; trans.lte(); trans++ ) {
			/* Reduce the transition. If it reduced to anything then add it. */
			appendTrans( outList, trans->lowKey, trans->highKey, trans );
		}
	}

	out << "      <trans_list length=\"" << outList.length() << "\">\n";
	for ( TransListVect::Iter tvi = outList; tvi.lte(); tvi++ )
		writeTrans( tvi->lowKey, tvi->highKey, tvi->value );
	out << "      </trans_list>\n";
}

void XMLCodeGen::writeEofTrans( StateAp *state )
{
	RedActionTable *eofActions = 0;
	if ( state->eofActionTable.length() > 0 )
		eofActions = actionTableMap.find( state->eofActionTable );
	
	/* The <eof_t> is used when there is an eof target, otherwise the eof
	 * action goes into state actions. */
	if ( state->eofTarget != 0 ) {
		out << "      <eof_t>" << state->eofTarget->alg.stateNum;

		if ( eofActions != 0 )
			out << " " << eofActions->id;
		else
			out << " x"; 

		out << "</eof_t>" << endl;
	}
}

void XMLCodeGen::writeText( InlineItem *item )
{
	if ( item->prev == 0 || item->prev->type != InlineItem::Text )
		out << "<text>";
	xmlEscapeHost( out, item->data, strlen(item->data) );
	if ( item->next == 0 || item->next->type != InlineItem::Text )
		out << "</text>";
}

void XMLCodeGen::writeGoto( InlineItem *item )
{
	if ( pd->generatingSectionSubset )
		out << "<goto>-1</goto>";
	else {
		EntryMapEl *targ = fsm->entryPoints.find( item->nameTarg->id );
		out << "<goto>" << targ->value->alg.stateNum << "</goto>";
	}
}

void XMLCodeGen::writeCall( InlineItem *item )
{
	if ( pd->generatingSectionSubset )
		out << "<call>-1</call>";
	else {
		EntryMapEl *targ = fsm->entryPoints.find( item->nameTarg->id );
		out << "<call>" << targ->value->alg.stateNum << "</call>";
	}
}

void XMLCodeGen::writeNext( InlineItem *item )
{
	if ( pd->generatingSectionSubset )
		out << "<next>-1</next>";
	else {
		EntryMapEl *targ = fsm->entryPoints.find( item->nameTarg->id );
		out << "<next>" << targ->value->alg.stateNum << "</next>";
	}
}

void XMLCodeGen::writeGotoExpr( InlineItem *item )
{
	out << "<goto_expr>";
	writeInlineList( item->children );
	out << "</goto_expr>";
}

void XMLCodeGen::writeCallExpr( InlineItem *item )
{
	out << "<call_expr>";
	writeInlineList( item->children );
	out << "</call_expr>";
}

void XMLCodeGen::writeNextExpr( InlineItem *item )
{
	out << "<next_expr>";
	writeInlineList( item->children );
	out << "</next_expr>";
}

void XMLCodeGen::writeEntry( InlineItem *item )
{
	if ( pd->generatingSectionSubset )
		out << "<entry>-1</entry>";
	else {
		EntryMapEl *targ = fsm->entryPoints.find( item->nameTarg->id );
		out << "<entry>" << targ->value->alg.stateNum << "</entry>";
	}
}

void XMLCodeGen::writeActionExec( InlineItem *item )
{
	out << "<exec>";
	writeInlineList( item->children );
	out << "</exec>";
}

void XMLCodeGen::writeLmOnLast( InlineItem *item )
{
	out << "<set_tokend>1</set_tokend>";

	if ( item->longestMatchPart->action != 0 ) {
		out << "<sub_action>";
		writeInlineList( item->longestMatchPart->action->inlineList );
		out << "</sub_action>";
	}
}

void XMLCodeGen::writeLmOnNext( InlineItem *item )
{
	out << "<set_tokend>0</set_tokend>";
	out << "<hold></hold>";

	if ( item->longestMatchPart->action != 0 ) {
		out << "<sub_action>";
		writeInlineList( item->longestMatchPart->action->inlineList );
		out << "</sub_action>";
	}
}

void XMLCodeGen::writeLmOnLagBehind( InlineItem *item )
{
	out << "<exec><get_tokend></get_tokend></exec>";

	if ( item->longestMatchPart->action != 0 ) {
		out << "<sub_action>";
		writeInlineList( item->longestMatchPart->action->inlineList );
		out << "</sub_action>";
	}
}

void XMLCodeGen::writeLmSwitch( InlineItem *item )
{
	LongestMatch *longestMatch = item->longestMatch;
	out << "<lm_switch>\n";

	/* We can't put the <exec> here because we may need to handle the error
	 * case and in that case p should not be changed. Instead use a default
	 * label in the switch to adjust p when user actions are not set. An id of
	 * -1 indicates the default. */

	if ( longestMatch->lmSwitchHandlesError ) {
		/* If the switch handles error then we should have also forced the
		 * error state. */
		assert( fsm->errState != 0 );

		out << "        <sub_action id=\"0\">";
		out << "<goto>" << fsm->errState->alg.stateNum << "</goto>";
		out << "</sub_action>\n";
	}
	
	bool needDefault = false;
	for ( LmPartList::Iter lmi = *longestMatch->longestMatchList; lmi.lte(); lmi++ ) {
		if ( lmi->inLmSelect ) {
			if ( lmi->action == 0 )
				needDefault = true;
			else {
				/* Open the action. Write it with the context that sets up _p 
				 * when doing control flow changes from inside the machine. */
				out << "        <sub_action id=\"" << lmi->longestMatchId << "\">";
				out << "<exec><get_tokend></get_tokend></exec>";
				writeInlineList( lmi->action->inlineList );
				out << "</sub_action>\n";
			}
		}
	}

	if ( needDefault ) {
		out << "        <sub_action id=\"-1\"><exec><get_tokend>"
				"</get_tokend></exec></sub_action>\n";
	}

	out << "    </lm_switch>";
}

void XMLCodeGen::writeInlineList( InlineList *inlineList )
{
	for ( InlineList::Iter item = *inlineList; item.lte(); item++ ) {
		switch ( item->type ) {
		case InlineItem::Text:
			writeText( item );
			break;
		case InlineItem::Goto:
			writeGoto( item );
			break;
		case InlineItem::GotoExpr:
			writeGotoExpr( item );
			break;
		case InlineItem::Call:
			writeCall( item );
			break;
		case InlineItem::CallExpr:
			writeCallExpr( item );
			break;
		case InlineItem::Next:
			writeNext( item );
			break;
		case InlineItem::NextExpr:
			writeNextExpr( item );
			break;
		case InlineItem::Break:
			out << "<break></break>";
			break;
		case InlineItem::Ret: 
			out << "<ret></ret>";
			break;
		case InlineItem::PChar:
			out << "<pchar></pchar>";
			break;
		case InlineItem::Char: 
			out << "<char></char>";
			break;
		case InlineItem::Curs: 
			out << "<curs></curs>";
			break;
		case InlineItem::Targs: 
			out << "<targs></targs>";
			break;
		case InlineItem::Entry:
			writeEntry( item );
			break;

		case InlineItem::Hold:
			out << "<hold></hold>";
			break;
		case InlineItem::Exec:
			writeActionExec( item );
			break;

		case InlineItem::LmSetActId:
			out << "<set_act>" << 
					item->longestMatchPart->longestMatchId << 
					"</set_act>";
			break;
		case InlineItem::LmSetTokEnd:
			out << "<set_tokend>1</set_tokend>";
			break;

		case InlineItem::LmOnLast:
			writeLmOnLast( item );
			break;
		case InlineItem::LmOnNext:
			writeLmOnNext( item );
			break;
		case InlineItem::LmOnLagBehind:
			writeLmOnLagBehind( item );
			break;
		case InlineItem::LmSwitch: 
			writeLmSwitch( item );
			break;

		case InlineItem::LmInitAct:
			out << "<init_act></init_act>";
			break;
		case InlineItem::LmInitTokStart:
			out << "<init_tokstart></init_tokstart>";
			break;
		case InlineItem::LmSetTokStart:
			out << "<set_tokstart></set_tokstart>";
			break;
		}
	}
}

BackendGen::BackendGen( char *fsmName, ParseData *pd, FsmAp *fsm, CodeGenData *cgd )
:
	GenBase(fsmName, pd, fsm),
	cgd(cgd)
{
}


void BackendGen::makeText( GenInlineList *outList, InlineItem *item )
{
	GenInlineItem *inlineItem = new GenInlineItem( InputLoc(), GenInlineItem::Text );
	inlineItem->data = item->data;

	outList->append( inlineItem );
}

void BackendGen::makeTargetItem( GenInlineList *outList, NameInst *nameTarg, 
		GenInlineItem::Type type )
{
	long targetState;
	if ( pd->generatingSectionSubset )
		targetState = -1;
	else {
		EntryMapEl *targ = fsm->entryPoints.find( nameTarg->id );
		targetState = targ->value->alg.stateNum;
	}

	/* Make the item. */
	GenInlineItem *inlineItem = new GenInlineItem( InputLoc(), type );
	inlineItem->targId = targetState;
	outList->append( inlineItem );
}

/* Make a sublist item with a given type. */
void BackendGen::makeSubList( GenInlineList *outList, 
		InlineList *inlineList, GenInlineItem::Type type )
{
	/* Fill the sub list. */
	GenInlineList *subList = new GenInlineList;
	makeGenInlineList( subList, inlineList );

	/* Make the item. */
	GenInlineItem *inlineItem = new GenInlineItem( InputLoc(), type );
	inlineItem->children = subList;
	outList->append( inlineItem );
}

void BackendGen::makeLmOnLast( GenInlineList *outList, InlineItem *item )
{
	makeSetTokend( outList, 1 );

	if ( item->longestMatchPart->action != 0 ) {
		makeSubList( outList, 
				item->longestMatchPart->action->inlineList, 
				GenInlineItem::SubAction );
	}
}

void BackendGen::makeLmOnNext( GenInlineList *outList, InlineItem *item )
{
	makeSetTokend( outList, 0 );
	outList->append( new GenInlineItem( InputLoc(), GenInlineItem::Hold ) );

	if ( item->longestMatchPart->action != 0 ) {
		makeSubList( outList, 
			item->longestMatchPart->action->inlineList,
			GenInlineItem::SubAction );
	}
}

void BackendGen::makeExecGetTokend( GenInlineList *outList )
{
	/* Make the Exec item. */
	GenInlineItem *execItem = new GenInlineItem( InputLoc(), GenInlineItem::Exec );
	execItem->children = new GenInlineList;

	/* Make the GetTokEnd */
	GenInlineItem *getTokend = new GenInlineItem( InputLoc(), GenInlineItem::LmGetTokEnd );
	execItem->children->append( getTokend );

	outList->append( execItem );
}

void BackendGen::makeLmOnLagBehind( GenInlineList *outList, InlineItem *item )
{
	/* Jump to the tokend. */
	makeExecGetTokend( outList );

	if ( item->longestMatchPart->action != 0 ) {
		makeSubList( outList,
			item->longestMatchPart->action->inlineList,
			GenInlineItem::SubAction );
	}
}

void BackendGen::makeLmSwitch( GenInlineList *outList, InlineItem *item )
{
	GenInlineItem *lmSwitch = new GenInlineItem( InputLoc(), GenInlineItem::LmSwitch );
	GenInlineList *lmList = lmSwitch->children = new GenInlineList;
	LongestMatch *longestMatch = item->longestMatch;

	/* We can't put the <exec> here because we may need to handle the error
	 * case and in that case p should not be changed. Instead use a default
	 * label in the switch to adjust p when user actions are not set. An id of
	 * -1 indicates the default. */

	if ( longestMatch->lmSwitchHandlesError ) {
		/* If the switch handles error then we should have also forced the
		 * error state. */
		assert( fsm->errState != 0 );

		GenInlineItem *errCase = new GenInlineItem( InputLoc(), GenInlineItem::SubAction );
		errCase->lmId = 0;
		errCase->children = new GenInlineList;

		/* Make the item. */
		GenInlineItem *gotoItem = new GenInlineItem( InputLoc(), GenInlineItem::Goto );
		gotoItem->targId = fsm->errState->alg.stateNum;
		errCase->children->append( gotoItem );

		lmList->append( errCase );
	}
	
	bool needDefault = false;
	for ( LmPartList::Iter lmi = *longestMatch->longestMatchList; lmi.lte(); lmi++ ) {
		if ( lmi->inLmSelect ) {
			if ( lmi->action == 0 )
				needDefault = true;
			else {
				/* Open the action. Write it with the context that sets up _p 
				 * when doing control flow changes from inside the machine. */
				GenInlineItem *lmCase = new GenInlineItem( InputLoc(), 
						GenInlineItem::SubAction );
				lmCase->lmId = lmi->longestMatchId;
				lmCase->children = new GenInlineList;

				makeExecGetTokend( lmCase->children );
				makeGenInlineList( lmCase->children, lmi->action->inlineList );

				lmList->append( lmCase );
			}
		}
	}

	if ( needDefault ) {
		GenInlineItem *defCase = new GenInlineItem( InputLoc(), 
				GenInlineItem::SubAction );
		defCase->lmId = -1;
		defCase->children = new GenInlineList;

		makeExecGetTokend( defCase->children );

		lmList->append( defCase );
	}

	outList->append( lmSwitch );
}

void BackendGen::makeSetTokend( GenInlineList *outList, long offset )
{
	GenInlineItem *inlineItem = new GenInlineItem( InputLoc(), GenInlineItem::LmSetTokEnd );
	inlineItem->offset = offset;
	outList->append( inlineItem );
}

void BackendGen::makeSetAct( GenInlineList *outList, long lmId )
{
	GenInlineItem *inlineItem = new GenInlineItem( InputLoc(), GenInlineItem::LmSetActId );
	inlineItem->lmId = lmId;
	outList->append( inlineItem );
}

void BackendGen::makeGenInlineList( GenInlineList *outList, InlineList *inList )
{
	for ( InlineList::Iter item = *inList; item.lte(); item++ ) {
		switch ( item->type ) {
		case InlineItem::Text:
			makeText( outList, item );
			break;
		case InlineItem::Goto:
			makeTargetItem( outList, item->nameTarg, GenInlineItem::Goto );
			break;
		case InlineItem::GotoExpr:
			makeSubList( outList, item->children, GenInlineItem::GotoExpr );
			break;
		case InlineItem::Call:
			makeTargetItem( outList, item->nameTarg, GenInlineItem::Call );
			break;
		case InlineItem::CallExpr:
			makeSubList( outList, item->children, GenInlineItem::CallExpr );
			break;
		case InlineItem::Next:
			makeTargetItem( outList, item->nameTarg, GenInlineItem::Next );
			break;
		case InlineItem::NextExpr:
			makeSubList( outList, item->children, GenInlineItem::NextExpr );
			break;
		case InlineItem::Break:
			outList->append( new GenInlineItem( InputLoc(), GenInlineItem::Break ) );
			break;
		case InlineItem::Ret: 
			outList->append( new GenInlineItem( InputLoc(), GenInlineItem::Ret ) );
			break;
		case InlineItem::PChar:
			outList->append( new GenInlineItem( InputLoc(), GenInlineItem::PChar ) );
			break;
		case InlineItem::Char: 
			outList->append( new GenInlineItem( InputLoc(), GenInlineItem::Char ) );
			break;
		case InlineItem::Curs: 
			outList->append( new GenInlineItem( InputLoc(), GenInlineItem::Curs ) );
			break;
		case InlineItem::Targs: 
			outList->append( new GenInlineItem( InputLoc(), GenInlineItem::Targs ) );
			break;
		case InlineItem::Entry:
			makeTargetItem( outList, item->nameTarg, GenInlineItem::Entry );
			break;

		case InlineItem::Hold:
			outList->append( new GenInlineItem( InputLoc(), GenInlineItem::Hold ) );
			break;
		case InlineItem::Exec:
			makeSubList( outList, item->children, GenInlineItem::Exec );
			break;

		case InlineItem::LmSetActId:
			makeSetAct( outList, item->longestMatchPart->longestMatchId );
			break;
		case InlineItem::LmSetTokEnd:
			makeSetTokend( outList, 1 );
			break;

		case InlineItem::LmOnLast:
			makeLmOnLast( outList, item );
			break;
		case InlineItem::LmOnNext:
			makeLmOnNext( outList, item );
			break;
		case InlineItem::LmOnLagBehind:
			makeLmOnLagBehind( outList, item );
			break;
		case InlineItem::LmSwitch: 
			makeLmSwitch( outList, item );
			break;

		case InlineItem::LmInitAct:
			outList->append( new GenInlineItem( InputLoc(), GenInlineItem::LmInitAct ) );
			break;
		case InlineItem::LmInitTokStart:
			outList->append( new GenInlineItem( InputLoc(), GenInlineItem::LmInitTokStart ) );
			break;
		case InlineItem::LmSetTokStart:
			outList->append( new GenInlineItem( InputLoc(), GenInlineItem::LmSetTokStart ) );
			cgd->hasLongestMatch = true;
			break;
		}
	}
}


void XMLCodeGen::writeAction( Action *action )
{
	out << "      <action id=\"" << action->actionId << "\"";
	if ( action->name != 0 ) 
		out << " name=\"" << action->name << "\"";
	out << " line=\"" << action->loc.line << "\" col=\"" << action->loc.col << "\">";
	writeInlineList( action->inlineList );
	out << "</action>\n";
}

void xmlEscapeHost( std::ostream &out, char *data, long len )
{
	char *end = data + len;
	while ( data != end ) {
		switch ( *data ) {
		case '<': out << "&lt;"; break;
		case '>': out << "&gt;"; break;
		case '&': out << "&amp;"; break;
		default: out << *data; break;
		}
		data += 1;
	}
}

void XMLCodeGen::writeStateActions( StateAp *state )
{
	RedActionTable *toStateActions = 0;
	if ( state->toStateActionTable.length() > 0 )
		toStateActions = actionTableMap.find( state->toStateActionTable );

	RedActionTable *fromStateActions = 0;
	if ( state->fromStateActionTable.length() > 0 )
		fromStateActions = actionTableMap.find( state->fromStateActionTable );

	/* EOF actions go out here only if the state has no eof target. If it has
	 * an eof target then an eof transition will be used instead. */
	RedActionTable *eofActions = 0;
	if ( state->eofTarget == 0 && state->eofActionTable.length() > 0 )
		eofActions = actionTableMap.find( state->eofActionTable );
	
	if ( toStateActions != 0 || fromStateActions != 0 || eofActions != 0 ) {
		out << "      <state_actions>";
		if ( toStateActions != 0 )
			out << toStateActions->id;
		else
			out << "x";

		if ( fromStateActions != 0 )
			out << " " << fromStateActions->id;
		else
			out << " x";

		if ( eofActions != 0 )
			out << " " << eofActions->id;
		else
			out << " x";

		out << "</state_actions>\n";
	}
}

void XMLCodeGen::writeStateConditions( StateAp *state )
{
	if ( state->stateCondList.length() > 0 ) {
		out << "      <cond_list length=\"" << state->stateCondList.length() << "\">\n";
		for ( StateCondList::Iter scdi = state->stateCondList; scdi.lte(); scdi++ ) {
			out << "        <c>";
			writeKey( scdi->lowKey );
			out << " ";
			writeKey( scdi->highKey );
			out << " ";
			out << scdi->condSpace->condSpaceId;
			out << "</c>\n";
		}
		out << "      </cond_list>\n";
	}
}

void XMLCodeGen::writeStateList()
{
	/* Write the list of states. */
	out << "    <state_list length=\"" << fsm->stateList.length() << "\">\n";
	for ( StateList::Iter st = fsm->stateList; st.lte(); st++ ) {
		out << "      <state id=\"" << st->alg.stateNum << "\"";
		if ( st->isFinState() )
			out << " final=\"t\"";
		out << ">\n";

		writeStateActions( st );
		writeEofTrans( st );
		writeStateConditions( st );
		writeTransList( st );

		out << "      </state>\n";

		if ( !st.last() )
			out << "\n";
	}
	out << "    </state_list>\n";
}

bool XMLCodeGen::writeNameInst( NameInst *nameInst )
{
	bool written = false;
	if ( nameInst->parent != 0 )
		written = writeNameInst( nameInst->parent );
	
	if ( nameInst->name != 0 ) {
		if ( written )
			out << '_';
		out << nameInst->name;
		written = true;
	}

	return written;
}

void XMLCodeGen::writeEntryPoints()
{
	/* List of entry points other than start state. */
	if ( fsm->entryPoints.length() > 0 || pd->lmRequiresErrorState ) {
		out << "    <entry_points";
		if ( pd->lmRequiresErrorState )
			out << " error=\"t\"";
		out << ">\n";
		for ( EntryMap::Iter en = fsm->entryPoints; en.lte(); en++ ) {
			/* Get the name instantiation from nameIndex. */
			NameInst *nameInst = pd->nameIndex[en->key];
			StateAp *state = en->value;
			out << "      <entry name=\"";
			writeNameInst( nameInst );
			out << "\">" << state->alg.stateNum << "</entry>\n";
		}
		out << "    </entry_points>\n";
	}
}

void XMLCodeGen::writeMachine()
{
	/* Open the machine. */
	out << "  <machine>\n"; 
	
	/* Action tables. */
	reduceActionTables();

	writeActionList();
	writeActionTableList();
	writeConditions();

	/* Start state. */
	out << "    <start_state>" << fsm->startState->alg.stateNum << 
			"</start_state>\n";
	
	/* Error state. */
	if ( fsm->errState != 0 ) {
		out << "    <error_state>" << fsm->errState->alg.stateNum << 
			"</error_state>\n";
	}

	writeEntryPoints();
	writeStateList();

	out << "  </machine>\n";
}


void XMLCodeGen::writeConditions()
{
	if ( condData->condSpaceMap.length() > 0 ) {
		long nextCondSpaceId = 0;
		for ( CondSpaceMap::Iter cs = condData->condSpaceMap; cs.lte(); cs++ )
			cs->condSpaceId = nextCondSpaceId++;

		out << "    <cond_space_list length=\"" << condData->condSpaceMap.length() << "\">\n";
		for ( CondSpaceMap::Iter cs = condData->condSpaceMap; cs.lte(); cs++ ) {
			out << "      <cond_space id=\"" << cs->condSpaceId << 
				"\" length=\"" << cs->condSet.length() << "\">";
			writeKey( cs->baseKey );
			for ( CondSet::Iter csi = cs->condSet; csi.lte(); csi++ )
				out << " " << (*csi)->actionId;
			out << "</cond_space>\n";
		}
		out << "    </cond_space_list>\n";
	}
}

void XMLCodeGen::writeExports()
{
	if ( pd->exportList.length() > 0 ) {
		out << "  <exports>\n";
		for ( ExportList::Iter exp = pd->exportList; exp.lte(); exp++ ) {
			out << "    <ex name=\"" << exp->name << "\">";
			writeKey( exp->key );
			out << "</ex>\n";
		}
		out << "  </exports>\n";
	}
}

void XMLCodeGen::writeXML()
{
	/* Open the definition. */
	out << "<ragel_def name=\"" << fsmName << "\">\n";

	/* Alphabet type. */
	out << "  <alphtype>" << keyOps->alphType->internalName << "</alphtype>\n";
	
	/* Getkey expression. */
	if ( pd->getKeyExpr != 0 ) {
		out << "  <getkey>";
		writeInlineList( pd->getKeyExpr );
		out << "</getkey>\n";
	}

	/* Access expression. */
	if ( pd->accessExpr != 0 ) {
		out << "  <access>";
		writeInlineList( pd->accessExpr );
		out << "</access>\n";
	}

	/* PrePush expression. */
	if ( pd->prePushExpr != 0 ) {
		out << "  <prepush>";
		writeInlineList( pd->prePushExpr );
		out << "</prepush>\n";
	}

	/* PostPop expression. */
	if ( pd->postPopExpr != 0 ) {
		out << "  <postpop>";
		writeInlineList( pd->postPopExpr );
		out << "</postpop>\n";
	}

	/*
	 * Variable expressions.
	 */

	if ( pd->pExpr != 0 ) {
		out << "  <p_expr>";
		writeInlineList( pd->pExpr );
		out << "</p_expr>\n";
	}
	
	if ( pd->peExpr != 0 ) {
		out << "  <pe_expr>";
		writeInlineList( pd->peExpr );
		out << "</pe_expr>\n";
	}

	if ( pd->eofExpr != 0 ) {
		out << "  <eof_expr>";
		writeInlineList( pd->eofExpr );
		out << "</eof_expr>\n";
	}
	
	if ( pd->csExpr != 0 ) {
		out << "  <cs_expr>";
		writeInlineList( pd->csExpr );
		out << "</cs_expr>\n";
	}
	
	if ( pd->topExpr != 0 ) {
		out << "  <top_expr>";
		writeInlineList( pd->topExpr );
		out << "</top_expr>\n";
	}
	
	if ( pd->stackExpr != 0 ) {
		out << "  <stack_expr>";
		writeInlineList( pd->stackExpr );
		out << "</stack_expr>\n";
	}
	
	if ( pd->actExpr != 0 ) {
		out << "  <act_expr>";
		writeInlineList( pd->actExpr );
		out << "</act_expr>\n";
	}
	
	if ( pd->tokstartExpr != 0 ) {
		out << "  <tokstart_expr>";
		writeInlineList( pd->tokstartExpr );
		out << "</tokstart_expr>\n";
	}
	
	if ( pd->tokendExpr != 0 ) {
		out << "  <tokend_expr>";
		writeInlineList( pd->tokendExpr );
		out << "</tokend_expr>\n";
	}
	
	if ( pd->dataExpr != 0 ) {
		out << "  <data_expr>";
		writeInlineList( pd->dataExpr );
		out << "</data_expr>\n";
	}
	
	writeExports();
	
	writeMachine();

	out <<
		"</ragel_def>\n";
}

void BackendGen::makeExports()
{
	for ( ExportList::Iter exp = pd->exportList; exp.lte(); exp++ )
		cgd->exportList.append( new Export( exp->name, exp->key ) );
}

void BackendGen::makeAction( Action *action )
{
	GenInlineList *genList = new GenInlineList;
	makeGenInlineList( genList, action->inlineList );

	cgd->newAction( curAction++, action->name, action->loc, genList );
}


void BackendGen::makeActionList()
{
	/* Determine which actions to write. */
	int nextActionId = 0;
	for ( ActionList::Iter act = pd->actionList; act.lte(); act++ ) {
		if ( act->numRefs() > 0 || act->numCondRefs > 0 )
			act->actionId = nextActionId++;
	}

	/* Write the list. */
	cgd->initActionList( nextActionId );
	curAction = 0;

	for ( ActionList::Iter act = pd->actionList; act.lte(); act++ ) {
		if ( act->actionId >= 0 )
			makeAction( act );
	}
}

void BackendGen::makeActionTableList()
{
	/* Must first order the action tables based on their id. */
	int numTables = nextActionTableId;
	RedActionTable **tables = new RedActionTable*[numTables];
	for ( ActionTableMap::Iter at = actionTableMap; at.lte(); at++ )
		tables[at->id] = at;

	cgd->initActionTableList( numTables );
	curActionTable = 0;

	for ( int t = 0; t < numTables; t++ ) {
		long length = tables[t]->key.length();

		/* Collect the action table. */
		RedAction *redAct = cgd->allActionTables + curActionTable;
		redAct->actListId = curActionTable;
		redAct->key.setAsNew( length );

		for ( ActionTable::Iter atel = tables[t]->key; atel.lte(); atel++ ) {
			redAct->key[atel.pos()].key = 0;
			redAct->key[atel.pos()].value = cgd->allActions + 
					atel->value->actionId;
		}

		/* Insert into the action table map. */
		cgd->redFsm->actionMap.insert( redAct );

		curActionTable += 1;
	}

	delete[] tables;
}

void BackendGen::makeConditions()
{
	if ( condData->condSpaceMap.length() > 0 ) {
		long nextCondSpaceId = 0;
		for ( CondSpaceMap::Iter cs = condData->condSpaceMap; cs.lte(); cs++ )
			cs->condSpaceId = nextCondSpaceId++;

		long listLength = condData->condSpaceMap.length();
		cgd->initCondSpaceList( listLength );
		curCondSpace = 0;

		for ( CondSpaceMap::Iter cs = condData->condSpaceMap; cs.lte(); cs++ ) {
			long id = cs->condSpaceId;
			cgd->newCondSpace( curCondSpace, id, cs->baseKey );
			for ( CondSet::Iter csi = cs->condSet; csi.lte(); csi++ )
				cgd->condSpaceItem( curCondSpace, (*csi)->actionId );
			curCondSpace += 1;
		}
	}
}

bool BackendGen::makeNameInst( std::string &res, NameInst *nameInst )
{
	bool written = false;
	if ( nameInst->parent != 0 )
		written = makeNameInst( res, nameInst->parent );
	
	if ( nameInst->name != 0 ) {
		if ( written )
			res += '_';
		res += nameInst->name;
		written = true;
	}

	return written;
}

void BackendGen::makeEntryPoints()
{
	/* List of entry points other than start state. */
	if ( fsm->entryPoints.length() > 0 || pd->lmRequiresErrorState ) {
		if ( pd->lmRequiresErrorState )
			cgd->setForcedErrorState();

		for ( EntryMap::Iter en = fsm->entryPoints; en.lte(); en++ ) {
			/* Get the name instantiation from nameIndex. */
			NameInst *nameInst = pd->nameIndex[en->key];
			std::string name;
			makeNameInst( name, nameInst );
			StateAp *state = en->value;
			cgd->addEntryPoint( strdup(name.c_str()), state->alg.stateNum );
		}
	}
}

void BackendGen::makeStateActions( StateAp *state )
{
	RedActionTable *toStateActions = 0;
	if ( state->toStateActionTable.length() > 0 )
		toStateActions = actionTableMap.find( state->toStateActionTable );

	RedActionTable *fromStateActions = 0;
	if ( state->fromStateActionTable.length() > 0 )
		fromStateActions = actionTableMap.find( state->fromStateActionTable );

	/* EOF actions go out here only if the state has no eof target. If it has
	 * an eof target then an eof transition will be used instead. */
	RedActionTable *eofActions = 0;
	if ( state->eofTarget == 0 && state->eofActionTable.length() > 0 )
		eofActions = actionTableMap.find( state->eofActionTable );
	
	if ( toStateActions != 0 || fromStateActions != 0 || eofActions != 0 ) {
		long to = -1;
		if ( toStateActions != 0 )
			to = toStateActions->id;

		long from = -1;
		if ( fromStateActions != 0 )
			from = fromStateActions->id;

		long eof = -1;
		if ( eofActions != 0 )
			eof = eofActions->id;

		cgd->setStateActions( curState, to, from, eof );
	}
}

void BackendGen::makeEofTrans( StateAp *state )
{
	RedActionTable *eofActions = 0;
	if ( state->eofActionTable.length() > 0 )
		eofActions = actionTableMap.find( state->eofActionTable );
	
	/* The EOF trans is used when there is an eof target, otherwise the eof
	 * action goes into state actions. */
	if ( state->eofTarget != 0 ) {
		long targ = state->eofTarget->alg.stateNum;
		long action = -1;
		if ( eofActions != 0 )
			action = eofActions->id;

		cgd->setEofTrans( curState, targ, action );
	}
}

void BackendGen::makeStateConditions( StateAp *state )
{
	if ( state->stateCondList.length() > 0 ) {
		long length = state->stateCondList.length();
		cgd->initStateCondList( curState, length );
		curStateCond = 0;

		for ( StateCondList::Iter scdi = state->stateCondList; scdi.lte(); scdi++ ) {
			cgd->addStateCond( curState, scdi->lowKey, scdi->highKey, 
					scdi->condSpace->condSpaceId );
		}
	}
}

void BackendGen::makeTrans( Key lowKey, Key highKey, TransAp *trans )
{
	/* First reduce the action. */
	RedActionTable *actionTable = 0;
	if ( trans->actionTable.length() > 0 )
		actionTable = actionTableMap.find( trans->actionTable );

	long targ = -1;
	if ( trans->toState != 0 )
		targ = trans->toState->alg.stateNum;

	long action = -1;
	if ( actionTable != 0 )
		action = actionTable->id;

	cgd->newTrans( curState, curTrans++, lowKey, highKey, targ, action );
}

void BackendGen::makeTransList( StateAp *state )
{
	TransListVect outList;

	/* If there is only are no ranges the task is simple. */
	if ( state->outList.length() > 0 ) {
		/* Loop each source range. */
		for ( TransList::Iter trans = state->outList; trans.lte(); trans++ ) {
			/* Reduce the transition. If it reduced to anything then add it. */
			appendTrans( outList, trans->lowKey, trans->highKey, trans );
		}
	}

	cgd->initTransList( curState, outList.length() );
	curTrans = 0;

	for ( TransListVect::Iter tvi = outList; tvi.lte(); tvi++ )
		makeTrans( tvi->lowKey, tvi->highKey, tvi->value );

	cgd->finishTransList( curState );
}


void BackendGen::makeStateList()
{
	/* Write the list of states. */
	long length = fsm->stateList.length();
	cgd->initStateList( length );
	curState = 0;
	for ( StateList::Iter st = fsm->stateList; st.lte(); st++ ) {
		makeStateActions( st );
		makeEofTrans( st );
		makeStateConditions( st );
		makeTransList( st );

		long id = st->alg.stateNum;
		cgd->setId( curState, id );

		if ( st->isFinState() )
			cgd->setFinal( curState );

		curState += 1;
	}
}


void BackendGen::makeMachine()
{
	cgd->createMachine();

	/* Action tables. */
	reduceActionTables();

	makeActionList();
	makeActionTableList();
	makeConditions();

	/* Start State. */
	cgd->setStartState( fsm->startState->alg.stateNum );

	/* Error state. */
	if ( fsm->errState != 0 )
		cgd->setErrorState( fsm->errState->alg.stateNum );

	makeEntryPoints();
	makeStateList();

	cgd->closeMachine();
}

void BackendGen::close_ragel_def()
{
	/* Do this before distributing transitions out to singles and defaults
	 * makes life easier. */
	cgd->redFsm->maxKey = cgd->findMaxKey();

	cgd->redFsm->assignActionLocs();

	/* Find the first final state (The final state with the lowest id). */
	cgd->redFsm->findFirstFinState();

	/* Call the user's callback. */
	cgd->finishRagelDef();
}


void BackendGen::makeBackend()
{
	/* Alphabet type. */
	cgd->setAlphType( keyOps->alphType->internalName );
	
	/* Getkey expression. */
	if ( pd->getKeyExpr != 0 ) {
		cgd->getKeyExpr = new GenInlineList;
		makeGenInlineList( cgd->getKeyExpr, pd->getKeyExpr );
	}

	/* Access expression. */
	if ( pd->accessExpr != 0 ) {
		cgd->accessExpr = new GenInlineList;
		makeGenInlineList( cgd->accessExpr, pd->accessExpr );
	}

	/* PrePush expression. */
	if ( pd->prePushExpr != 0 ) {
		cgd->prePushExpr = new GenInlineList;
		makeGenInlineList( cgd->prePushExpr, pd->prePushExpr );
	}

	/* PostPop expression. */
	if ( pd->postPopExpr != 0 ) {
		cgd->postPopExpr = new GenInlineList;
		makeGenInlineList( cgd->postPopExpr, pd->postPopExpr );
	}

	/*
	 * Variable expressions.
	 */

	if ( pd->pExpr != 0 ) {
		cgd->pExpr = new GenInlineList;
		makeGenInlineList( cgd->pExpr, pd->pExpr );
	}
	
	if ( pd->peExpr != 0 ) {
		cgd->peExpr = new GenInlineList;
		makeGenInlineList( cgd->peExpr, pd->peExpr );
	}

	if ( pd->eofExpr != 0 ) {
		cgd->eofExpr = new GenInlineList;
		makeGenInlineList( cgd->eofExpr, pd->eofExpr );
	}
	
	if ( pd->csExpr != 0 ) {
		cgd->csExpr = new GenInlineList;
		makeGenInlineList( cgd->csExpr, pd->csExpr );
	}
	
	if ( pd->topExpr != 0 ) {
		cgd->topExpr = new GenInlineList;
		makeGenInlineList( cgd->topExpr, pd->topExpr );
	}
	
	if ( pd->stackExpr != 0 ) {
		cgd->stackExpr = new GenInlineList;
		makeGenInlineList( cgd->stackExpr, pd->stackExpr );
	}
	
	if ( pd->actExpr != 0 ) {
		cgd->actExpr = new GenInlineList;
		makeGenInlineList( cgd->actExpr, pd->actExpr );
	}
	
	if ( pd->tokstartExpr != 0 ) {
		cgd->tokstartExpr = new GenInlineList;
		makeGenInlineList( cgd->tokstartExpr, pd->tokstartExpr );
	}
	
	if ( pd->tokendExpr != 0 ) {
		cgd->tokendExpr = new GenInlineList;
		makeGenInlineList( cgd->tokendExpr, pd->tokendExpr );
	}
	
	if ( pd->dataExpr != 0 ) {
		cgd->dataExpr = new GenInlineList;
		makeGenInlineList( cgd->dataExpr, pd->dataExpr );
	}
	
	makeExports();
	makeMachine();

	close_ragel_def();
}

void InputData::writeLanguage( std::ostream &out )
{
	out << " lang=\"";
	switch ( hostLang->lang ) {
		case HostLang::C:    out << "C"; break;
		case HostLang::D:    out << "D"; break;
		case HostLang::D2:    out << "D2"; break;
		case HostLang::Go:   out << "Go"; break;
		case HostLang::Java: out << "Java"; break;
		case HostLang::Ruby: out << "Ruby"; break;
		case HostLang::CSharp: out << "C#"; break;
		case HostLang::OCaml: out << "OCaml"; break;
	}
	out << "\"";
}

void InputData::writeXML( std::ostream &out )
{
	out << "<ragel version=\"" VERSION "\" filename=\"" << inputFileName << "\"";
	writeLanguage( out );
	out << ">\n";

	for ( ParserDict::Iter parser = parserDict; parser.lte(); parser++ ) {
		ParseData *pd = parser->value->pd;
		if ( pd->instanceList.length() > 0 )
			pd->generateXML( *outStream );
	}

	out << "</ragel>\n";
}
