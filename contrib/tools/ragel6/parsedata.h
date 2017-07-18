/*
 *  Copyright 2001-2008 Adrian Thurston <thurston@complang.org>
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

#ifndef _PARSEDATA_H
#define _PARSEDATA_H

#include <iostream>
#include <limits.h>
#include <sstream>
#include "avlmap.h"
#include "bstmap.h"
#include "vector.h"
#include "dlist.h"
#include "fsmgraph.h"
#include "compare.h"
#include "vector.h"
#include "common.h"
#include "parsetree.h"

/* Forwards. */
using std::ostream;

struct VarDef;
struct Join;
struct Expression;
struct Term;
struct FactorWithAug;
struct FactorWithLabel;
struct FactorWithRep;
struct FactorWithNeg;
struct Factor;
struct Literal;
struct Range;
struct RegExpr;
struct ReItem;
struct ReOrBlock;
struct ReOrItem;
struct LongestMatch;
struct InputData;
struct CodeGenData;
typedef DList<LongestMatch> LmList;


/* Graph dictionary. */
struct GraphDictEl 
:
	public AvlTreeEl<GraphDictEl>,
	public DListEl<GraphDictEl>
{
	GraphDictEl( const char *k ) 
		: key(k), value(0), isInstance(false) { }
	GraphDictEl( const char *k, VarDef *value ) 
		: key(k), value(value), isInstance(false) { }

	const char *getKey() { return key; }

	const char *key;
	VarDef *value;
	bool isInstance;

	/* Location info of graph definition. Points to variable name of assignment. */
	InputLoc loc;
};

typedef AvlTree<GraphDictEl, const char*, CmpStr> GraphDict;
typedef DList<GraphDictEl> GraphList;

/* Priority name dictionary. */
typedef AvlMapEl<char*, int> PriorDictEl;
typedef AvlMap<char*, int, CmpStr> PriorDict;

/* Local error name dictionary. */
typedef AvlMapEl<const char*, int> LocalErrDictEl;
typedef AvlMap<const char*, int, CmpStr> LocalErrDict;

/* Tree of instantiated names. */
typedef BstMapEl<const char*, NameInst*> NameMapEl;
typedef BstMap<const char*, NameInst*, CmpStr> NameMap;
typedef Vector<NameInst*> NameVect;
typedef BstSet<NameInst*> NameSet;

/* Node in the tree of instantiated names. */
struct NameInst
{
	NameInst( const InputLoc &loc, NameInst *parent, const char *name, int id, bool isLabel ) : 
		loc(loc), parent(parent), name(name), id(id), isLabel(isLabel),
		isLongestMatch(false), numRefs(0), numUses(0), start(0), final(0) {}

	InputLoc loc;

	/* Keep parent pointers in the name tree to retrieve 
	 * fully qulified names. */
	NameInst *parent;

	const char *name;
	int id;
	bool isLabel;
	bool isLongestMatch;

	int numRefs;
	int numUses;

	/* Names underneath us, excludes anonymous names. */
	NameMap children;

	/* All names underneath us in order of appearance. */
	NameVect childVect;

	/* Join scopes need an implicit "final" target. */
	NameInst *start, *final;

	/* During a fsm generation walk, lists the names that are referenced by
	 * epsilon operations in the current scope. After the link is made by the
	 * epsilon reference and the join operation is complete, the label can
	 * have its refcount decremented. Once there are no more references the
	 * entry point can be removed from the fsm returned. */
	NameVect referencedNames;

	/* Pointers for the name search queue. */
	NameInst *prev, *next;

	/* Check if this name inst or any name inst below is referenced. */
	bool anyRefsRec();
};

typedef DList<NameInst> NameInstList;

/* Stack frame used in walking the name tree. */
struct NameFrame 
{
	NameInst *prevNameInst;
	int prevNameChild;
	NameInst *prevLocalScope;
};

struct LengthDef
{
	LengthDef( char *name )
		: name(name) {}

	char *name;
	LengthDef *prev, *next;
};

typedef DList<LengthDef> LengthDefList;

/* Class to collect information about the machine during the 
 * parse of input. */
struct ParseData
{
	/* Create a new parse data object. This is done at the beginning of every
	 * fsm specification. */
	ParseData( const char *fileName, char *sectionName, const InputLoc &sectionLoc );
	~ParseData();

	/*
	 * Setting up the graph dict.
	 */

	/* Initialize a graph dict with the basic fsms. */
	void initGraphDict();
	void createBuiltin( const char *name, BuiltinMachine builtin );

	/* Make a name id in the current name instantiation scope if it is not
	 * already there. */
	NameInst *addNameInst( const InputLoc &loc, const char *data, bool isLabel );
	void makeRootNames();
	void makeNameTree( GraphDictEl *gdNode );
	void makeExportsNameTree();
	void fillNameIndex( NameInst *from );
	void printNameTree();

	/* Increments the usage count on entry names. Names that are no longer
	 * needed will have their entry points unset. */
	void unsetObsoleteEntries( FsmAp *graph );

	/* Resove name references in action code and epsilon transitions. */
	NameSet resolvePart( NameInst *refFrom, const char *data, bool recLabelsOnly );
	void resolveFrom( NameSet &result, NameInst *refFrom, 
			const NameRef &nameRef, int namePos );
	NameInst *resolveStateRef( const NameRef &nameRef, InputLoc &loc, Action *action );
	void resolveNameRefs( InlineList *inlineList, Action *action );
	void resolveActionNameRefs();

	/* Set the alphabet type. If type types are not valid returns false. */
	bool setAlphType( const InputLoc &loc, char *s1, char *s2 );
	bool setAlphType( const InputLoc &loc, char *s1 );

	/* Override one of the variables ragel uses. */
	bool setVariable( char *var, InlineList *inlineList );

	/* Unique actions. */
	void removeDups( ActionTable &actionTable );
	void removeActionDups( FsmAp *graph );

	/* Dumping the name instantiation tree. */
	void printNameInst( NameInst *nameInst, int level );

	/* Make the graph from a graph dict node. Does minimization. */
	FsmAp *makeInstance( GraphDictEl *gdNode );
	FsmAp *makeSpecific( GraphDictEl *gdNode );
	FsmAp *makeAll();

	/* Checking the contents of actions. */
	void checkAction( Action *action );
	void checkInlineList( Action *act, InlineList *inlineList );

	void analyzeAction( Action *action, InlineList *inlineList );
	void analyzeGraph( FsmAp *graph );
	void makeExports();

	void prepareMachineGen( GraphDictEl *graphDictEl );
	void prepareMachineGenTBWrapped( GraphDictEl *graphDictEl );
	void generateXML( ostream &out );
	void generateReduced( InputData &inputData );
	FsmAp *sectionGraph;
	bool generatingSectionSubset;

	void initKeyOps();

	/*
	 * Data collected during the parse.
	 */

	/* Dictionary of graphs. Both instances and non-instances go here. */
	GraphDict graphDict;

	/* The list of instances. */
	GraphList instanceList;

	/* Dictionary of actions. Lets actions be defined and then referenced. */
	ActionDict actionDict;

	/* Dictionary of named priorities. */
	PriorDict priorDict;

	/* Dictionary of named local errors. */
	LocalErrDict localErrDict;

	/* List of actions. Will be pasted into a switch statement. */
	ActionList actionList;

	/* The id of the next priority name and label. */
	int nextPriorKey, nextLocalErrKey, nextNameId, nextCondId;
	
	/* The default priority number key for a machine. This is active during
	 * the parse of the rhs of a machine assignment. */
	int curDefPriorKey;

	int curDefLocalErrKey;

	/* Alphabet type. */
	HostType *userAlphType;
	bool alphTypeSet;
	InputLoc alphTypeLoc;

	/* Element type and get key expression. */
	InlineList *getKeyExpr;
	InlineList *accessExpr;

	/* Stack management */
	InlineList *prePushExpr;
	InlineList *postPopExpr;

	/* Overriding variables. */
	InlineList *pExpr;
	InlineList *peExpr;
	InlineList *eofExpr;
	InlineList *csExpr;
	InlineList *topExpr;
	InlineList *stackExpr;
	InlineList *actExpr;
	InlineList *tokstartExpr;
	InlineList *tokendExpr;
	InlineList *dataExpr;

	/* The alphabet range. */
	char *lowerNum, *upperNum;
	Key lowKey, highKey;
	InputLoc rangeLowLoc, rangeHighLoc;

	/* The name of the file the fsm is from, and the spec name. */
	const char *fileName;
	char *sectionName;
	InputLoc sectionLoc;

	/* Counting the action and priority ordering. */
	int curActionOrd;
	int curPriorOrd;

	/* Root of the name tree. One root is for the instantiated machines. The
	 * other root is for exported definitions. */
	NameInst *rootName;
	NameInst *exportsRootName;
	
	/* Name tree walking. */
	NameInst *curNameInst;
	int curNameChild;

	/* The place where resolved epsilon transitions go. These cannot go into
	 * the parse tree because a single epsilon op can resolve more than once
	 * to different nameInsts if the machine it's in is used more than once. */
	NameVect epsilonResolvedLinks;
	int nextEpsilonResolvedLink;

	/* Root of the name tree used for doing local name searches. */
	NameInst *localNameScope;

	void setLmInRetLoc( InlineList *inlineList );
	void initLongestMatchData();
	void setLongestMatchData( FsmAp *graph );
	void initNameWalk();
	void initExportsNameWalk();
	NameInst *nextNameScope() { return curNameInst->childVect[curNameChild]; }
	NameFrame enterNameScope( bool isLocal, int numScopes );
	void popNameScope( const NameFrame &frame );
	void resetNameScope( const NameFrame &frame );

	/* Make name ids to name inst pointers. */
	NameInst **nameIndex;

	/* Counter for assigning ids to longest match items. */
	int nextLongestMatchId;
	bool lmRequiresErrorState;

	/* List of all longest match parse tree items. */
	LmList lmList;

	Action *newAction( const char *name, InlineList *inlineList );

	Action *initTokStart;
	int initTokStartOrd;

	Action *setTokStart;
	int setTokStartOrd;

	Action *initActId;
	int initActIdOrd;

	Action *setTokEnd;
	int setTokEndOrd;

	void beginProcessing()
	{
		::condData = &thisCondData;
		::keyOps = &thisKeyOps;
	}

	CondData thisCondData;
	KeyOps thisKeyOps;

	ExportList exportList;
	LengthDefList lengthDefList;

	CodeGenData *cgd;
};

void afterOpMinimize( FsmAp *fsm, bool lastInSeq = true );
Key makeFsmKeyHex( char *str, const InputLoc &loc, ParseData *pd );
Key makeFsmKeyDec( char *str, const InputLoc &loc, ParseData *pd );
Key makeFsmKeyNum( char *str, const InputLoc &loc, ParseData *pd );
Key makeFsmKeyChar( char c, ParseData *pd );
void makeFsmKeyArray( Key *result, char *data, int len, ParseData *pd );
void makeFsmUniqueKeyArray( KeySet &result, char *data, int len, 
		bool caseInsensitive, ParseData *pd );
FsmAp *makeBuiltin( BuiltinMachine builtin, ParseData *pd );
FsmAp *dotFsm( ParseData *pd );
FsmAp *dotStarFsm( ParseData *pd );

void errorStateLabels( const NameSet &locations );


#endif
