/*
 *  Copyright 2001-2006 Adrian Thurston <thurston@complang.org>
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

#ifndef _PARSETREE_H
#define _PARSETREE_H

#include "ragel.h"
#include "avlmap.h"
#include "bstmap.h"
#include "vector.h"
#include "dlist.h"

struct NameInst;

/* Types of builtin machines. */
enum BuiltinMachine
{
	BT_Any,
	BT_Ascii,
	BT_Extend,
	BT_Alpha,
	BT_Digit,
	BT_Alnum,
	BT_Lower,
	BT_Upper,
	BT_Cntrl,
	BT_Graph,
	BT_Print,
	BT_Punct,
	BT_Space,
	BT_Xdigit,
	BT_Lambda,
	BT_Empty
};


struct ParseData;

/* Leaf type. */
struct Literal;

/* Tree nodes. */

struct Term;
struct FactorWithAug;
struct FactorWithRep;
struct FactorWithNeg;
struct Factor;
struct Expression;
struct Join;
struct MachineDef;
struct LongestMatch;
struct LongestMatchPart;
struct LmPartList;
struct Range;
struct LengthDef;

/* Type of augmentation. Describes locations in the machine. */
enum AugType
{
	/* Transition actions/priorities. */
	at_start,
	at_all,
	at_finish,
	at_leave,

	/* Global error actions. */
	at_start_gbl_error,
	at_all_gbl_error,
	at_final_gbl_error,
	at_not_start_gbl_error,
	at_not_final_gbl_error,
	at_middle_gbl_error,

	/* Local error actions. */
	at_start_local_error,
	at_all_local_error,
	at_final_local_error,
	at_not_start_local_error,
	at_not_final_local_error,
	at_middle_local_error,
	
	/* To State Action embedding. */
	at_start_to_state,
	at_all_to_state,
	at_final_to_state,
	at_not_start_to_state,
	at_not_final_to_state,
	at_middle_to_state,

	/* From State Action embedding. */
	at_start_from_state,
	at_all_from_state,
	at_final_from_state,
	at_not_start_from_state,
	at_not_final_from_state,
	at_middle_from_state,

	/* EOF Action embedding. */
	at_start_eof,
	at_all_eof,
	at_final_eof,
	at_not_start_eof,
	at_not_final_eof,
	at_middle_eof
};

/* IMPORTANT: These must follow the same order as the state augs in AugType
 * since we will be using this to compose AugType. */
enum StateAugType
{
	sat_start = 0,
	sat_all,
	sat_final,
	sat_not_start,
	sat_not_final,
	sat_middle
};

struct Action;
struct PriorDesc;
struct RegExpr;
struct ReItem;
struct ReOrBlock;
struct ReOrItem;
struct ExplicitMachine;
struct InlineItem;
struct InlineList;

/* Reference to a named state. */
typedef Vector<char*> NameRef;
typedef Vector<NameRef*> NameRefList;
typedef Vector<NameInst*> NameTargList;

/* Structure for storing location of epsilon transitons. */
struct EpsilonLink
{
	EpsilonLink( const InputLoc &loc, NameRef &target )
		: loc(loc), target(target) { }

	InputLoc loc;
	NameRef target;
};

struct Label
{
	Label( const InputLoc &loc, char *data )
		: loc(loc), data(data) { }

	InputLoc loc;
	char *data;
};

/* Structrue represents an action assigned to some FactorWithAug node. The
 * factor with aug will keep an array of these. */
struct ParserAction
{
	ParserAction( const InputLoc &loc, AugType type, int localErrKey, Action *action )
		: loc(loc), type(type), localErrKey(localErrKey), action(action) { }

	InputLoc loc;
	AugType type;
	int localErrKey;
	Action *action;
};

struct ConditionTest
{
	ConditionTest( const InputLoc &loc, AugType type, Action *action, bool sense ) : 
		loc(loc), type(type), action(action), sense(sense) { }

	InputLoc loc;
	AugType type;
	Action *action;
	bool sense;
};

struct Token
{
	char *data;
	int length;
	InputLoc loc;

	void append( const Token &other );
	void set( const char *str, int len );
};

char *prepareLitString( const InputLoc &loc, const char *src, long length, 
			long &resLen, bool &caseInsensitive );

/* Store the value and type of a priority augmentation. */
struct PriorityAug
{
	PriorityAug( AugType type, int priorKey, int priorValue ) :
		type(type), priorKey(priorKey), priorValue(priorValue) { }

	AugType type;
	int priorKey;
	int priorValue;
};

/*
 * A Variable Definition
 */
struct VarDef
{
	VarDef( const char *name, MachineDef *machineDef )
		: name(name), machineDef(machineDef), isExport(false) { }
	
	/* Parse tree traversal. */
	FsmAp *walk( ParseData *pd );
	void makeNameTree( const InputLoc &loc, ParseData *pd );
	void resolveNameRefs( ParseData *pd );

	const char *name;
	MachineDef *machineDef;
	bool isExport;
};


/*
 * LongestMatch
 *
 * Wherever possible the item match will execute on the character. If not
 * possible the item match will execute on a lookahead character and either
 * hold the current char (if one away) or backup.
 *
 * How to handle the problem of backing up over a buffer break?
 * 
 * Don't want to use pending out transitions for embedding item match because
 * the role of item match action is different: it may sometimes match on the
 * final transition, or may match on a lookahead character.
 *
 * Don't want to invent a new operator just for this. So just trail action
 * after machine, this means we can only use literal actions.
 *
 * The item action may 
 *
 * What states of the machine will be final. The item actions that wrap around
 * on the last character will go straight to the start state.
 *
 * Some transitions will be lookahead transitions, they will hold the current
 * character. Crossing them with regular transitions must be restricted
 * because it does not make sense. The transition cannot simultaneously hold
 * and consume the current character.
 */
struct LongestMatchPart
{
	LongestMatchPart( Join *join, Action *action, 
			InputLoc &semiLoc, int longestMatchId )
	: 
		join(join), action(action), semiLoc(semiLoc), 
		longestMatchId(longestMatchId), inLmSelect(false) { }

	InputLoc getLoc();
	
	Join *join;
	Action *action;
	InputLoc semiLoc;

	Action *setActId;
	Action *actOnLast;
	Action *actOnNext;
	Action *actLagBehind;
	int longestMatchId;
	bool inLmSelect;
	LongestMatch *longestMatch;

	LongestMatchPart *prev, *next;
};

/* Declare a new type so that ptreetypes.h need not include dlist.h. */
struct LmPartList : DList<LongestMatchPart> {};

struct LongestMatch
{
	/* Construct with a list of joins */
	LongestMatch( const InputLoc &loc, LmPartList *longestMatchList ) : 
		loc(loc), longestMatchList(longestMatchList), name(0), 
		lmSwitchHandlesError(false) { }

	/* Tree traversal. */
	FsmAp *walk( ParseData *pd );
	void makeNameTree( ParseData *pd );
	void resolveNameRefs( ParseData *pd );
	void transferScannerLeavingActions( FsmAp *graph );
	void runLongestMatch( ParseData *pd, FsmAp *graph );
	Action *newAction( ParseData *pd, const InputLoc &loc, const char *name, 
			InlineList *inlineList );
	void makeActions( ParseData *pd );
	void findName( ParseData *pd );
	void restart( FsmAp *graph, TransAp *trans );

	InputLoc loc;
	LmPartList *longestMatchList;
	const char *name;

	Action *lmActSelect;
	bool lmSwitchHandlesError;

	LongestMatch *next, *prev;
};


/* List of Expressions. */
typedef DList<Expression> ExprList;

struct MachineDef
{
	enum Type {
		JoinType,
		LongestMatchType,
		LengthDefType
	};

	MachineDef( Join *join )
		: join(join), longestMatch(0), lengthDef(0), type(JoinType) {}
	MachineDef( LongestMatch *longestMatch )
		: join(0), longestMatch(longestMatch), lengthDef(0), type(LongestMatchType) {}
	MachineDef( LengthDef *lengthDef )
		: join(0), longestMatch(0), lengthDef(lengthDef), type(LengthDefType) {}

	FsmAp *walk( ParseData *pd );
	void makeNameTree( ParseData *pd );
	void resolveNameRefs( ParseData *pd );
	
	Join *join;
	LongestMatch *longestMatch;
	LengthDef *lengthDef;
	Type type;
};

/*
 * Join
 */
struct Join
{
	/* Construct with the first expression. */
	Join( Expression *expr );
	Join( const InputLoc &loc, Expression *expr );

	/* Tree traversal. */
	FsmAp *walk( ParseData *pd );
	FsmAp *walkJoin( ParseData *pd );
	void makeNameTree( ParseData *pd );
	void resolveNameRefs( ParseData *pd );

	/* Data. */
	InputLoc loc;
	ExprList exprList;
};

/*
 * Expression
 */
struct Expression
{
	enum Type { 
		OrType,
		IntersectType, 
		SubtractType, 
		StrongSubtractType,
		TermType, 
		BuiltinType
	};

	/* Construct with an expression on the left and a term on the right. */
	Expression( Expression *expression, Term *term, Type type ) : 
		expression(expression), term(term), 
		type(type), prev(this), next(this) { }

	/* Construct with only a term. */
	Expression( Term *term ) : 
		expression(0), term(term),
		type(TermType) , prev(this), next(this) { }
	
	/* Construct with a builtin type. */
	Expression( BuiltinMachine builtin ) : 
		expression(0), term(0), builtin(builtin), 
		type(BuiltinType), prev(this), next(this) { }

	~Expression();

	/* Tree traversal. */
	FsmAp *walk( ParseData *pd, bool lastInSeq = true );
	void makeNameTree( ParseData *pd );
	void resolveNameRefs( ParseData *pd );

	/* Node data. */
	Expression *expression;
	Term *term;
	BuiltinMachine builtin;
	Type type;

	Expression *prev, *next;
};

/*
 * Term
 */
struct Term 
{
	enum Type { 
		ConcatType, 
		RightStartType,
		RightFinishType,
		LeftType,
		FactorWithAugType
	};

	Term( Term *term, FactorWithAug *factorWithAug ) :
		term(term), factorWithAug(factorWithAug), type(ConcatType) { }

	Term( Term *term, FactorWithAug *factorWithAug, Type type ) :
		term(term), factorWithAug(factorWithAug), type(type) { }

	Term( FactorWithAug *factorWithAug ) :
		term(0), factorWithAug(factorWithAug), type(FactorWithAugType) { }
	
	~Term();

	FsmAp *walk( ParseData *pd, bool lastInSeq = true );
	void makeNameTree( ParseData *pd );
	void resolveNameRefs( ParseData *pd );

	Term *term;
	FactorWithAug *factorWithAug;
	Type type;

	/* Priority descriptor for RightFinish type. */
	PriorDesc priorDescs[2];
};


/* Third level of precedence. Augmenting nodes with actions and priorities. */
struct FactorWithAug
{
	FactorWithAug( FactorWithRep *factorWithRep ) :
		priorDescs(0), factorWithRep(factorWithRep) { }
	~FactorWithAug();

	/* Tree traversal. */
	FsmAp *walk( ParseData *pd );
	void makeNameTree( ParseData *pd );
	void resolveNameRefs( ParseData *pd );

	void assignActions( ParseData *pd, FsmAp *graph, int *actionOrd );
	void assignPriorities( FsmAp *graph, int *priorOrd );

	void assignConditions( FsmAp *graph );

	/* Actions and priorities assigned to the factor node. */
	Vector<ParserAction> actions;
	Vector<PriorityAug> priorityAugs;
	PriorDesc *priorDescs;
	Vector<Label> labels;
	Vector<EpsilonLink> epsilonLinks;
	Vector<ConditionTest> conditions;

	FactorWithRep *factorWithRep;
};

/* Fourth level of precedence. Trailing unary operators. Provide kleen star,
 * optional and plus. */
struct FactorWithRep
{
	enum Type { 
		StarType,
		StarStarType,
		OptionalType,
		PlusType, 
		ExactType,
		MaxType,
		MinType,
		RangeType,
		FactorWithNegType
	};

	 FactorWithRep( const InputLoc &loc, FactorWithRep *factorWithRep, 
			int lowerRep, int upperRep, Type type ) :
		loc(loc), factorWithRep(factorWithRep), 
		factorWithNeg(0), lowerRep(lowerRep), 
		upperRep(upperRep), type(type) { }
	
	FactorWithRep( FactorWithNeg *factorWithNeg )
		: factorWithNeg(factorWithNeg), type(FactorWithNegType) { }

	~FactorWithRep();

	/* Tree traversal. */
	FsmAp *walk( ParseData *pd );
	void makeNameTree( ParseData *pd );
	void resolveNameRefs( ParseData *pd );

	InputLoc loc;
	FactorWithRep *factorWithRep;
	FactorWithNeg *factorWithNeg;
	int lowerRep, upperRep;
	Type type;

	/* Priority descriptor for StarStar type. */
	PriorDesc priorDescs[2];
};

/* Fifth level of precedence. Provides Negation. */
struct FactorWithNeg
{
	enum Type { 
		NegateType, 
		CharNegateType,
		FactorType
	};

	FactorWithNeg( const InputLoc &loc, FactorWithNeg *factorWithNeg, Type type) :
		loc(loc), factorWithNeg(factorWithNeg), factor(0), type(type) { }

	FactorWithNeg( Factor *factor ) :
		factorWithNeg(0), factor(factor), type(FactorType) { }

	~FactorWithNeg();

	/* Tree traversal. */
	FsmAp *walk( ParseData *pd );
	void makeNameTree( ParseData *pd );
	void resolveNameRefs( ParseData *pd );

	InputLoc loc;
	FactorWithNeg *factorWithNeg;
	Factor *factor;
	Type type;
};

/*
 * Factor
 */
struct Factor
{
	/* Language elements a factor node can be. */
	enum Type {
		LiteralType, 
		RangeType, 
		OrExprType,
		RegExprType, 
		ReferenceType,
		ParenType,
		LongestMatchType,
	}; 

	/* Construct with a literal fsm. */
	Factor( Literal *literal ) :
		literal(literal), type(LiteralType) { }

	/* Construct with a range. */
	Factor( Range *range ) : 
		range(range), type(RangeType) { }
	
	/* Construct with the or part of a regular expression. */
	Factor( ReItem *reItem ) :
		reItem(reItem), type(OrExprType) { }

	/* Construct with a regular expression. */
	Factor( RegExpr *regExpr ) :
		regExpr(regExpr), type(RegExprType) { }

	/* Construct with a reference to a var def. */
	Factor( const InputLoc &loc, VarDef *varDef ) :
		loc(loc), varDef(varDef), type(ReferenceType) {}

	/* Construct with a parenthesized join. */
	Factor( Join *join ) :
		join(join), type(ParenType) {}
	
	/* Construct with a longest match operator. */
	Factor( LongestMatch *longestMatch ) :
		longestMatch(longestMatch), type(LongestMatchType) {}

	/* Cleanup. */
	~Factor();

	/* Tree traversal. */
	FsmAp *walk( ParseData *pd );
	void makeNameTree( ParseData *pd );
	void resolveNameRefs( ParseData *pd );

	InputLoc loc;
	Literal *literal;
	Range *range;
	ReItem *reItem;
	RegExpr *regExpr;
	VarDef *varDef;
	Join *join;
	LongestMatch *longestMatch;
	int lower, upper;
	Type type;
};

/* A range machine. Only ever composed of two literals. */
struct Range
{
	Range( Literal *lowerLit, Literal *upperLit ) 
		: lowerLit(lowerLit), upperLit(upperLit) { }

	~Range();
	FsmAp *walk( ParseData *pd );

	Literal *lowerLit;
	Literal *upperLit;
};

/* Some literal machine. Can be a number or literal string. */
struct Literal
{
	enum LiteralType { Number, LitString };

	Literal( const Token &token, LiteralType type )
		: token(token), type(type) { }

	FsmAp *walk( ParseData *pd );
	
	Token token;
	LiteralType type;
};

/* Regular expression. */
struct RegExpr
{
	enum RegExpType { RecurseItem, Empty };

	/* Constructors. */
	RegExpr() : 
		type(Empty), caseInsensitive(false) { }
	RegExpr(RegExpr *regExpr, ReItem *item) : 
		regExpr(regExpr), item(item), 
		type(RecurseItem), caseInsensitive(false) { }

	~RegExpr();
	FsmAp *walk( ParseData *pd, RegExpr *rootRegex );

	RegExpr *regExpr;
	ReItem *item;
	RegExpType type;
	bool caseInsensitive;
};

/* An item in a regular expression. */
struct ReItem
{
	enum ReItemType { Data, Dot, OrBlock, NegOrBlock };
	
	ReItem( const InputLoc &loc, const Token &token ) 
		: loc(loc), token(token), star(false), type(Data) { }
	ReItem( const InputLoc &loc, ReItemType type )
		: loc(loc), star(false), type(type) { }
	ReItem( const InputLoc &loc, ReOrBlock *orBlock, ReItemType type )
		: loc(loc), orBlock(orBlock), star(false), type(type) { }

	~ReItem();
	FsmAp *walk( ParseData *pd, RegExpr *rootRegex );

	InputLoc loc;
	Token token;
	ReOrBlock *orBlock;
	bool star;
	ReItemType type;
};

/* An or block item. */
struct ReOrBlock
{
	enum ReOrBlockType { RecurseItem, Empty };

	/* Constructors. */
	ReOrBlock()
		: type(Empty) { }
	ReOrBlock(ReOrBlock *orBlock, ReOrItem *item)
		: orBlock(orBlock), item(item), type(RecurseItem) { }

	~ReOrBlock();
	FsmAp *walk( ParseData *pd, RegExpr *rootRegex );
	
	ReOrBlock *orBlock;
	ReOrItem *item;
	ReOrBlockType type;
};

/* An item in an or block. */
struct ReOrItem
{
	enum ReOrItemType { Data, Range };

	ReOrItem( const InputLoc &loc, const Token &token ) 
		: loc(loc), token(token), type(Data) {}
	ReOrItem( const InputLoc &loc, char lower, char upper )
		: loc(loc), lower(lower), upper(upper), type(Range) { }

	FsmAp *walk( ParseData *pd, RegExpr *rootRegex );

	InputLoc loc;
	Token token;
	char lower;
	char upper;
	ReOrItemType type;
};


/*
 * Inline code tree
 */
struct InlineList;
struct InlineItem
{
	enum Type 
	{
		Text, Goto, Call, Next, GotoExpr, CallExpr, NextExpr, Ret, PChar,
		Char, Hold, Curs, Targs, Entry, Exec, LmSwitch, LmSetActId,
		LmSetTokEnd, LmOnLast, LmOnNext, LmOnLagBehind, LmInitAct,
		LmInitTokStart, LmSetTokStart, Break
	};

	InlineItem( const InputLoc &loc, char *data, Type type ) : 
		loc(loc), data(data), nameRef(0), children(0), type(type) { }

	InlineItem( const InputLoc &loc, NameRef *nameRef, Type type ) : 
		loc(loc), data(0), nameRef(nameRef), children(0), type(type) { }

	InlineItem( const InputLoc &loc, LongestMatch *longestMatch, 
		LongestMatchPart *longestMatchPart, Type type ) : loc(loc), data(0),
		nameRef(0), children(0), longestMatch(longestMatch),
		longestMatchPart(longestMatchPart), type(type) { } 

	InlineItem( const InputLoc &loc, NameInst *nameTarg, Type type ) : 
		loc(loc), data(0), nameRef(0), nameTarg(nameTarg), children(0),
		type(type) { }

	InlineItem( const InputLoc &loc, Type type ) : 
		loc(loc), data(0), nameRef(0), children(0), type(type) { }
	
	InputLoc loc;
	char *data;
	NameRef *nameRef;
	NameInst *nameTarg;
	InlineList *children;
	LongestMatch *longestMatch;
	LongestMatchPart *longestMatchPart;
	Type type;

	InlineItem *prev, *next;
};

/* Normally this would be atypedef, but that would entail including DList from
 * ptreetypes, which should be just typedef forwards. */
struct InlineList : public DList<InlineItem> { };

#endif
