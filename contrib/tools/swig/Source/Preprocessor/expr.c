/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at https://www.swig.org/legal.html.
 *
 * expr.c
 *
 * Integer arithmetic expression evaluator used to handle expressions
 * encountered during preprocessing.
 *
 * Note that this is used for expressions in `#if` and the like, but not
 * for expressions in `#define` which SWIG wraps as constants - for those
 * we inject a `%constant` directive which is handled by the parser in
 * `Source/CParse/parser.y`.
 * ----------------------------------------------------------------------------- */

#include "swig.h"
#include "preprocessor.h"

static Scanner *scan = 0;

typedef struct {
  /* One of the EXPR_xxx values defined below. */
  int op;
  /* op == EXPR_OP: value is the token specifying which operator.
   *
   * op == EXPR_VALUE && svalue == NULL: Numeric expression value.
   *
   * Otherwise unused.
   */
  long value;
  /* op == EXPR_VALUE: If non-NULL, string expression value; if NULL see value.
   *
   * Otherwise unused.
   */
  String *svalue;
} exprval;

#define  EXPR_TOP      1
#define  EXPR_VALUE    2
#define  EXPR_OP       3
#define  EXPR_GROUP    4

/* Special token values used here to distinguish from SWIG_TOKEN_MINUS
 * and SWIG_TOKEN_PLUS (which we use here for a two argument versions).
 */
#define  OP_UMINUS   100
#define  OP_UPLUS    101

static exprval stack[256];	/* Parsing stack       */
static int sp = 0;		/* Stack pointer       */
static int prec[256];		/* Precedence rules    */
static int expr_init = 0;	/* Initialization flag */
static const char *errmsg = 0;	/* Parsing error       */

/* Initialize the precedence table for various operators.  Low values have higher precedence */
static void init_precedence(void) {
  prec[SWIG_TOKEN_NOT] = 10;
  prec[SWIG_TOKEN_LNOT] = 10;
  prec[OP_UMINUS] = 10;
  prec[OP_UPLUS] = 10;
  prec[SWIG_TOKEN_STAR] = 20;
  prec[SWIG_TOKEN_SLASH] = 20;
  prec[SWIG_TOKEN_PERCENT] = 20;
  prec[SWIG_TOKEN_PLUS] = 30;
  prec[SWIG_TOKEN_MINUS] = 30;
  prec[SWIG_TOKEN_LSHIFT] = 40;
  prec[SWIG_TOKEN_RSHIFT] = 40;
  prec[SWIG_TOKEN_LESSTHAN] = 50;
  prec[SWIG_TOKEN_GREATERTHAN] = 50;
  prec[SWIG_TOKEN_LTEQUAL] = 50;
  prec[SWIG_TOKEN_GTEQUAL] = 50;
  prec[SWIG_TOKEN_EQUALTO] = 60;
  prec[SWIG_TOKEN_NOTEQUAL] = 60;
  prec[SWIG_TOKEN_AND] = 70;
  prec[SWIG_TOKEN_XOR] = 80;
  prec[SWIG_TOKEN_OR] = 90;
  prec[SWIG_TOKEN_LAND] = 100;
  prec[SWIG_TOKEN_LOR] = 110;
  expr_init = 1;
}

#define UNARY_OP(token) (((token) == SWIG_TOKEN_NOT) || \
			 ((token) == SWIG_TOKEN_LNOT) || \
			 ((token) == OP_UMINUS) || \
			 ((token) == OP_UPLUS))

/* Reduce a single operator on the stack */
/* return 0 on failure, 1 on success */
static int reduce_op(void) {
  long op_token = stack[sp - 1].value;
  assert(sp > 0);
  assert(stack[sp - 1].op == EXPR_OP);
  /* do some basic checking first: */
  if (stack[sp].op != EXPR_VALUE) {
    errmsg = "Right-hand side is not value";
    return 0;
  }
  if (UNARY_OP(op_token)) {
    if (stack[sp].svalue) {
      errmsg = "Syntax error: attempt to apply unary operator to string";
      return 0;
    }
  } else {
    /* binary operator: */
    if (sp == 1) {
      /* top of stack: don't attempt to use sp-2! */
      errmsg = "Missing left-hand side for binary operator";
      return 0;
    }
    if (stack[sp].op != EXPR_VALUE) {
      errmsg = "Left-hand side of binary operator is not a value";
      return 0;
    }
    if ((!stack[sp - 2].svalue) != (!stack[sp].svalue)) {
      errmsg = "Can't mix strings and integers in expression";
      return 0;
    }
  }
  if (stack[sp].svalue) {
    /* A binary string expression */
    switch (stack[sp - 1].value) {
    case SWIG_TOKEN_EQUALTO:
      stack[sp - 2].value = (Strcmp(stack[sp - 2].svalue, stack[sp].svalue) == 0);
      Delete(stack[sp - 2].svalue);
      Delete(stack[sp].svalue);
      sp -= 2;
      break;
    case SWIG_TOKEN_NOTEQUAL:
      stack[sp - 2].value = (Strcmp(stack[sp - 2].svalue, stack[sp].svalue) != 0);
      Delete(stack[sp - 2].svalue);
      Delete(stack[sp].svalue);
      sp -= 2;
      break;
    default:
      errmsg = "Syntax error: bad binary operator for strings";
      return 0;
      break;
    }
  } else {
    switch (op_token) {
    case SWIG_TOKEN_STAR:
      stack[sp - 2].value = stack[sp - 2].value * stack[sp].value;
      sp -= 2;
      break;
    case SWIG_TOKEN_EQUALTO:
      stack[sp - 2].value = stack[sp - 2].value == stack[sp].value;
      sp -= 2;
      break;
    case SWIG_TOKEN_NOTEQUAL:
      stack[sp - 2].value = stack[sp - 2].value != stack[sp].value;
      sp -= 2;
      break;
    case SWIG_TOKEN_PLUS:
      stack[sp - 2].value = stack[sp - 2].value + stack[sp].value;
      sp -= 2;
      break;
    case SWIG_TOKEN_MINUS:
      stack[sp - 2].value = stack[sp - 2].value - stack[sp].value;
      sp -= 2;
      break;
    case SWIG_TOKEN_AND:
      stack[sp - 2].value = stack[sp - 2].value & stack[sp].value;
      sp -= 2;
      break;
    case SWIG_TOKEN_LAND:
      stack[sp - 2].value = stack[sp - 2].value && stack[sp].value;
      sp -= 2;
      break;
    case SWIG_TOKEN_OR:
      stack[sp - 2].value = stack[sp - 2].value | stack[sp].value;
      sp -= 2;
      break;
    case SWIG_TOKEN_LOR:
      stack[sp - 2].value = stack[sp - 2].value || stack[sp].value;
      sp -= 2;
      break;
    case SWIG_TOKEN_XOR:
      stack[sp - 2].value = stack[sp - 2].value ^ stack[sp].value;
      sp -= 2;
      break;
    case SWIG_TOKEN_LESSTHAN:
      stack[sp - 2].value = stack[sp - 2].value < stack[sp].value;
      sp -= 2;
      break;
    case SWIG_TOKEN_GREATERTHAN:
      stack[sp - 2].value = stack[sp - 2].value > stack[sp].value;
      sp -= 2;
      break;
    case SWIG_TOKEN_LTEQUAL:
      stack[sp - 2].value = stack[sp - 2].value <= stack[sp].value;
      sp -= 2;
      break;
    case SWIG_TOKEN_GTEQUAL:
      stack[sp - 2].value = stack[sp - 2].value >= stack[sp].value;
      sp -= 2;
      break;
    case SWIG_TOKEN_NOT:
      stack[sp - 1].value = ~stack[sp].value;
      sp--;
      break;
    case SWIG_TOKEN_LNOT:
      stack[sp - 1].value = !stack[sp].value;
      sp--;
      break;
    case OP_UMINUS:
      stack[sp - 1].value = -stack[sp].value;
      sp--;
      break;
    case OP_UPLUS:
      stack[sp - 1].value = stack[sp].value;
      sp--;
      break;
    case SWIG_TOKEN_SLASH:
      if (stack[sp].value != 0) {
	stack[sp - 2].value = stack[sp - 2].value / stack[sp].value;
	sp -= 2;
      } else {
	errmsg = "Division by zero in expression";
	return 0;
      }
      break;
    case SWIG_TOKEN_PERCENT:
      if (stack[sp].value != 0) {
	stack[sp - 2].value = stack[sp - 2].value % stack[sp].value;
	sp -= 2;
      } else {
	errmsg = "Modulo by zero in expression";
	return 0;
      }
      break;
    case SWIG_TOKEN_LSHIFT:
      stack[sp - 2].value = stack[sp - 2].value << stack[sp].value;
      sp -= 2;
      break;
    case SWIG_TOKEN_RSHIFT:
      stack[sp - 2].value = stack[sp - 2].value >> stack[sp].value;
      sp -= 2;
      break;
    default:
      errmsg = "Syntax error: bad operator";
      return 0;
      break;
    }
  }
  stack[sp].op = EXPR_VALUE;
  stack[sp].svalue = 0;		/* ensure it's not a string! */
  return 1;
}

/* -----------------------------------------------------------------------------
 * Preprocessor_expr_init()
 *
 * Initialize the expression evaluator
 * ----------------------------------------------------------------------------- */

void Preprocessor_expr_init(void) {
  if (!expr_init)
    init_precedence();
  if (!scan)
    scan = NewScanner();
}

void Preprocessor_expr_delete(void) {
  DelScanner(scan);
}


/* -----------------------------------------------------------------------------
 * Tokenizer 
 * ----------------------------------------------------------------------------- */

static int expr_token(Scanner * s) {
  int t;
  while (1) {
    t = Scanner_token(s);
    if (!((t == SWIG_TOKEN_BACKSLASH) || (t == SWIG_TOKEN_ENDLINE) || (t == SWIG_TOKEN_COMMENT)))
      break;
  }
  return t;
}

/* -----------------------------------------------------------------------------
 * Preprocessor_expr()
 *
 * Evaluates an arithmetic expression.  Returns the result and sets an error code.
 * ----------------------------------------------------------------------------- */

int Preprocessor_expr(DOH *s, int *error) {
  int token = 0;
  int op = 0;

  sp = 0;
  assert(s);
  assert(scan);

  Seek(s, 0, SEEK_SET);
  /* Printf(stdout,"evaluating : '%s'\n", s); */
  *error = 0;
  Scanner_clear(scan);
  Scanner_push(scan, s);

  /* Put initial state onto the stack */
  stack[sp].op = EXPR_TOP;

  while (1) {
    /* Look at the top of the stack */
    switch (stack[sp].op) {
    case EXPR_TOP:
      /* EXPR_TOP is a place-holder which can only appear on the top of the
       * stack.  We can reduce it to any expression - a number, a string, an
       * unary operator, or another expression enclosed in parentheses.
       */
      token = expr_token(scan);
      if (!token) {
	errmsg = "Expected an expression";
	*error = 1;
	return 0;
      }
      if ((token == SWIG_TOKEN_INT) || (token == SWIG_TOKEN_UINT) || (token == SWIG_TOKEN_LONG) || (token == SWIG_TOKEN_ULONG)) {
	/* A number.  Reduce EXPR_TOP to an EXPR_VALUE */
	char *c = Char(Scanner_text(scan));
	if (c[0] == '0' && (c[1] == 'b' || c[1] == 'B')) {
	  /* strtol() doesn't handle binary constants */
	  stack[sp].value = (long) strtol(c + 2, 0, 2);
	} else {
	  stack[sp].value = (long) strtol(c, 0, 0);
	}
	stack[sp].svalue = 0;
	stack[sp].op = EXPR_VALUE;
      } else if ((token == SWIG_TOKEN_MINUS) || (token == SWIG_TOKEN_PLUS) || (token == SWIG_TOKEN_LNOT) || (token == SWIG_TOKEN_NOT)) {
	if (token == SWIG_TOKEN_MINUS)
	  token = OP_UMINUS;
	else if (token == SWIG_TOKEN_PLUS)
	  token = OP_UPLUS;
	stack[sp].value = token;
	stack[sp].op = EXPR_OP;
	sp++;
	stack[sp].op = EXPR_TOP;
      } else if (token == SWIG_TOKEN_LPAREN) {
	stack[sp].op = EXPR_GROUP;
	sp++;
	stack[sp].op = EXPR_TOP;
      } else if (token == SWIG_TOKEN_ENDLINE) {
      } else if (token == SWIG_TOKEN_STRING) {
	stack[sp].svalue = NewString(Scanner_text(scan));
	stack[sp].op = EXPR_VALUE;
      } else if (token == SWIG_TOKEN_ID) {
	/* Defined macros have been expanded already so this is an unknown
	 * macro, which gets treated as zero.
	 */
	stack[sp].value = 0;
	stack[sp].svalue = 0;
	stack[sp].op = EXPR_VALUE;
      } else if ((token == SWIG_TOKEN_FLOAT) || (token == SWIG_TOKEN_DOUBLE)) {
	errmsg = "Floating point constant in preprocessor expression";
	*error = 1;
	return 0;
      } else
	goto syntax_error;
      break;
    case EXPR_VALUE:
      /* A value is on top of the stack.  We may reduce or evaluate depending
       * on what the next token is.
       */
      token = expr_token(scan);
      if (!token) {
	/* End of input. Might have to reduce if an operator is on stack */
	while (sp > 0) {
	  if (stack[sp - 1].op == EXPR_OP) {
	    if (!reduce_op())
	      goto reduce_error;
	  } else if (stack[sp - 1].op == EXPR_GROUP) {
	    errmsg = "Missing \')\'";
	    *error = 1;
	    return 0;
	  } else
	    goto syntax_error;
	}
	return stack[sp].value;
      }
      /* Token must be an operator */
      switch (token) {
      case SWIG_TOKEN_STAR:
      case SWIG_TOKEN_EQUALTO:
      case SWIG_TOKEN_NOTEQUAL:
      case SWIG_TOKEN_PLUS:
      case SWIG_TOKEN_MINUS:
      case SWIG_TOKEN_AND:
      case SWIG_TOKEN_LAND:
      case SWIG_TOKEN_OR:
      case SWIG_TOKEN_LOR:
      case SWIG_TOKEN_XOR:
      case SWIG_TOKEN_LESSTHAN:
      case SWIG_TOKEN_GREATERTHAN:
      case SWIG_TOKEN_LTEQUAL:
      case SWIG_TOKEN_GTEQUAL:
      case SWIG_TOKEN_SLASH:
      case SWIG_TOKEN_PERCENT:
      case SWIG_TOKEN_LSHIFT:
      case SWIG_TOKEN_RSHIFT:
	if ((sp == 0) || (stack[sp - 1].op == EXPR_GROUP)) {
	  /* No possibility of reduce. Push operator and expression */
	  sp++;
	  stack[sp].op = EXPR_OP;
	  stack[sp].value = token;
	  sp++;
	  stack[sp].op = EXPR_TOP;
	} else {
	  if (stack[sp - 1].op != EXPR_OP)
	    goto syntax_error_expected_operator;
	  op = stack[sp - 1].value;	/* Previous operator */

	  /* Now, depending on the precedence relationship between the last operator and the current
	     we will reduce or push */

	  if (prec[op] <= prec[token]) {
	    /* Reduce the previous operator */
	    if (!reduce_op())
	      goto reduce_error;
	  }
	  sp++;
	  stack[sp].op = EXPR_OP;
	  stack[sp].value = token;
	  sp++;
	  stack[sp].op = EXPR_TOP;
	}
	break;
      case SWIG_TOKEN_RPAREN:
	if (sp == 0)
	  goto extra_rparen;

	/* Might have to reduce operators first */
	while ((sp > 0) && (stack[sp - 1].op == EXPR_OP)) {
	  if (!reduce_op())
	    goto reduce_error;
	}
	if ((sp == 0) || (stack[sp - 1].op != EXPR_GROUP))
	  goto extra_rparen;
	stack[sp - 1].op = EXPR_VALUE;
	stack[sp - 1].value = stack[sp].value;
	stack[sp - 1].svalue = stack[sp].svalue;
	sp--;
	break;
      case SWIG_TOKEN_LTEQUALGT:
	goto spaceship_not_allowed;
      default:
	goto syntax_error_expected_operator;
	break;
      }
      break;

    default:
      fprintf(stderr, "Internal error in expression evaluator.\n");
      Exit(EXIT_FAILURE);
    }
  }

syntax_error:
  errmsg = "Syntax error";
  *error = 1;
  return 0;

syntax_error_expected_operator:
  errmsg = "Syntax error: expected operator";
  *error = 1;
  return 0;

reduce_error:
  /*  errmsg has been set by reduce_op */
  *error = 1;
  return 0;

extra_rparen:
  errmsg = "Extra \')\'";
  *error = 1;
  return 0;

spaceship_not_allowed:
  errmsg = "Spaceship operator (<=>) not allowed in preprocessor expression";
  *error = 1;
  return 0;
}

/* -----------------------------------------------------------------------------
 * Preprocessor_expr_error()
 *
 * Return error message set by the evaluator (if any)
 * ----------------------------------------------------------------------------- */

const char *Preprocessor_expr_error(void) {
  return errmsg;
}
