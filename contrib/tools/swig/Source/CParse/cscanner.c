/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at https://www.swig.org/legal.html.
 *
 * scanner.c
 *
 * SWIG tokenizer.  This file is a wrapper around the generic C scanner
 * found in Swig/scanner.c.   Extra logic is added both to accommodate the
 * bison-based grammar and certain peculiarities of C++ parsing (e.g.,
 * operator overloading, typedef resolution, etc.).  This code also splits
 * C identifiers up into keywords and SWIG directives.
 * ----------------------------------------------------------------------------- */

#include "cparse.h"
#include "parser.h"
#include <string.h>
#include <ctype.h>
#include <errno.h>

/* Scanner object */
static Scanner *scan = 0;

/* Global string containing C code. Used by the parser to grab code blocks */
String *scanner_ccode = 0;

/* The main file being parsed */
static String *main_input_file = 0;

/* Error reporting/location information */
int     cparse_line = 1;
String *cparse_file = 0;
int     cparse_start_line = 0;

/* C++ mode */
int cparse_cplusplus = 0;

/* Generate C++ compatible code when wrapping C code */
int cparse_cplusplusout = 0;

/* To allow better error reporting */
String *cparse_unknown_directive = 0;

// Default-initialised instances of token types to avoid uninitialised fields.
// The compiler will initialise all fields to zero or NULL for us.

static const struct Define default_dtype;

/* Private vars */
static int scan_init = 0;
static int num_brace = 0;
static int last_id = 0;
static int rename_active = 0;

/* Doxygen comments scanning */
int scan_doxygen_comments = 0;

static int isStructuralDoxygen(String *s) {
  static const char* const structuralTags[] = {
    "addtogroup",
    "callgraph",
    "callergraph",
    "category",
    "def",
    "defgroup",
    "dir",
    "example",
    "file",
    "headerfile",
    "internal",
    "mainpage",
    "name",
    "nosubgrouping",
    "overload",
    "package",
    "page",
    "protocol",
    "relates",
    "relatesalso",
    "showinitializer",
    "weakgroup",
  };

  unsigned n;
  char *slashPointer = Strchr(s, '\\');
  char *atPointer = Strchr(s,'@');
  if (slashPointer == NULL && atPointer == NULL)
    return 0;
  else if(slashPointer == NULL)
    slashPointer = atPointer;

  slashPointer++; /* skip backslash or at sign */

  for (n = 0; n < sizeof(structuralTags)/sizeof(structuralTags[0]); n++) {
    const size_t len = strlen(structuralTags[n]);
    if (strncmp(slashPointer, structuralTags[n], len) == 0) {
      /* Take care to avoid false positives with prefixes of other tags. */
      if (slashPointer[len] == '\0' || isspace((int)slashPointer[len]))
	return 1;
    }
  }

  return 0;
}

/* -----------------------------------------------------------------------------
 * Swig_cparse_cplusplus()
 * ----------------------------------------------------------------------------- */

void Swig_cparse_cplusplus(int v) {
  cparse_cplusplus = v;
}

/* -----------------------------------------------------------------------------
 * Swig_cparse_cplusplusout()
 * ----------------------------------------------------------------------------- */

void Swig_cparse_cplusplusout(int v) {
  cparse_cplusplusout = v;
}

/* ----------------------------------------------------------------------------
 * scanner_init()
 *
 * Initialize buffers
 * ------------------------------------------------------------------------- */

static void scanner_init(void) {
  scan = NewScanner();
  Scanner_idstart(scan,"%");
  scan_init = 1;
  scanner_ccode = NewStringEmpty();
}

/* ----------------------------------------------------------------------------
 * scanner_file(DOHFile *f)
 *
 * Start reading from new file
 * ------------------------------------------------------------------------- */
void scanner_file(DOHFile * f) {
  if (!scan_init) scanner_init();
  Scanner_clear(scan);
  Scanner_push(scan,f);
}

/* ----------------------------------------------------------------------------
 * scanner_start_inline(String *text, int line)
 *
 * Take a chunk of text and recursively feed it back into the scanner.  Used
 * by the %inline directive.
 * ------------------------------------------------------------------------- */

void scanner_start_inline(String *text, int line) {
  String *stext = Copy(text);

  Seek(stext,0,SEEK_SET);
  Setfile(stext,cparse_file);
  Setline(stext,line);
  Scanner_push(scan,stext);
  Delete(stext);
}

/* -----------------------------------------------------------------------------
 * skip_balanced()
 *
 * Skips a piece of code enclosed in begin/end symbols such as '{...}' or
 * (...).  Ignores symbols inside comments or strings.
 *
 * Returns 0 if successfully skipped, -1 if EOF found first.
 * ----------------------------------------------------------------------------- */

int skip_balanced(int startchar, int endchar) {
  int start_line = Scanner_line(scan);
  Clear(scanner_ccode);

  if (Scanner_skip_balanced(scan,startchar,endchar) < 0) {
    Swig_error(cparse_file, start_line, "Missing '%c'. Reached end of input.\n", endchar);
    return -1;
  }

  cparse_line = Scanner_line(scan);
  cparse_file = Scanner_file(scan);

  Append(scanner_ccode, Scanner_text(scan));
  if (endchar == '}')
    num_brace--;
  return 0;
}

/* -----------------------------------------------------------------------------
 * get_raw_text_balanced()
 *
 * Returns raw text between 2 braces
 * ----------------------------------------------------------------------------- */

String *get_raw_text_balanced(int startchar, int endchar) {
  return Scanner_get_raw_text_balanced(scan, startchar, endchar);
}

/* ----------------------------------------------------------------------------
 * void skip_decl(void)
 *
 * This tries to skip over an entire declaration.   For example
 *
 *  friend ostream& operator<<(ostream&, const char *s);
 *
 * or
 *  friend ostream& operator<<(ostream&, const char *s) { }
 *
 * ------------------------------------------------------------------------- */

void skip_decl(void) {
  int tok;
  int done = 0;
  int start_line = Scanner_line(scan);

  while (!done) {
    tok = Scanner_token(scan);
    if (tok == 0) {
      if (!Swig_error_count()) {
	Swig_error(cparse_file, start_line, "Missing semicolon (';'). Reached end of input.\n");
      }
      return;
    }
    if (tok == SWIG_TOKEN_LBRACE) {
      if (Scanner_skip_balanced(scan,'{','}') < 0) {
	Swig_error(cparse_file, start_line, "Missing closing brace ('}'). Reached end of input.\n");
      }
      break;
    }
    if (tok == SWIG_TOKEN_SEMI) {
      done = 1;
    }
  }
  cparse_file = Scanner_file(scan);
  cparse_line = Scanner_line(scan);
}

/* ----------------------------------------------------------------------------
 * int yylook()
 *
 * Lexical scanner.
 * ------------------------------------------------------------------------- */

static int yylook(void) {

  int tok = 0;

  while (1) {
    if ((tok = Scanner_token(scan)) == 0)
      return 0;
    if (tok == SWIG_TOKEN_ERROR)
      return 0;
    cparse_start_line = Scanner_start_line(scan);
    cparse_line = Scanner_line(scan);
    cparse_file = Scanner_file(scan);

    switch(tok) {
    case SWIG_TOKEN_ID:
      return ID;
    case SWIG_TOKEN_LPAREN: 
      return LPAREN;
    case SWIG_TOKEN_RPAREN: 
      return RPAREN;
    case SWIG_TOKEN_SEMI:
      return SEMI;
    case SWIG_TOKEN_COMMA:
      return COMMA;
    case SWIG_TOKEN_STAR:
      return STAR;
    case SWIG_TOKEN_RBRACE:
      num_brace--;
      if (num_brace < 0) {
	Swig_error(cparse_file, cparse_line, "Syntax error. Extraneous closing brace ('}')\n");
	num_brace = 0;
      } else {
	return RBRACE;
      }
      break;
    case SWIG_TOKEN_LBRACE:
      num_brace++;
      return LBRACE;
    case SWIG_TOKEN_EQUAL:
      return EQUAL;
    case SWIG_TOKEN_EQUALTO:
      return EQUALTO;
    case SWIG_TOKEN_PLUS:
      return PLUS;
    case SWIG_TOKEN_MINUS:
      return MINUS;
    case SWIG_TOKEN_SLASH:
      return SLASH;
    case SWIG_TOKEN_AND:
      return AND;
    case SWIG_TOKEN_LAND:
      return LAND;
    case SWIG_TOKEN_OR:
      return OR;
    case SWIG_TOKEN_LOR:
      return LOR;
    case SWIG_TOKEN_XOR:
      return XOR;
    case SWIG_TOKEN_NOT:
      return NOT;
    case SWIG_TOKEN_LNOT:
      return LNOT;
    case SWIG_TOKEN_NOTEQUAL:
      return NOTEQUALTO;
    case SWIG_TOKEN_LBRACKET:
      return LBRACKET;
    case SWIG_TOKEN_RBRACKET:
      return RBRACKET;
    case SWIG_TOKEN_QUESTION:
      return QUESTIONMARK;
    case SWIG_TOKEN_LESSTHAN:
      return LESSTHAN;
    case SWIG_TOKEN_LTEQUAL:
      return LESSTHANOREQUALTO;
    case SWIG_TOKEN_LSHIFT:
      return LSHIFT;
    case SWIG_TOKEN_GREATERTHAN:
      return GREATERTHAN;
    case SWIG_TOKEN_GTEQUAL:
      return GREATERTHANOREQUALTO;
    case SWIG_TOKEN_RSHIFT:
      return RSHIFT;
    case SWIG_TOKEN_ARROW:
      return ARROW;
    case SWIG_TOKEN_PERIOD:
      return PERIOD;
    case SWIG_TOKEN_PERCENT:
      return MODULO;
    case SWIG_TOKEN_COLON:
      return COLON;
    case SWIG_TOKEN_DCOLONSTAR:
      return DSTAR;
    case SWIG_TOKEN_LTEQUALGT:
      return LESSEQUALGREATER;
      
    case SWIG_TOKEN_DCOLON:
      {
	int nexttok = Scanner_token(scan);
	if (nexttok == SWIG_TOKEN_STAR) {
	  return DSTAR;
	} else if (nexttok == SWIG_TOKEN_NOT) {
	  return DCNOT;
	} else {
	  Scanner_pushtoken(scan,nexttok,Scanner_text(scan));
	  if (!last_id) {
	    scanner_next_token(DCOLON);
	    return NONID;
	  } else {
	    return DCOLON;
	  }
	}
      }
      break;
      
    case SWIG_TOKEN_ELLIPSIS:
      return ELLIPSIS;

    case SWIG_TOKEN_LLBRACKET:
      do {
        tok = Scanner_token(scan);
      } while ((tok != SWIG_TOKEN_RRBRACKET) && (tok > 0));
      if (tok <= 0) {
        Swig_error(cparse_file, cparse_line, "Unbalanced double brackets, missing closing (']]'). Reached end of input.\n");
      }
      break;

    case SWIG_TOKEN_RRBRACKET:
      /* Turn an unmatched ]] back into two ] - e.g. `a[a[0]]` */
      scanner_next_token(RBRACKET);
      return RBRACKET;

      /* Look for multi-character sequences */
      
    case SWIG_TOKEN_RSTRING:
      yylval.type = NewString(Scanner_text(scan));
      return TYPE_RAW;
      
    case SWIG_TOKEN_STRING:
      yylval.str = NewString(Scanner_text(scan));
      return STRING;

    case SWIG_TOKEN_WSTRING:
      yylval.str = NewString(Scanner_text(scan));
      return WSTRING;
      
    case SWIG_TOKEN_CHAR:
      yylval.str = NewString(Scanner_text(scan));
      if (Len(yylval.str) == 0) {
	Swig_error(cparse_file, cparse_line, "Empty character constant\n");
      }
      return CHARCONST;

    case SWIG_TOKEN_WCHAR:
      yylval.str = NewString(Scanner_text(scan));
      if (Len(yylval.str) == 0) {
	Swig_error(cparse_file, cparse_line, "Empty character constant\n");
      }
      return WCHARCONST;

      /* Numbers */
      
    case SWIG_TOKEN_INT:
      return NUM_INT;
      
    case SWIG_TOKEN_UINT:
      return NUM_UNSIGNED;
      
    case SWIG_TOKEN_LONG:
      return NUM_LONG;
      
    case SWIG_TOKEN_ULONG:
      return NUM_ULONG;
      
    case SWIG_TOKEN_LONGLONG:
      return NUM_LONGLONG;
      
    case SWIG_TOKEN_ULONGLONG:
      return NUM_ULONGLONG;
      
    case SWIG_TOKEN_DOUBLE:
      return NUM_DOUBLE;

    case SWIG_TOKEN_FLOAT:
      return NUM_FLOAT;
      
    case SWIG_TOKEN_LONGDOUBLE:
      return NUM_LONGDOUBLE;

    case SWIG_TOKEN_BOOL:
      return NUM_BOOL;
      
    case SWIG_TOKEN_POUND:
      Scanner_skip_line(scan);
      yylval.id = Swig_copy_string(Char(Scanner_text(scan)));
      return POUND;
      
    case SWIG_TOKEN_CODEBLOCK:
      yylval.str = NewString(Scanner_text(scan));
      return HBLOCK;
      
    case SWIG_TOKEN_COMMENT:
      {
	typedef enum {
	  DOX_COMMENT_PRE = -1,
	  DOX_COMMENT_NONE,
	  DOX_COMMENT_POST
	} comment_kind_t;
	comment_kind_t existing_comment = DOX_COMMENT_NONE;

	/* Concatenate or skip all consecutive comments at once. */
	do {
	  String *cmt = Scanner_text(scan);
	  String *cmt_modified = 0;
	  char *loc = Char(cmt);
	  if ((strncmp(loc, "/*@SWIG", 7) == 0) && (loc[Len(cmt)-3] == '@')) {
	    Scanner_locator(scan, cmt);
	  }
	  if (scan_doxygen_comments) { /* else just skip this node, to avoid crashes in parser module*/

	    int slashStyle = 0; /* Flag for "///" style doxygen comments */
	    if (strncmp(loc, "///", 3) == 0) {
	      slashStyle = 1;
	      if (Len(cmt) == 3) {
		/* Modify to make length=4 to ensure that the empty comment does
		   get processed to preserve the newlines in the original comments. */
		cmt_modified = NewStringf("%s ", cmt);
		cmt = cmt_modified;
		loc = Char(cmt);
	      }
	    }
	    
	    /* Check for all possible Doxygen comment start markers while ignoring
	       comments starting with a row of asterisks or slashes just as
	       Doxygen itself does.  Also skip empty comment (slash-star-star-slash), 
	       which causes a crash due to begin > end. */
	    if (Len(cmt) > 3 && loc[0] == '/' &&
		((loc[1] == '/' && ((loc[2] == '/' && loc[3] != '/') || loc[2] == '!')) ||
		 (loc[1] == '*' && ((loc[2] == '*' && loc[3] != '*' && loc[3] != '/') || loc[2] == '!')))) {
	      comment_kind_t this_comment = loc[3] == '<' ? DOX_COMMENT_POST : DOX_COMMENT_PRE;
	      if (existing_comment != DOX_COMMENT_NONE && this_comment != existing_comment) {
		/* We can't concatenate together Doxygen pre- and post-comments. */
		break;
	      }

	      if (this_comment == DOX_COMMENT_POST || !isStructuralDoxygen(loc)) {
		String *str;

		int begin = this_comment == DOX_COMMENT_POST ? 4 : 3;
		int end = Len(cmt);
		if (loc[end - 1] == '/' && loc[end - 2] == '*') {
		  end -= 2;
		}

		str = NewStringWithSize(loc + begin, end - begin);

		if (existing_comment == DOX_COMMENT_NONE) {
		  yylval.str = str;
		  Setline(yylval.str, Scanner_start_line(scan));
		  Setfile(yylval.str, Scanner_file(scan));
		} else {
		  if (slashStyle) {
		    /* Add a newline to the end of each doxygen "///" comment,
		       since they are processed individually, unlike the
		       slash-star style, which gets processed as a block with
		       newlines included. */
		    Append(yylval.str, "\n");
		  }
		  Append(yylval.str, str);
		}

		existing_comment = this_comment;
	      }
	    }
	  }
	  do {
	    tok = Scanner_token(scan);
	  } while (tok == SWIG_TOKEN_ENDLINE);
	  Delete(cmt_modified);
	} while (tok == SWIG_TOKEN_COMMENT);

	Scanner_pushtoken(scan, tok, Scanner_text(scan));

	switch (existing_comment) {
	  case DOX_COMMENT_PRE:
	    return DOXYGENSTRING;
	  case DOX_COMMENT_NONE:
	    break;
	  case DOX_COMMENT_POST:
	    return DOXYGENPOSTSTRING;
	}
      }
      break;
    case SWIG_TOKEN_ENDLINE:
      break;
    case SWIG_TOKEN_BACKSLASH:
      break;
    default:
      Swig_error(cparse_file, cparse_line, "Unexpected token '%s'.\n", Scanner_text(scan));
      Exit(EXIT_FAILURE);
    }
  }
}

void scanner_set_location(String *file, int line) {
  Scanner_set_location(scan,file,line-1);
}

void scanner_last_id(int x) {
  last_id = x;
}

void scanner_clear_rename(void) {
  rename_active = 0;
}

/* Used to push a fictitious token into the scanner */
static int next_token = 0;
void scanner_next_token(int tok) {
  next_token = tok;
}

void scanner_set_main_input_file(String *file) {
  main_input_file = file;
}

String *scanner_get_main_input_file(void) {
  return main_input_file;
}

/* ----------------------------------------------------------------------------
 * int yylex()
 *
 * Gets the lexene and returns tokens.
 * ------------------------------------------------------------------------- */

int yylex(void) {

  int l;
  char *yytext;

  if (!scan_init) {
    scanner_init();
  }

  Delete(cparse_unknown_directive);
  cparse_unknown_directive = NULL;

  if (next_token) {
    l = next_token;
    next_token = 0;
    return l;
  }

  l = yylook();

  /*   Swig_diagnostic(cparse_file, cparse_line, ":::%d: '%s'\n", l, Scanner_text(scan)); */

  if (l == NONID) {
    last_id = 1;
  } else {
    last_id = 0;
  }

  /* We got some sort of non-white space object.  We set the start_line
     variable unless it has already been set */

  if (!cparse_start_line) {
    cparse_start_line = cparse_line;
  }

  /* Copy the lexene */

  switch (l) {

  case NUM_INT:
    yylval.dtype = default_dtype;
    yylval.dtype.type = T_INT;
    goto num_common;
  case NUM_DOUBLE:
    yylval.dtype = default_dtype;
    yylval.dtype.type = T_DOUBLE;
    goto num_common;
  case NUM_FLOAT:
    yylval.dtype = default_dtype;
    yylval.dtype.type = T_FLOAT;
    goto num_common;
  case NUM_LONGDOUBLE:
    yylval.dtype = default_dtype;
    yylval.dtype.type = T_LONGDOUBLE;
    goto num_common;
  case NUM_ULONG:
    yylval.dtype = default_dtype;
    yylval.dtype.type = T_ULONG;
    goto num_common;
  case NUM_LONG:
    yylval.dtype = default_dtype;
    yylval.dtype.type = T_LONG;
    goto num_common;
  case NUM_UNSIGNED:
    yylval.dtype = default_dtype;
    yylval.dtype.type = T_UINT;
    goto num_common;
  case NUM_LONGLONG:
    yylval.dtype = default_dtype;
    yylval.dtype.type = T_LONGLONG;
    goto num_common;
  case NUM_ULONGLONG:
    yylval.dtype = default_dtype;
    yylval.dtype.type = T_ULONGLONG;
    goto num_common;
num_common: {
    yylval.dtype.val = NewString(Scanner_text(scan));
    const char *c = Char(yylval.dtype.val);
    if (c[0] == '0') {
      // Convert to base 10 using strtoull().
      unsigned long long value;
      char *e;
      errno = 0;
      if (c[1] == 'b' || c[1] == 'B') {
	/* strtoull() doesn't handle binary literal prefixes so skip the prefix
	 * and specify base 2 explicitly. */
	value = strtoull(c + 2, &e, 2);
      } else {
	value = strtoull(c, &e, 0);
      }
      if (errno != ERANGE) {
	while (*e && strchr("ULul", *e)) ++e;
      }
      if (errno != ERANGE && *e == '\0') {
	yylval.dtype.numval = NewStringf("%llu", value);
      } else {
	// Our unsigned long long isn't wide enough or this isn't an integer.
      }
    } else {
      const char *e = c;
      while (isdigit((unsigned char)*e)) ++e;
      int len = e - c;
      while (*e && strchr("ULul", *e)) ++e;
      if (*e == '\0') {
        yylval.dtype.numval = NewStringWithSize(c, len);
      }
    }
    return (l);
  }
  case NUM_BOOL:
    yylval.dtype = default_dtype;
    yylval.dtype.type = T_BOOL;
    yylval.dtype.val = NewString(Scanner_text(scan));
    yylval.dtype.numval = NewString(Equal(yylval.dtype.val, "false") ? "0" : "1");
    return (l);

  case ID:
    yytext = Char(Scanner_text(scan));
    if (yytext[0] != '%') {
      /* Look for keywords now */

      if (strcmp(yytext, "int") == 0) {
	yylval.type = NewSwigType(T_INT);
	return (TYPE_INT);
      }
      if (strcmp(yytext, "double") == 0) {
	yylval.type = NewSwigType(T_DOUBLE);
	return (TYPE_DOUBLE);
      }
      if (strcmp(yytext, "void") == 0) {
	yylval.type = NewSwigType(T_VOID);
	return (TYPE_VOID);
      }
      if (strcmp(yytext, "char") == 0) {
	yylval.type = NewSwigType(T_CHAR);
	return (TYPE_CHAR);
      }
      if (strcmp(yytext, "wchar_t") == 0) {
	yylval.type = NewSwigType(T_WCHAR);
	return (TYPE_WCHAR);
      }
      if (strcmp(yytext, "short") == 0) {
	yylval.type = NewSwigType(T_SHORT);
	return (TYPE_SHORT);
      }
      if (strcmp(yytext, "long") == 0) {
	yylval.type = NewSwigType(T_LONG);
	return (TYPE_LONG);
      }
      if (strcmp(yytext, "float") == 0) {
	yylval.type = NewSwigType(T_FLOAT);
	return (TYPE_FLOAT);
      }
      if (strcmp(yytext, "signed") == 0) {
	yylval.type = NewSwigType(T_INT);
	return (TYPE_SIGNED);
      }
      if (strcmp(yytext, "unsigned") == 0) {
	yylval.type = NewSwigType(T_UINT);
	return (TYPE_UNSIGNED);
      }
      if (strcmp(yytext, "bool") == 0) {
	yylval.type = NewSwigType(T_BOOL);
	return (TYPE_BOOL);
      }

      /* Non ISO (Windows) C extensions */
      if (strcmp(yytext, "__int8") == 0) {
	yylval.type = NewString(yytext);
	return (TYPE_NON_ISO_INT8);
      }
      if (strcmp(yytext, "__int16") == 0) {
	yylval.type = NewString(yytext);
	return (TYPE_NON_ISO_INT16);
      }
      if (strcmp(yytext, "__int32") == 0) {
	yylval.type = NewString(yytext);
	return (TYPE_NON_ISO_INT32);
      }
      if (strcmp(yytext, "__int64") == 0) {
	yylval.type = NewString(yytext);
	return (TYPE_NON_ISO_INT64);
      }

      /* C++ keywords */
      if (cparse_cplusplus) {
	if (strcmp(yytext, "class") == 0)
	  return (CLASS);
	if (strcmp(yytext, "private") == 0)
	  return (PRIVATE);
	if (strcmp(yytext, "public") == 0)
	  return (PUBLIC);
	if (strcmp(yytext, "protected") == 0)
	  return (PROTECTED);
	if (strcmp(yytext, "friend") == 0)
	  return (FRIEND);
	if (strcmp(yytext, "constexpr") == 0)
	  return (CONSTEXPR);
	if (strcmp(yytext, "thread_local") == 0)
	  return (THREAD_LOCAL);
	if (strcmp(yytext, "decltype") == 0)
	  return (DECLTYPE);
	if (strcmp(yytext, "virtual") == 0)
	  return (VIRTUAL);
	if (strcmp(yytext, "static_assert") == 0)
	  return (STATIC_ASSERT);
	if (strcmp(yytext, "operator") == 0) {
	  int nexttok;
	  String *s = NewString("operator ");

	  /* If we have an operator, we have to collect the operator symbol and attach it to
             the operator identifier.   To do this, we need to scan ahead by several tokens.
             Cases include:

             (1) If the next token is an operator as determined by Scanner_isoperator(),
                 it means that the operator applies to one of the standard C++ mathematical,
                 assignment, or logical operator symbols (e.g., '+','<=','==','&', etc.)
                 In this case, we merely append the symbol text to the operator string above.

             (2) If the next token is (, we look for ).  This is operator ().
             (3) If the next token is [, we look for ].  This is operator [].
	     (4) If the next token is an identifier.  The operator is possibly a conversion operator.
                      (a) Must check for special case new[] and delete[]

             Error handling is somewhat tricky here.  We'll try to back out gracefully if we can.
 
	  */

	  do {
	    nexttok = Scanner_token(scan);
	  } while (nexttok == SWIG_TOKEN_ENDLINE || nexttok == SWIG_TOKEN_COMMENT);

	  if (Scanner_isoperator(nexttok)) {
	    /* One of the standard C/C++ symbolic operators */
	    Append(s,Scanner_text(scan));
	    yylval.str = s;
	    return OPERATOR;
	  } else if (nexttok == SWIG_TOKEN_LPAREN) {
	    /* Function call operator.  The next token MUST be a RPAREN */
	    nexttok = Scanner_token(scan);
	    if (nexttok != SWIG_TOKEN_RPAREN) {
	      Swig_error(Scanner_file(scan),Scanner_line(scan),"Syntax error. Bad operator name.\n");
	    } else {
	      Append(s,"()");
	      yylval.str = s;
	      return OPERATOR;
	    }
	  } else if (nexttok == SWIG_TOKEN_LBRACKET) {
	    /* Array access operator.  The next token MUST be a RBRACKET */
	    nexttok = Scanner_token(scan);
	    if (nexttok != SWIG_TOKEN_RBRACKET) {
	      Swig_error(Scanner_file(scan),Scanner_line(scan),"Syntax error. Bad operator name.\n");	      
	    } else {
	      Append(s,"[]");
	      yylval.str = s;
	      return OPERATOR;
	    }
	  } else if (nexttok == SWIG_TOKEN_STRING) {
	    /* Operator "" or user-defined string literal ""_suffix */
	    Append(s,"\"\"");
	    yylval.str = s;
	    return OPERATOR;
	  } else if (nexttok == SWIG_TOKEN_ID) {
	    /* We have an identifier.  It could be "new" or "delete",
	     * potentially followed by "[]", or it could be a conversion
	     * operator (it can't be "and_eq" or similar as those are returned
	     * as SWIG_TOKEN_ANDEQUAL, etc by Scanner_token()).  To deal with
	     * this we read tokens until we encounter a suitable terminating
	     * token.  Some care is needed for formatting. */
	    int needspace = 1;
	    int termtoken = 0;
	    const char *termvalue = 0;

	    Append(s,Scanner_text(scan));
	    while (1) {

	      nexttok = Scanner_token(scan);
	      if (nexttok <= 0) {
		Swig_error(Scanner_file(scan),Scanner_line(scan),"Syntax error. Bad operator name.\n");	      
	      }
	      if (nexttok == SWIG_TOKEN_LPAREN) {
		termtoken = SWIG_TOKEN_LPAREN;
		termvalue = "(";
		break;
              } else if (nexttok == SWIG_TOKEN_CODEBLOCK) {
                termtoken = SWIG_TOKEN_CODEBLOCK;
                termvalue = Char(Scanner_text(scan));
                break;
              } else if (nexttok == SWIG_TOKEN_LBRACE) {
                termtoken = SWIG_TOKEN_LBRACE;
                termvalue = "{";
                break;
              } else if (nexttok == SWIG_TOKEN_SEMI) {
		termtoken = SWIG_TOKEN_SEMI;
		termvalue = ";";
		break;
              } else if (nexttok == SWIG_TOKEN_STRING) {
		termtoken = SWIG_TOKEN_STRING;
                termvalue = Swig_copy_string(Char(Scanner_text(scan)));
		break;
	      } else if (nexttok == SWIG_TOKEN_ID) {
		if (needspace) {
		  Append(s," ");
		}
		Append(s,Scanner_text(scan));
	      } else if (nexttok == SWIG_TOKEN_ENDLINE) {
	      } else if (nexttok == SWIG_TOKEN_COMMENT) {
	      } else {
		Append(s,Scanner_text(scan));
		needspace = 0;
	      }
	    }
	    yylval.str = s;
	    if (!rename_active) {
	      String *cs;
	      char *t = Char(s) + 9;
	      if (!((strcmp(t, "new") == 0)
		    || (strcmp(t, "delete") == 0)
		    || (strcmp(t, "new[]") == 0)
		    || (strcmp(t, "delete[]") == 0)
		    )) {
		/*              retract(strlen(t)); */

		/* The operator is a conversion operator.   In order to deal with this, we need to feed the
                   type information back into the parser.  For now this is a hack.  Needs to be cleaned up later. */
		cs = NewString(t);
		if (termtoken) Append(cs,termvalue);
		Seek(cs,0,SEEK_SET);
		Setline(cs,cparse_line);
		Setfile(cs,cparse_file);
		Scanner_push(scan,cs);
		Delete(cs);
		return CONVERSIONOPERATOR;
	      }
	    }
	    if (termtoken)
              Scanner_pushtoken(scan, termtoken, termvalue);
	    return (OPERATOR);
	  }
	}
	if (strcmp(yytext, "throw") == 0)
	  return (THROW);
	if (strcmp(yytext, "noexcept") == 0)
	  return (NOEXCEPT);
	if (strcmp(yytext, "try") == 0)
	  return (yylex());
	if (strcmp(yytext, "catch") == 0)
	  return (CATCH);
	if (strcmp(yytext, "inline") == 0)
	  return (yylex());
	if (strcmp(yytext, "mutable") == 0)
	  return (yylex());
	if (strcmp(yytext, "explicit") == 0)
	  return (EXPLICIT);
	if (strcmp(yytext, "auto") == 0)
	  return (AUTO);
	if (strcmp(yytext, "export") == 0)
	  return (yylex());
	if (strcmp(yytext, "typename") == 0)
	  return (TYPENAME);
	if (strcmp(yytext, "template") == 0) {
	  yylval.intvalue = cparse_line;
	  return (TEMPLATE);
	}
	if (strcmp(yytext, "delete") == 0)
	  return (DELETE_KW);
	if (strcmp(yytext, "default") == 0)
	  return (DEFAULT);
	if (strcmp(yytext, "using") == 0)
	  return (USING);
	if (strcmp(yytext, "namespace") == 0)
	  return (NAMESPACE);
	if (strcmp(yytext, "alignof") == 0)
	  return (ALIGNOF);
	if (strcmp(yytext, "override") == 0) {
	  last_id = 1;
	  return (OVERRIDE);
	}
	if (strcmp(yytext, "final") == 0) {
	  last_id = 1;
	  return (FINAL);
	}
      } else {
	if (strcmp(yytext, "class") == 0) {
	  Swig_warning(WARN_PARSE_CLASS_KEYWORD, cparse_file, cparse_line, "class keyword used, but not in C++ mode.\n");
	}
	if (strcmp(yytext, "_Bool") == 0) {
	  /* C99 boolean type. */
	  yylval.type = NewSwigType(T_BOOL);
	  return (TYPE_BOOL);
	}
	if (strcmp(yytext, "_Complex") == 0) {
	  yylval.type = NewSwigType(T_COMPLEX);
	  return (TYPE_COMPLEX);
	}
	if (strcmp(yytext, "restrict") == 0)
	  return (yylex());
      }

      /* Misc keywords */

      if (strcmp(yytext, "extern") == 0)
	return (EXTERN);
      if (strcmp(yytext, "const") == 0)
	return (CONST_QUAL);
      if (strcmp(yytext, "static") == 0)
	return (STATIC);
      if (strcmp(yytext, "struct") == 0)
	return (STRUCT);
      if (strcmp(yytext, "union") == 0)
	return (UNION);
      if (strcmp(yytext, "enum") == 0)
	return (ENUM);
      if (strcmp(yytext, "sizeof") == 0)
	return (SIZEOF);

      if (strcmp(yytext, "typedef") == 0) {
	return (TYPEDEF);
      }

      /* Ignored keywords */

      if (strcmp(yytext, "volatile") == 0)
	return (VOLATILE);
      if (strcmp(yytext, "register") == 0)
	return (REGISTER);
      if (strcmp(yytext, "inline") == 0)
	return (yylex());

    } else {
      /* SWIG directives */
      String *stext = 0;
      if (strcmp(yytext, "%module") == 0)
	return (MODULE);
      if (strcmp(yytext, "%insert") == 0)
	return (INSERT);
      if (strcmp(yytext, "%rename") == 0) {
	rename_active = 1;
	return (RENAME);
      }
      if (strcmp(yytext, "%namewarn") == 0) {
	rename_active = 1;
	return (NAMEWARN);
      }
      if (strcmp(yytext, "%includefile") == 0)
	return (INCLUDE);
      if (strcmp(yytext, "%beginfile") == 0)
	return (BEGINFILE);
      if (strcmp(yytext, "%endoffile") == 0)
	return (ENDOFFILE);
      if (strcmp(yytext, "%constant") == 0)
	return (CONSTANT);
      if (strcmp(yytext, "%typedef") == 0) {
	return (TYPEDEF);
      }
      if (strcmp(yytext, "%native") == 0)
	return (NATIVE);
      if (strcmp(yytext, "%pragma") == 0)
	return (PRAGMA);
      if (strcmp(yytext, "%extend") == 0)
	return (EXTEND);
      if (strcmp(yytext, "%fragment") == 0)
	return (FRAGMENT);
      if (strcmp(yytext, "%inline") == 0)
	return (INLINE);
      if (strcmp(yytext, "%typemap") == 0)
	return (TYPEMAP);
      if (strcmp(yytext, "%feature") == 0) {
        /* The rename_active indicates we don't need the information of the 
         * following function's return type. This applied for %rename, so do
         * %feature. 
         */
        rename_active = 1;
	return (FEATURE);
      }
      if (strcmp(yytext, "%importfile") == 0)
	return (IMPORT);
      if (strcmp(yytext, "%echo") == 0)
	return (ECHO);
      if (strcmp(yytext, "%apply") == 0)
	return (APPLY);
      if (strcmp(yytext, "%clear") == 0)
	return (CLEAR);
      if (strcmp(yytext, "%types") == 0)
	return (TYPES);
      if (strcmp(yytext, "%parms") == 0)
	return (PARMS);
      if (strcmp(yytext, "%varargs") == 0)
	return (VARARGS);
      if (strcmp(yytext, "%template") == 0) {
	return (SWIGTEMPLATE);
      }
      if (strcmp(yytext, "%warn") == 0)
	return (WARN);

      /* Note down the apparently unknown directive for error reporting - if
       * we end up reporting a generic syntax error we'll instead report an
       * error for this as an unknown directive.  Then we treat it as MODULO
       * (`%`) followed by an identifier and if that parses OK then
       * `cparse_unknown_directive` doesn't get used.
       *
       * This allows `a%b` to be handled in expressions without a space after
       * the operator.
       */
      cparse_unknown_directive = NewString(yytext);
      stext = NewString(yytext + 1);
      Seek(stext,0,SEEK_SET);
      Setfile(stext,cparse_file);
      Setline(stext,cparse_line);
      Scanner_push(scan,stext);
      Delete(stext);
      return (MODULO);
    }

    yylval.id = Swig_copy_string(yytext);
    last_id = 1;
    return (ID);
  case POUND:
    return yylex();
  default:
    return (l);
  }
}
