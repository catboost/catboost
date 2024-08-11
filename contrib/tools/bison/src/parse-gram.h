/* A Bison parser, made by GNU Bison 3.7.6.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

#ifndef YY_GRAM_SRC_PARSE_GRAM_H_INCLUDED
# define YY_GRAM_SRC_PARSE_GRAM_H_INCLUDED
/* Debug traces.  */
#ifndef GRAM_DEBUG
# if defined YYDEBUG
#if YYDEBUG
#   define GRAM_DEBUG 1
#  else
#   define GRAM_DEBUG 0
#  endif
# else /* ! defined YYDEBUG */
#  define GRAM_DEBUG 1
# endif /* ! defined YYDEBUG */
#endif  /* ! defined GRAM_DEBUG */
#if GRAM_DEBUG
extern int gram_debug;
#endif
/* "%code requires" blocks.  */
#line 21 "src/parse-gram.y"

  #include "symlist.h"
  #include "symtab.h"
#line 269 "src/parse-gram.y"

  typedef enum
  {
    param_none   = 0,
    param_lex    = 1 << 0,
    param_parse  = 1 << 1,
    param_both   = param_lex | param_parse
  } param_type;
#line 730 "src/parse-gram.y"

  #include "muscle-tab.h"
  typedef struct
  {
    char const *chars;
    muscle_kind kind;
  } value_type;

#line 79 "src/parse-gram.h"

/* Token kinds.  */
#ifndef GRAM_TOKENTYPE
# define GRAM_TOKENTYPE
  enum gram_tokentype
  {
    GRAM_EMPTY = -2,
    GRAM_EOF = 0,                  /* "end of file"  */
    GRAM_error = 1,                /* error  */
    GRAM_UNDEF = 2,                /* "invalid token"  */
    STRING = 3,                    /* "string"  */
    TSTRING = 4,                   /* "translatable string"  */
    PERCENT_TOKEN = 5,             /* "%token"  */
    PERCENT_NTERM = 6,             /* "%nterm"  */
    PERCENT_TYPE = 7,              /* "%type"  */
    PERCENT_DESTRUCTOR = 8,        /* "%destructor"  */
    PERCENT_PRINTER = 9,           /* "%printer"  */
    PERCENT_LEFT = 10,             /* "%left"  */
    PERCENT_RIGHT = 11,            /* "%right"  */
    PERCENT_NONASSOC = 12,         /* "%nonassoc"  */
    PERCENT_PRECEDENCE = 13,       /* "%precedence"  */
    PERCENT_PREC = 14,             /* "%prec"  */
    PERCENT_DPREC = 15,            /* "%dprec"  */
    PERCENT_MERGE = 16,            /* "%merge"  */
    PERCENT_CODE = 17,             /* "%code"  */
    PERCENT_DEFAULT_PREC = 18,     /* "%default-prec"  */
    PERCENT_DEFINE = 19,           /* "%define"  */
    PERCENT_DEFINES = 20,          /* "%defines"  */
    PERCENT_ERROR_VERBOSE = 21,    /* "%error-verbose"  */
    PERCENT_EXPECT = 22,           /* "%expect"  */
    PERCENT_EXPECT_RR = 23,        /* "%expect-rr"  */
    PERCENT_FLAG = 24,             /* "%<flag>"  */
    PERCENT_FILE_PREFIX = 25,      /* "%file-prefix"  */
    PERCENT_GLR_PARSER = 26,       /* "%glr-parser"  */
    PERCENT_INITIAL_ACTION = 27,   /* "%initial-action"  */
    PERCENT_LANGUAGE = 28,         /* "%language"  */
    PERCENT_NAME_PREFIX = 29,      /* "%name-prefix"  */
    PERCENT_NO_DEFAULT_PREC = 30,  /* "%no-default-prec"  */
    PERCENT_NO_LINES = 31,         /* "%no-lines"  */
    PERCENT_NONDETERMINISTIC_PARSER = 32, /* "%nondeterministic-parser"  */
    PERCENT_OUTPUT = 33,           /* "%output"  */
    PERCENT_PURE_PARSER = 34,      /* "%pure-parser"  */
    PERCENT_REQUIRE = 35,          /* "%require"  */
    PERCENT_SKELETON = 36,         /* "%skeleton"  */
    PERCENT_START = 37,            /* "%start"  */
    PERCENT_TOKEN_TABLE = 38,      /* "%token-table"  */
    PERCENT_VERBOSE = 39,          /* "%verbose"  */
    PERCENT_YACC = 40,             /* "%yacc"  */
    BRACED_CODE = 41,              /* "{...}"  */
    BRACED_PREDICATE = 42,         /* "%?{...}"  */
    BRACKETED_ID = 43,             /* "[identifier]"  */
    CHAR_LITERAL = 44,             /* "character literal"  */
    COLON = 45,                    /* ":"  */
    EPILOGUE = 46,                 /* "epilogue"  */
    EQUAL = 47,                    /* "="  */
    ID = 48,                       /* "identifier"  */
    ID_COLON = 49,                 /* "identifier:"  */
    PERCENT_PERCENT = 50,          /* "%%"  */
    PIPE = 51,                     /* "|"  */
    PROLOGUE = 52,                 /* "%{...%}"  */
    SEMICOLON = 53,                /* ";"  */
    TAG = 54,                      /* "<tag>"  */
    TAG_ANY = 55,                  /* "<*>"  */
    TAG_NONE = 56,                 /* "<>"  */
    INT_LITERAL = 57,              /* "integer literal"  */
    PERCENT_PARAM = 58,            /* "%param"  */
    PERCENT_UNION = 59,            /* "%union"  */
    PERCENT_EMPTY = 60             /* "%empty"  */
  };
  typedef enum gram_tokentype gram_token_kind_t;
#endif

/* Value type.  */
#if ! defined GRAM_STYPE && ! defined GRAM_STYPE_IS_DECLARED
union GRAM_STYPE
{
  assoc precedence_declarator;             /* precedence_declarator  */
  char* STRING;                            /* "string"  */
  char* TSTRING;                           /* "translatable string"  */
  char* BRACED_CODE;                       /* "{...}"  */
  char* BRACED_PREDICATE;                  /* "%?{...}"  */
  char* EPILOGUE;                          /* "epilogue"  */
  char* PROLOGUE;                          /* "%{...%}"  */
  code_props_type code_props_type;         /* code_props_type  */
  int INT_LITERAL;                         /* "integer literal"  */
  int yykind_82;                           /* int.opt  */
  named_ref* yykind_95;                    /* named_ref.opt  */
  param_type PERCENT_PARAM;                /* "%param"  */
  symbol* token_decl;                      /* token_decl  */
  symbol* alias;                           /* alias  */
  symbol* token_decl_for_prec;             /* token_decl_for_prec  */
  symbol* id;                              /* id  */
  symbol* id_colon;                        /* id_colon  */
  symbol* symbol;                          /* symbol  */
  symbol* string_as_id;                    /* string_as_id  */
  symbol_list* generic_symlist;            /* generic_symlist  */
  symbol_list* generic_symlist_item;       /* generic_symlist_item  */
  symbol_list* nterm_decls;                /* nterm_decls  */
  symbol_list* token_decls;                /* token_decls  */
  symbol_list* yykind_80;                  /* token_decl.1  */
  symbol_list* token_decls_for_prec;       /* token_decls_for_prec  */
  symbol_list* yykind_85;                  /* token_decl_for_prec.1  */
  symbol_list* symbol_decls;               /* symbol_decls  */
  symbol_list* yykind_88;                  /* symbol_decl.1  */
  uniqstr PERCENT_ERROR_VERBOSE;           /* "%error-verbose"  */
  uniqstr PERCENT_FLAG;                    /* "%<flag>"  */
  uniqstr PERCENT_FILE_PREFIX;             /* "%file-prefix"  */
  uniqstr PERCENT_NAME_PREFIX;             /* "%name-prefix"  */
  uniqstr PERCENT_PURE_PARSER;             /* "%pure-parser"  */
  uniqstr BRACKETED_ID;                    /* "[identifier]"  */
  uniqstr ID;                              /* "identifier"  */
  uniqstr ID_COLON;                        /* "identifier:"  */
  uniqstr TAG;                             /* "<tag>"  */
  uniqstr yykind_74;                       /* tag.opt  */
  uniqstr tag;                             /* tag  */
  uniqstr variable;                        /* variable  */
  unsigned char CHAR_LITERAL;              /* "character literal"  */
  value_type value;                        /* value  */

#line 199 "src/parse-gram.h"

};
typedef union GRAM_STYPE GRAM_STYPE;
# define GRAM_STYPE_IS_TRIVIAL 1
# define GRAM_STYPE_IS_DECLARED 1
#endif

/* Location type.  */
#if ! defined GRAM_LTYPE && ! defined GRAM_LTYPE_IS_DECLARED
typedef struct GRAM_LTYPE GRAM_LTYPE;
struct GRAM_LTYPE
{
  int first_line;
  int first_column;
  int last_line;
  int last_column;
};
# define GRAM_LTYPE_IS_DECLARED 1
# define GRAM_LTYPE_IS_TRIVIAL 1
#endif



int gram_parse (void);
/* "%code provides" blocks.  */
#line 27 "src/parse-gram.y"

  /* Initialize unquote.  */
  void parser_init (void);
  /* Deallocate storage for unquote.  */
  void parser_free (void);

#line 232 "src/parse-gram.h"

#endif /* !YY_GRAM_SRC_PARSE_GRAM_H_INCLUDED  */
