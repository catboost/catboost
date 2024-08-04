/* A Bison parser, made by GNU Bison 3.5.4.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2020 Free Software Foundation,
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
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

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

/* Undocumented macros, especially those whose name start with YY_,
   are private implementation details.  Do not rely on them.  */

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
#line 251 "src/parse-gram.y"

  typedef enum
  {
    param_none   = 0,
    param_lex    = 1 << 0,
    param_parse  = 1 << 1,
    param_both   = param_lex | param_parse
  } param_type;
#line 703 "src/parse-gram.y"

  #include "muscle-tab.h"
  typedef struct
  {
    char const *chars;
    muscle_kind kind;
  } value_type;

#line 78 "src/parse-gram.h"

/* Token type.  */
#ifndef GRAM_TOKENTYPE
# define GRAM_TOKENTYPE
  enum gram_tokentype
  {
    GRAM_EOF = 0,
    STRING = 3,
    PERCENT_TOKEN = 4,
    PERCENT_NTERM = 5,
    PERCENT_TYPE = 6,
    PERCENT_DESTRUCTOR = 7,
    PERCENT_PRINTER = 8,
    PERCENT_LEFT = 9,
    PERCENT_RIGHT = 10,
    PERCENT_NONASSOC = 11,
    PERCENT_PRECEDENCE = 12,
    PERCENT_PREC = 13,
    PERCENT_DPREC = 14,
    PERCENT_MERGE = 15,
    PERCENT_CODE = 16,
    PERCENT_DEFAULT_PREC = 17,
    PERCENT_DEFINE = 18,
    PERCENT_DEFINES = 19,
    PERCENT_ERROR_VERBOSE = 20,
    PERCENT_EXPECT = 21,
    PERCENT_EXPECT_RR = 22,
    PERCENT_FLAG = 23,
    PERCENT_FILE_PREFIX = 24,
    PERCENT_GLR_PARSER = 25,
    PERCENT_INITIAL_ACTION = 26,
    PERCENT_LANGUAGE = 27,
    PERCENT_NAME_PREFIX = 28,
    PERCENT_NO_DEFAULT_PREC = 29,
    PERCENT_NO_LINES = 30,
    PERCENT_NONDETERMINISTIC_PARSER = 31,
    PERCENT_OUTPUT = 32,
    PERCENT_PURE_PARSER = 33,
    PERCENT_REQUIRE = 34,
    PERCENT_SKELETON = 35,
    PERCENT_START = 36,
    PERCENT_TOKEN_TABLE = 37,
    PERCENT_VERBOSE = 38,
    PERCENT_YACC = 39,
    BRACED_CODE = 40,
    BRACED_PREDICATE = 41,
    BRACKETED_ID = 42,
    CHAR = 43,
    COLON = 44,
    EPILOGUE = 45,
    EQUAL = 46,
    ID = 47,
    ID_COLON = 48,
    PERCENT_PERCENT = 49,
    PIPE = 50,
    PROLOGUE = 51,
    SEMICOLON = 52,
    TAG = 53,
    TAG_ANY = 54,
    TAG_NONE = 55,
    INT = 56,
    PERCENT_PARAM = 57,
    PERCENT_UNION = 58,
    PERCENT_EMPTY = 59
  };
#endif

/* Value type.  */
#if ! defined GRAM_STYPE && ! defined GRAM_STYPE_IS_DECLARED
union GRAM_STYPE
{

  /* precedence_declarator  */
  assoc precedence_declarator;
  /* "string"  */
  char* STRING;
  /* "{...}"  */
  char* BRACED_CODE;
  /* "%?{...}"  */
  char* BRACED_PREDICATE;
  /* "epilogue"  */
  char* EPILOGUE;
  /* "%{...%}"  */
  char* PROLOGUE;
  /* code_props_type  */
  code_props_type code_props_type;
  /* "integer"  */
  int INT;
  /* int.opt  */
  int yytype_81;
  /* named_ref.opt  */
  named_ref* yytype_93;
  /* "%param"  */
  param_type PERCENT_PARAM;
  /* token_decl  */
  symbol* token_decl;
  /* token_decl_for_prec  */
  symbol* token_decl_for_prec;
  /* id  */
  symbol* id;
  /* id_colon  */
  symbol* id_colon;
  /* symbol  */
  symbol* symbol;
  /* string_as_id  */
  symbol* string_as_id;
  /* string_as_id.opt  */
  symbol* yytype_100;
  /* generic_symlist  */
  symbol_list* generic_symlist;
  /* generic_symlist_item  */
  symbol_list* generic_symlist_item;
  /* nterm_decls  */
  symbol_list* nterm_decls;
  /* token_decls  */
  symbol_list* token_decls;
  /* token_decl.1  */
  symbol_list* yytype_79;
  /* token_decls_for_prec  */
  symbol_list* token_decls_for_prec;
  /* token_decl_for_prec.1  */
  symbol_list* yytype_83;
  /* symbol_decls  */
  symbol_list* symbol_decls;
  /* symbol_decl.1  */
  symbol_list* yytype_86;
  /* "%error-verbose"  */
  uniqstr PERCENT_ERROR_VERBOSE;
  /* "%<flag>"  */
  uniqstr PERCENT_FLAG;
  /* "%file-prefix"  */
  uniqstr PERCENT_FILE_PREFIX;
  /* "%name-prefix"  */
  uniqstr PERCENT_NAME_PREFIX;
  /* "%pure-parser"  */
  uniqstr PERCENT_PURE_PARSER;
  /* "[identifier]"  */
  uniqstr BRACKETED_ID;
  /* "identifier"  */
  uniqstr ID;
  /* "identifier:"  */
  uniqstr ID_COLON;
  /* "<tag>"  */
  uniqstr TAG;
  /* tag.opt  */
  uniqstr yytype_73;
  /* tag  */
  uniqstr tag;
  /* variable  */
  uniqstr variable;
  /* "character literal"  */
  unsigned char CHAR;
  /* value  */
  value_type value;
#line 233 "src/parse-gram.h"

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

#endif /* !YY_GRAM_SRC_PARSE_GRAM_H_INCLUDED  */
