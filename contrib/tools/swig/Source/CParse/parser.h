/* A Bison parser, made by GNU Bison 3.8.2.  */

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

#ifndef YY_YY_CPARSE_PARSER_H_INCLUDED
# define YY_YY_CPARSE_PARSER_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    ID = 258,                      /* ID  */
    HBLOCK = 259,                  /* HBLOCK  */
    POUND = 260,                   /* POUND  */
    STRING = 261,                  /* STRING  */
    WSTRING = 262,                 /* WSTRING  */
    INCLUDE = 263,                 /* INCLUDE  */
    IMPORT = 264,                  /* IMPORT  */
    INSERT = 265,                  /* INSERT  */
    CHARCONST = 266,               /* CHARCONST  */
    WCHARCONST = 267,              /* WCHARCONST  */
    NUM_INT = 268,                 /* NUM_INT  */
    NUM_FLOAT = 269,               /* NUM_FLOAT  */
    NUM_UNSIGNED = 270,            /* NUM_UNSIGNED  */
    NUM_LONG = 271,                /* NUM_LONG  */
    NUM_ULONG = 272,               /* NUM_ULONG  */
    NUM_LONGLONG = 273,            /* NUM_LONGLONG  */
    NUM_ULONGLONG = 274,           /* NUM_ULONGLONG  */
    NUM_BOOL = 275,                /* NUM_BOOL  */
    TYPEDEF = 276,                 /* TYPEDEF  */
    TYPE_INT = 277,                /* TYPE_INT  */
    TYPE_UNSIGNED = 278,           /* TYPE_UNSIGNED  */
    TYPE_SHORT = 279,              /* TYPE_SHORT  */
    TYPE_LONG = 280,               /* TYPE_LONG  */
    TYPE_FLOAT = 281,              /* TYPE_FLOAT  */
    TYPE_DOUBLE = 282,             /* TYPE_DOUBLE  */
    TYPE_CHAR = 283,               /* TYPE_CHAR  */
    TYPE_WCHAR = 284,              /* TYPE_WCHAR  */
    TYPE_VOID = 285,               /* TYPE_VOID  */
    TYPE_SIGNED = 286,             /* TYPE_SIGNED  */
    TYPE_BOOL = 287,               /* TYPE_BOOL  */
    TYPE_COMPLEX = 288,            /* TYPE_COMPLEX  */
    TYPE_TYPEDEF = 289,            /* TYPE_TYPEDEF  */
    TYPE_RAW = 290,                /* TYPE_RAW  */
    TYPE_NON_ISO_INT8 = 291,       /* TYPE_NON_ISO_INT8  */
    TYPE_NON_ISO_INT16 = 292,      /* TYPE_NON_ISO_INT16  */
    TYPE_NON_ISO_INT32 = 293,      /* TYPE_NON_ISO_INT32  */
    TYPE_NON_ISO_INT64 = 294,      /* TYPE_NON_ISO_INT64  */
    LPAREN = 295,                  /* LPAREN  */
    RPAREN = 296,                  /* RPAREN  */
    COMMA = 297,                   /* COMMA  */
    SEMI = 298,                    /* SEMI  */
    EXTERN = 299,                  /* EXTERN  */
    INIT = 300,                    /* INIT  */
    LBRACE = 301,                  /* LBRACE  */
    RBRACE = 302,                  /* RBRACE  */
    PERIOD = 303,                  /* PERIOD  */
    CONST_QUAL = 304,              /* CONST_QUAL  */
    VOLATILE = 305,                /* VOLATILE  */
    REGISTER = 306,                /* REGISTER  */
    STRUCT = 307,                  /* STRUCT  */
    UNION = 308,                   /* UNION  */
    EQUAL = 309,                   /* EQUAL  */
    SIZEOF = 310,                  /* SIZEOF  */
    MODULE = 311,                  /* MODULE  */
    LBRACKET = 312,                /* LBRACKET  */
    RBRACKET = 313,                /* RBRACKET  */
    BEGINFILE = 314,               /* BEGINFILE  */
    ENDOFFILE = 315,               /* ENDOFFILE  */
    ILLEGAL = 316,                 /* ILLEGAL  */
    CONSTANT = 317,                /* CONSTANT  */
    NAME = 318,                    /* NAME  */
    RENAME = 319,                  /* RENAME  */
    NAMEWARN = 320,                /* NAMEWARN  */
    EXTEND = 321,                  /* EXTEND  */
    PRAGMA = 322,                  /* PRAGMA  */
    FEATURE = 323,                 /* FEATURE  */
    VARARGS = 324,                 /* VARARGS  */
    ENUM = 325,                    /* ENUM  */
    CLASS = 326,                   /* CLASS  */
    TYPENAME = 327,                /* TYPENAME  */
    PRIVATE = 328,                 /* PRIVATE  */
    PUBLIC = 329,                  /* PUBLIC  */
    PROTECTED = 330,               /* PROTECTED  */
    COLON = 331,                   /* COLON  */
    STATIC = 332,                  /* STATIC  */
    VIRTUAL = 333,                 /* VIRTUAL  */
    FRIEND = 334,                  /* FRIEND  */
    THROW = 335,                   /* THROW  */
    CATCH = 336,                   /* CATCH  */
    EXPLICIT = 337,                /* EXPLICIT  */
    STATIC_ASSERT = 338,           /* STATIC_ASSERT  */
    CONSTEXPR = 339,               /* CONSTEXPR  */
    THREAD_LOCAL = 340,            /* THREAD_LOCAL  */
    DECLTYPE = 341,                /* DECLTYPE  */
    AUTO = 342,                    /* AUTO  */
    NOEXCEPT = 343,                /* NOEXCEPT  */
    OVERRIDE = 344,                /* OVERRIDE  */
    FINAL = 345,                   /* FINAL  */
    USING = 346,                   /* USING  */
    NAMESPACE = 347,               /* NAMESPACE  */
    NATIVE = 348,                  /* NATIVE  */
    INLINE = 349,                  /* INLINE  */
    TYPEMAP = 350,                 /* TYPEMAP  */
    EXCEPT = 351,                  /* EXCEPT  */
    ECHO = 352,                    /* ECHO  */
    APPLY = 353,                   /* APPLY  */
    CLEAR = 354,                   /* CLEAR  */
    SWIGTEMPLATE = 355,            /* SWIGTEMPLATE  */
    FRAGMENT = 356,                /* FRAGMENT  */
    WARN = 357,                    /* WARN  */
    LESSTHAN = 358,                /* LESSTHAN  */
    GREATERTHAN = 359,             /* GREATERTHAN  */
    DELETE_KW = 360,               /* DELETE_KW  */
    DEFAULT = 361,                 /* DEFAULT  */
    LESSTHANOREQUALTO = 362,       /* LESSTHANOREQUALTO  */
    GREATERTHANOREQUALTO = 363,    /* GREATERTHANOREQUALTO  */
    EQUALTO = 364,                 /* EQUALTO  */
    NOTEQUALTO = 365,              /* NOTEQUALTO  */
    ARROW = 366,                   /* ARROW  */
    QUESTIONMARK = 367,            /* QUESTIONMARK  */
    TYPES = 368,                   /* TYPES  */
    PARMS = 369,                   /* PARMS  */
    NONID = 370,                   /* NONID  */
    DSTAR = 371,                   /* DSTAR  */
    DCNOT = 372,                   /* DCNOT  */
    TEMPLATE = 373,                /* TEMPLATE  */
    OPERATOR = 374,                /* OPERATOR  */
    CONVERSIONOPERATOR = 375,      /* CONVERSIONOPERATOR  */
    PARSETYPE = 376,               /* PARSETYPE  */
    PARSEPARM = 377,               /* PARSEPARM  */
    PARSEPARMS = 378,              /* PARSEPARMS  */
    DOXYGENSTRING = 379,           /* DOXYGENSTRING  */
    DOXYGENPOSTSTRING = 380,       /* DOXYGENPOSTSTRING  */
    CAST = 381,                    /* CAST  */
    LOR = 382,                     /* LOR  */
    LAND = 383,                    /* LAND  */
    OR = 384,                      /* OR  */
    XOR = 385,                     /* XOR  */
    AND = 386,                     /* AND  */
    LSHIFT = 387,                  /* LSHIFT  */
    RSHIFT = 388,                  /* RSHIFT  */
    PLUS = 389,                    /* PLUS  */
    MINUS = 390,                   /* MINUS  */
    STAR = 391,                    /* STAR  */
    SLASH = 392,                   /* SLASH  */
    MODULO = 393,                  /* MODULO  */
    UMINUS = 394,                  /* UMINUS  */
    NOT = 395,                     /* NOT  */
    LNOT = 396,                    /* LNOT  */
    DCOLON = 397                   /* DCOLON  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif
/* Token kinds.  */
#define YYEMPTY -2
#define YYEOF 0
#define YYerror 256
#define YYUNDEF 257
#define ID 258
#define HBLOCK 259
#define POUND 260
#define STRING 261
#define WSTRING 262
#define INCLUDE 263
#define IMPORT 264
#define INSERT 265
#define CHARCONST 266
#define WCHARCONST 267
#define NUM_INT 268
#define NUM_FLOAT 269
#define NUM_UNSIGNED 270
#define NUM_LONG 271
#define NUM_ULONG 272
#define NUM_LONGLONG 273
#define NUM_ULONGLONG 274
#define NUM_BOOL 275
#define TYPEDEF 276
#define TYPE_INT 277
#define TYPE_UNSIGNED 278
#define TYPE_SHORT 279
#define TYPE_LONG 280
#define TYPE_FLOAT 281
#define TYPE_DOUBLE 282
#define TYPE_CHAR 283
#define TYPE_WCHAR 284
#define TYPE_VOID 285
#define TYPE_SIGNED 286
#define TYPE_BOOL 287
#define TYPE_COMPLEX 288
#define TYPE_TYPEDEF 289
#define TYPE_RAW 290
#define TYPE_NON_ISO_INT8 291
#define TYPE_NON_ISO_INT16 292
#define TYPE_NON_ISO_INT32 293
#define TYPE_NON_ISO_INT64 294
#define LPAREN 295
#define RPAREN 296
#define COMMA 297
#define SEMI 298
#define EXTERN 299
#define INIT 300
#define LBRACE 301
#define RBRACE 302
#define PERIOD 303
#define CONST_QUAL 304
#define VOLATILE 305
#define REGISTER 306
#define STRUCT 307
#define UNION 308
#define EQUAL 309
#define SIZEOF 310
#define MODULE 311
#define LBRACKET 312
#define RBRACKET 313
#define BEGINFILE 314
#define ENDOFFILE 315
#define ILLEGAL 316
#define CONSTANT 317
#define NAME 318
#define RENAME 319
#define NAMEWARN 320
#define EXTEND 321
#define PRAGMA 322
#define FEATURE 323
#define VARARGS 324
#define ENUM 325
#define CLASS 326
#define TYPENAME 327
#define PRIVATE 328
#define PUBLIC 329
#define PROTECTED 330
#define COLON 331
#define STATIC 332
#define VIRTUAL 333
#define FRIEND 334
#define THROW 335
#define CATCH 336
#define EXPLICIT 337
#define STATIC_ASSERT 338
#define CONSTEXPR 339
#define THREAD_LOCAL 340
#define DECLTYPE 341
#define AUTO 342
#define NOEXCEPT 343
#define OVERRIDE 344
#define FINAL 345
#define USING 346
#define NAMESPACE 347
#define NATIVE 348
#define INLINE 349
#define TYPEMAP 350
#define EXCEPT 351
#define ECHO 352
#define APPLY 353
#define CLEAR 354
#define SWIGTEMPLATE 355
#define FRAGMENT 356
#define WARN 357
#define LESSTHAN 358
#define GREATERTHAN 359
#define DELETE_KW 360
#define DEFAULT 361
#define LESSTHANOREQUALTO 362
#define GREATERTHANOREQUALTO 363
#define EQUALTO 364
#define NOTEQUALTO 365
#define ARROW 366
#define QUESTIONMARK 367
#define TYPES 368
#define PARMS 369
#define NONID 370
#define DSTAR 371
#define DCNOT 372
#define TEMPLATE 373
#define OPERATOR 374
#define CONVERSIONOPERATOR 375
#define PARSETYPE 376
#define PARSEPARM 377
#define PARSEPARMS 378
#define DOXYGENSTRING 379
#define DOXYGENPOSTSTRING 380
#define CAST 381
#define LOR 382
#define LAND 383
#define OR 384
#define XOR 385
#define AND 386
#define LSHIFT 387
#define RSHIFT 388
#define PLUS 389
#define MINUS 390
#define STAR 391
#define SLASH 392
#define MODULO 393
#define UMINUS 394
#define NOT 395
#define LNOT 396
#define DCOLON 397

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 1542 "../../Source/CParse/parser.y"

  const char  *id;
  List  *bases;
  struct Define {
    String *val;
    String *rawval;
    int     type;
    String *qualifier;
    String *refqualifier;
    String *bitfield;
    Parm   *throws;
    String *throwf;
    String *nexcept;
    String *final;
  } dtype;
  struct {
    const char *type;
    String *filename;
    int   line;
  } loc;
  struct {
    char      *id;
    SwigType  *type;
    String    *defarg;
    ParmList  *parms;
    short      have_parms;
    ParmList  *throws;
    String    *throwf;
    String    *nexcept;
    String    *final;
  } decl;
  Parm         *tparms;
  struct {
    String     *method;
    Hash       *kwargs;
  } tmap;
  struct {
    String     *type;
    String     *us;
  } ptype;
  SwigType     *type;
  String       *str;
  Parm         *p;
  ParmList     *pl;
  int           intvalue;
  Node         *node;

#line 399 "CParse/parser.h"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;


int yyparse (void);


#endif /* !YY_YY_CPARSE_PARSER_H_INCLUDED  */
