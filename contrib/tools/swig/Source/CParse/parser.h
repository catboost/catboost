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
    END = 0,                       /* END  */
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
    ELLIPSIS = 304,                /* ELLIPSIS  */
    CONST_QUAL = 305,              /* CONST_QUAL  */
    VOLATILE = 306,                /* VOLATILE  */
    REGISTER = 307,                /* REGISTER  */
    STRUCT = 308,                  /* STRUCT  */
    UNION = 309,                   /* UNION  */
    EQUAL = 310,                   /* EQUAL  */
    SIZEOF = 311,                  /* SIZEOF  */
    MODULE = 312,                  /* MODULE  */
    LBRACKET = 313,                /* LBRACKET  */
    RBRACKET = 314,                /* RBRACKET  */
    BEGINFILE = 315,               /* BEGINFILE  */
    ENDOFFILE = 316,               /* ENDOFFILE  */
    ILLEGAL = 317,                 /* ILLEGAL  */
    CONSTANT = 318,                /* CONSTANT  */
    NAME = 319,                    /* NAME  */
    RENAME = 320,                  /* RENAME  */
    NAMEWARN = 321,                /* NAMEWARN  */
    EXTEND = 322,                  /* EXTEND  */
    PRAGMA = 323,                  /* PRAGMA  */
    FEATURE = 324,                 /* FEATURE  */
    VARARGS = 325,                 /* VARARGS  */
    ENUM = 326,                    /* ENUM  */
    CLASS = 327,                   /* CLASS  */
    TYPENAME = 328,                /* TYPENAME  */
    PRIVATE = 329,                 /* PRIVATE  */
    PUBLIC = 330,                  /* PUBLIC  */
    PROTECTED = 331,               /* PROTECTED  */
    COLON = 332,                   /* COLON  */
    STATIC = 333,                  /* STATIC  */
    VIRTUAL = 334,                 /* VIRTUAL  */
    FRIEND = 335,                  /* FRIEND  */
    THROW = 336,                   /* THROW  */
    CATCH = 337,                   /* CATCH  */
    EXPLICIT = 338,                /* EXPLICIT  */
    STATIC_ASSERT = 339,           /* STATIC_ASSERT  */
    CONSTEXPR = 340,               /* CONSTEXPR  */
    THREAD_LOCAL = 341,            /* THREAD_LOCAL  */
    DECLTYPE = 342,                /* DECLTYPE  */
    AUTO = 343,                    /* AUTO  */
    NOEXCEPT = 344,                /* NOEXCEPT  */
    OVERRIDE = 345,                /* OVERRIDE  */
    FINAL = 346,                   /* FINAL  */
    USING = 347,                   /* USING  */
    NAMESPACE = 348,               /* NAMESPACE  */
    NATIVE = 349,                  /* NATIVE  */
    INLINE = 350,                  /* INLINE  */
    TYPEMAP = 351,                 /* TYPEMAP  */
    EXCEPT = 352,                  /* EXCEPT  */
    ECHO = 353,                    /* ECHO  */
    APPLY = 354,                   /* APPLY  */
    CLEAR = 355,                   /* CLEAR  */
    SWIGTEMPLATE = 356,            /* SWIGTEMPLATE  */
    FRAGMENT = 357,                /* FRAGMENT  */
    WARN = 358,                    /* WARN  */
    LESSTHAN = 359,                /* LESSTHAN  */
    GREATERTHAN = 360,             /* GREATERTHAN  */
    DELETE_KW = 361,               /* DELETE_KW  */
    DEFAULT = 362,                 /* DEFAULT  */
    LESSTHANOREQUALTO = 363,       /* LESSTHANOREQUALTO  */
    GREATERTHANOREQUALTO = 364,    /* GREATERTHANOREQUALTO  */
    EQUALTO = 365,                 /* EQUALTO  */
    NOTEQUALTO = 366,              /* NOTEQUALTO  */
    LESSEQUALGREATER = 367,        /* LESSEQUALGREATER  */
    ARROW = 368,                   /* ARROW  */
    QUESTIONMARK = 369,            /* QUESTIONMARK  */
    TYPES = 370,                   /* TYPES  */
    PARMS = 371,                   /* PARMS  */
    NONID = 372,                   /* NONID  */
    DSTAR = 373,                   /* DSTAR  */
    DCNOT = 374,                   /* DCNOT  */
    TEMPLATE = 375,                /* TEMPLATE  */
    OPERATOR = 376,                /* OPERATOR  */
    CONVERSIONOPERATOR = 377,      /* CONVERSIONOPERATOR  */
    PARSETYPE = 378,               /* PARSETYPE  */
    PARSEPARM = 379,               /* PARSEPARM  */
    PARSEPARMS = 380,              /* PARSEPARMS  */
    DOXYGENSTRING = 381,           /* DOXYGENSTRING  */
    DOXYGENPOSTSTRING = 382,       /* DOXYGENPOSTSTRING  */
    CAST = 383,                    /* CAST  */
    LOR = 384,                     /* LOR  */
    LAND = 385,                    /* LAND  */
    OR = 386,                      /* OR  */
    XOR = 387,                     /* XOR  */
    AND = 388,                     /* AND  */
    LSHIFT = 389,                  /* LSHIFT  */
    RSHIFT = 390,                  /* RSHIFT  */
    PLUS = 391,                    /* PLUS  */
    MINUS = 392,                   /* MINUS  */
    STAR = 393,                    /* STAR  */
    SLASH = 394,                   /* SLASH  */
    MODULO = 395,                  /* MODULO  */
    UMINUS = 396,                  /* UMINUS  */
    NOT = 397,                     /* NOT  */
    LNOT = 398,                    /* LNOT  */
    DCOLON = 399                   /* DCOLON  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif
/* Token kinds.  */
#define YYEMPTY -2
#define END 0
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
#define ELLIPSIS 304
#define CONST_QUAL 305
#define VOLATILE 306
#define REGISTER 307
#define STRUCT 308
#define UNION 309
#define EQUAL 310
#define SIZEOF 311
#define MODULE 312
#define LBRACKET 313
#define RBRACKET 314
#define BEGINFILE 315
#define ENDOFFILE 316
#define ILLEGAL 317
#define CONSTANT 318
#define NAME 319
#define RENAME 320
#define NAMEWARN 321
#define EXTEND 322
#define PRAGMA 323
#define FEATURE 324
#define VARARGS 325
#define ENUM 326
#define CLASS 327
#define TYPENAME 328
#define PRIVATE 329
#define PUBLIC 330
#define PROTECTED 331
#define COLON 332
#define STATIC 333
#define VIRTUAL 334
#define FRIEND 335
#define THROW 336
#define CATCH 337
#define EXPLICIT 338
#define STATIC_ASSERT 339
#define CONSTEXPR 340
#define THREAD_LOCAL 341
#define DECLTYPE 342
#define AUTO 343
#define NOEXCEPT 344
#define OVERRIDE 345
#define FINAL 346
#define USING 347
#define NAMESPACE 348
#define NATIVE 349
#define INLINE 350
#define TYPEMAP 351
#define EXCEPT 352
#define ECHO 353
#define APPLY 354
#define CLEAR 355
#define SWIGTEMPLATE 356
#define FRAGMENT 357
#define WARN 358
#define LESSTHAN 359
#define GREATERTHAN 360
#define DELETE_KW 361
#define DEFAULT 362
#define LESSTHANOREQUALTO 363
#define GREATERTHANOREQUALTO 364
#define EQUALTO 365
#define NOTEQUALTO 366
#define LESSEQUALGREATER 367
#define ARROW 368
#define QUESTIONMARK 369
#define TYPES 370
#define PARMS 371
#define NONID 372
#define DSTAR 373
#define DCNOT 374
#define TEMPLATE 375
#define OPERATOR 376
#define CONVERSIONOPERATOR 377
#define PARSETYPE 378
#define PARSEPARM 379
#define PARSEPARMS 380
#define DOXYGENSTRING 381
#define DOXYGENPOSTSTRING 382
#define CAST 383
#define LOR 384
#define LAND 385
#define OR 386
#define XOR 387
#define AND 388
#define LSHIFT 389
#define RSHIFT 390
#define PLUS 391
#define MINUS 392
#define STAR 393
#define SLASH 394
#define MODULO 395
#define UMINUS 396
#define NOT 397
#define LNOT 398
#define DCOLON 399

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 1554 "../../Source/CParse/parser.y"

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

#line 403 "CParse/parser.h"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;


int yyparse (void);


#endif /* !YY_YY_CPARSE_PARSER_H_INCLUDED  */
