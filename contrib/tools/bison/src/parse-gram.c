/* A Bison parser, made by GNU Bison 3.6.4.  */

/* Bison implementation for Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.6.4"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 2

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* "%code top" blocks.  */
#line 27 "src/parse-gram.y"

  /* On column 0 to please syntax-check.  */
#include <config.h>

#line 73 "src/parse-gram.c"
/* Substitute the type names.  */
#define YYSTYPE         GRAM_STYPE
#define YYLTYPE         GRAM_LTYPE
/* Substitute the variable and function names.  */
#define yyparse         gram_parse
#define yylex           gram_lex
#define yyerror         gram_error
#define yydebug         gram_debug
#define yynerrs         gram_nerrs


# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

#include "parse-gram.h"
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_STRING = 3,                     /* "string"  */
  YYSYMBOL_TSTRING = 4,                    /* "translatable string"  */
  YYSYMBOL_PERCENT_TOKEN = 5,              /* "%token"  */
  YYSYMBOL_PERCENT_NTERM = 6,              /* "%nterm"  */
  YYSYMBOL_PERCENT_TYPE = 7,               /* "%type"  */
  YYSYMBOL_PERCENT_DESTRUCTOR = 8,         /* "%destructor"  */
  YYSYMBOL_PERCENT_PRINTER = 9,            /* "%printer"  */
  YYSYMBOL_PERCENT_LEFT = 10,              /* "%left"  */
  YYSYMBOL_PERCENT_RIGHT = 11,             /* "%right"  */
  YYSYMBOL_PERCENT_NONASSOC = 12,          /* "%nonassoc"  */
  YYSYMBOL_PERCENT_PRECEDENCE = 13,        /* "%precedence"  */
  YYSYMBOL_PERCENT_PREC = 14,              /* "%prec"  */
  YYSYMBOL_PERCENT_DPREC = 15,             /* "%dprec"  */
  YYSYMBOL_PERCENT_MERGE = 16,             /* "%merge"  */
  YYSYMBOL_PERCENT_CODE = 17,              /* "%code"  */
  YYSYMBOL_PERCENT_DEFAULT_PREC = 18,      /* "%default-prec"  */
  YYSYMBOL_PERCENT_DEFINE = 19,            /* "%define"  */
  YYSYMBOL_PERCENT_DEFINES = 20,           /* "%defines"  */
  YYSYMBOL_PERCENT_ERROR_VERBOSE = 21,     /* "%error-verbose"  */
  YYSYMBOL_PERCENT_EXPECT = 22,            /* "%expect"  */
  YYSYMBOL_PERCENT_EXPECT_RR = 23,         /* "%expect-rr"  */
  YYSYMBOL_PERCENT_FLAG = 24,              /* "%<flag>"  */
  YYSYMBOL_PERCENT_FILE_PREFIX = 25,       /* "%file-prefix"  */
  YYSYMBOL_PERCENT_GLR_PARSER = 26,        /* "%glr-parser"  */
  YYSYMBOL_PERCENT_INITIAL_ACTION = 27,    /* "%initial-action"  */
  YYSYMBOL_PERCENT_LANGUAGE = 28,          /* "%language"  */
  YYSYMBOL_PERCENT_NAME_PREFIX = 29,       /* "%name-prefix"  */
  YYSYMBOL_PERCENT_NO_DEFAULT_PREC = 30,   /* "%no-default-prec"  */
  YYSYMBOL_PERCENT_NO_LINES = 31,          /* "%no-lines"  */
  YYSYMBOL_PERCENT_NONDETERMINISTIC_PARSER = 32, /* "%nondeterministic-parser"  */
  YYSYMBOL_PERCENT_OUTPUT = 33,            /* "%output"  */
  YYSYMBOL_PERCENT_PURE_PARSER = 34,       /* "%pure-parser"  */
  YYSYMBOL_PERCENT_REQUIRE = 35,           /* "%require"  */
  YYSYMBOL_PERCENT_SKELETON = 36,          /* "%skeleton"  */
  YYSYMBOL_PERCENT_START = 37,             /* "%start"  */
  YYSYMBOL_PERCENT_TOKEN_TABLE = 38,       /* "%token-table"  */
  YYSYMBOL_PERCENT_VERBOSE = 39,           /* "%verbose"  */
  YYSYMBOL_PERCENT_YACC = 40,              /* "%yacc"  */
  YYSYMBOL_BRACED_CODE = 41,               /* "{...}"  */
  YYSYMBOL_BRACED_PREDICATE = 42,          /* "%?{...}"  */
  YYSYMBOL_BRACKETED_ID = 43,              /* "[identifier]"  */
  YYSYMBOL_CHAR = 44,                      /* "character literal"  */
  YYSYMBOL_COLON = 45,                     /* ":"  */
  YYSYMBOL_EPILOGUE = 46,                  /* "epilogue"  */
  YYSYMBOL_EQUAL = 47,                     /* "="  */
  YYSYMBOL_ID = 48,                        /* "identifier"  */
  YYSYMBOL_ID_COLON = 49,                  /* "identifier:"  */
  YYSYMBOL_PERCENT_PERCENT = 50,           /* "%%"  */
  YYSYMBOL_PIPE = 51,                      /* "|"  */
  YYSYMBOL_PROLOGUE = 52,                  /* "%{...%}"  */
  YYSYMBOL_SEMICOLON = 53,                 /* ";"  */
  YYSYMBOL_TAG = 54,                       /* "<tag>"  */
  YYSYMBOL_TAG_ANY = 55,                   /* "<*>"  */
  YYSYMBOL_TAG_NONE = 56,                  /* "<>"  */
  YYSYMBOL_INT = 57,                       /* "integer literal"  */
  YYSYMBOL_PERCENT_PARAM = 58,             /* "%param"  */
  YYSYMBOL_PERCENT_UNION = 59,             /* "%union"  */
  YYSYMBOL_PERCENT_EMPTY = 60,             /* "%empty"  */
  YYSYMBOL_YYACCEPT = 61,                  /* $accept  */
  YYSYMBOL_input = 62,                     /* input  */
  YYSYMBOL_prologue_declarations = 63,     /* prologue_declarations  */
  YYSYMBOL_prologue_declaration = 64,      /* prologue_declaration  */
  YYSYMBOL_65_1 = 65,                      /* $@1  */
  YYSYMBOL_params = 66,                    /* params  */
  YYSYMBOL_grammar_declaration = 67,       /* grammar_declaration  */
  YYSYMBOL_code_props_type = 68,           /* code_props_type  */
  YYSYMBOL_union_name = 69,                /* union_name  */
  YYSYMBOL_symbol_declaration = 70,        /* symbol_declaration  */
  YYSYMBOL_71_2 = 71,                      /* $@2  */
  YYSYMBOL_72_3 = 72,                      /* $@3  */
  YYSYMBOL_precedence_declarator = 73,     /* precedence_declarator  */
  YYSYMBOL_74_tag_opt = 74,                /* tag.opt  */
  YYSYMBOL_generic_symlist = 75,           /* generic_symlist  */
  YYSYMBOL_generic_symlist_item = 76,      /* generic_symlist_item  */
  YYSYMBOL_tag = 77,                       /* tag  */
  YYSYMBOL_nterm_decls = 78,               /* nterm_decls  */
  YYSYMBOL_token_decls = 79,               /* token_decls  */
  YYSYMBOL_80_token_decl_1 = 80,           /* token_decl.1  */
  YYSYMBOL_token_decl = 81,                /* token_decl  */
  YYSYMBOL_82_int_opt = 82,                /* int.opt  */
  YYSYMBOL_alias = 83,                     /* alias  */
  YYSYMBOL_token_decls_for_prec = 84,      /* token_decls_for_prec  */
  YYSYMBOL_85_token_decl_for_prec_1 = 85,  /* token_decl_for_prec.1  */
  YYSYMBOL_token_decl_for_prec = 86,       /* token_decl_for_prec  */
  YYSYMBOL_symbol_decls = 87,              /* symbol_decls  */
  YYSYMBOL_88_symbol_decl_1 = 88,          /* symbol_decl.1  */
  YYSYMBOL_grammar = 89,                   /* grammar  */
  YYSYMBOL_rules_or_grammar_declaration = 90, /* rules_or_grammar_declaration  */
  YYSYMBOL_rules = 91,                     /* rules  */
  YYSYMBOL_92_4 = 92,                      /* $@4  */
  YYSYMBOL_93_rhses_1 = 93,                /* rhses.1  */
  YYSYMBOL_rhs = 94,                       /* rhs  */
  YYSYMBOL_95_named_ref_opt = 95,          /* named_ref.opt  */
  YYSYMBOL_variable = 96,                  /* variable  */
  YYSYMBOL_value = 97,                     /* value  */
  YYSYMBOL_id = 98,                        /* id  */
  YYSYMBOL_id_colon = 99,                  /* id_colon  */
  YYSYMBOL_symbol = 100,                   /* symbol  */
  YYSYMBOL_string_as_id = 101,             /* string_as_id  */
  YYSYMBOL_102_epilogue_opt = 102          /* epilogue.opt  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;



/* Unqualified %code blocks.  */
#line 33 "src/parse-gram.y"

  #include "system.h"

  #include <c-ctype.h>
  #include <errno.h>
  #include <intprops.h>
  #include <quotearg.h>
  #include <vasnprintf.h>
  #include <xmemdup0.h>

  #include "complain.h"
  #include "conflicts.h"
  #include "files.h"
  #include "getargs.h"
  #include "gram.h"
  #include "named-ref.h"
  #include "reader.h"
  #include "scan-code.h"
  #include "scan-gram.h"

  static int current_prec = 0;
  static location current_lhs_loc;
  static named_ref *current_lhs_named_ref;
  static symbol *current_lhs_symbol;
  static symbol_class current_class = unknown_sym;

  /** Set the new current left-hand side symbol, possibly common
   * to several right-hand side parts of rule.
   */
  static void current_lhs (symbol *sym, location loc, named_ref *ref);

  #define YYLLOC_DEFAULT(Current, Rhs, N)         \
    (Current) = lloc_default (Rhs, N)
  static YYLTYPE lloc_default (YYLTYPE const *, int);

  #define YY_LOCATION_PRINT(File, Loc)            \
    location_print (Loc, File)

  /* Strip initial '{' and final '}' (must be first and last characters).
     Return the result.  */
  static char *strip_braces (char *code);

  /* Convert CODE by calling code_props_plain_init if PLAIN, otherwise
     code_props_symbol_action_init.  Call
     gram_scanner_last_string_free to release the latest string from
     the scanner (should be CODE). */
  static char const *translate_code (char *code, location loc, bool plain);

  /* Convert CODE by calling code_props_plain_init after having
     stripped the first and last characters (expected to be '{', and
     '}').  Call gram_scanner_last_string_free to release the latest
     string from the scanner (should be CODE). */
  static char const *translate_code_braceless (char *code, location loc);

  /* Handle a %error-verbose directive.  */
  static void handle_error_verbose (location const *loc, char const *directive);

  /* Handle a %file-prefix directive.  */
  static void handle_file_prefix (location const *loc,
                                  location const *dir_loc,
                                  char const *directive, char const *value);

  /* Handle a %name-prefix directive.  */
  static void handle_name_prefix (location const *loc,
                                  char const *directive, char const *value);

  /* Handle a %pure-parser directive.  */
  static void handle_pure_parser (location const *loc, char const *directive);

  /* Handle a %require directive.  */
  static void handle_require (location const *loc, char const *version);

  /* Handle a %skeleton directive.  */
  static void handle_skeleton (location const *loc, char const *skel);

  /* Handle a %yacc directive.  */
  static void handle_yacc (location const *loc);

  /* Implementation of yyerror.  */
  static void gram_error (location const *, char const *);

  /* A string that describes a char (e.g., 'a' -> "'a'").  */
  static char const *char_name (char);

  /* Add style to semantic values in traces.  */
  static void tron (FILE *yyo);
  static void troff (FILE *yyo);
#line 255 "src/parse-gram.y"

  /** Add a lex-param and/or a parse-param.
   *
   * \param type  where to push this formal argument.
   * \param decl  the formal argument.  Destroyed.
   * \param loc   the location in the source.
   */
  static void add_param (param_type type, char *decl, location loc);
  static param_type current_param = param_none;

#line 319 "src/parse-gram.c"

#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_uint8 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

#ifndef N_
# define N_(Msgid) Msgid
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && ! defined __ICC && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                            \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if 1

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
# define YYCOPY_NEEDED 1
#endif /* 1 */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined GRAM_LTYPE_IS_TRIVIAL && GRAM_LTYPE_IS_TRIVIAL \
             && defined GRAM_STYPE_IS_TRIVIAL && GRAM_STYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
  YYLTYPE yyls_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE) \
             + YYSIZEOF (YYLTYPE)) \
      + 2 * YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   234

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  61
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  42
/* YYNRULES -- Number of rules.  */
#define YYNRULES  123
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  167

#define YYMAXUTOK   315


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX) YY_CAST (yysymbol_kind_t, YYX)

#if GRAM_DEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,   287,   287,   296,   297,   301,   302,   308,   312,   317,
     318,   323,   324,   325,   326,   327,   332,   337,   338,   339,
     340,   341,   342,   342,   343,   344,   345,   346,   347,   348,
     349,   350,   354,   355,   364,   365,   369,   380,   384,   388,
     396,   406,   407,   417,   418,   424,   437,   437,   442,   442,
     447,   451,   461,   462,   463,   464,   468,   469,   474,   475,
     479,   480,   484,   485,   486,   499,   508,   512,   516,   524,
     525,   529,   542,   543,   548,   549,   550,   568,   572,   576,
     584,   586,   591,   598,   608,   612,   616,   624,   629,   641,
     642,   648,   649,   650,   657,   657,   665,   666,   667,   672,
     675,   677,   679,   681,   683,   685,   687,   689,   691,   696,
     697,   706,   730,   731,   732,   733,   745,   747,   771,   776,
     777,   782,   790,   791
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if 1
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  static const char *const yy_sname[] =
  {
  N_("end of file"), N_("error"), N_("invalid token"), N_("string"),
  N_("translatable string"), "%token", "%nterm", "%type", "%destructor",
  "%printer", "%left", "%right", "%nonassoc", "%precedence", "%prec",
  "%dprec", "%merge", "%code", "%default-prec", "%define", "%defines",
  "%error-verbose", "%expect", "%expect-rr", "%<flag>", "%file-prefix",
  "%glr-parser", "%initial-action", "%language", "%name-prefix",
  "%no-default-prec", "%no-lines", "%nondeterministic-parser", "%output",
  "%pure-parser", "%require", "%skeleton", "%start", "%token-table",
  "%verbose", "%yacc", "{...}", "%?{...}", N_("[identifier]"),
  N_("character literal"), ":", N_("epilogue"), "=", N_("identifier"),
  N_("identifier:"), "%%", "|", "%{...%}", ";", N_("<tag>"), "<*>", "<>",
  N_("integer literal"), "%param", "%union", "%empty", "$accept", "input",
  "prologue_declarations", "prologue_declaration", "$@1", "params",
  "grammar_declaration", "code_props_type", "union_name",
  "symbol_declaration", "$@2", "$@3", "precedence_declarator", "tag.opt",
  "generic_symlist", "generic_symlist_item", "tag", "nterm_decls",
  "token_decls", "token_decl.1", "token_decl", "int.opt", "alias",
  "token_decls_for_prec", "token_decl_for_prec.1", "token_decl_for_prec",
  "symbol_decls", "symbol_decl.1", "grammar",
  "rules_or_grammar_declaration", "rules", "$@4", "rhses.1", "rhs",
  "named_ref.opt", "variable", "value", "id", "id_colon", "symbol",
  "string_as_id", "epilogue.opt", YY_NULLPTR
  };
  /* YYTRANSLATABLE[SYMBOL-NUM] -- Whether YY_SNAME[SYMBOL-NUM] is
     internationalizable.  */
  static yytype_int8 yytranslatable[] =
  {
       1,     1,     1,     1,     1,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     1,     1,     0,     1,     0,     1,     1,
       0,     0,     0,     0,     1,     0,     0,     1,     0,     0,
       0
  };
  return (yysymbol < YYNTOKENS && yytranslatable[yysymbol]
          ? _(yy_sname[yysymbol])
          : yy_sname[yysymbol]);
}
#endif

#ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_int16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315
};
#endif

#define YYPACT_NINF (-80)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-123)

#define yytable_value_is_error(Yyn) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     -80,     5,   120,   -80,   -17,   -80,   -80,    23,   -80,   -80,
     -80,   -80,   -80,   -80,    -7,   -80,    10,    58,   -80,     6,
      34,   -80,    82,   -80,    54,    95,   101,   -80,   -80,   -80,
     103,   -80,   105,   107,    24,   -80,   -80,   -80,   175,   -80,
     -80,   -80,    64,   -80,   -80,    75,   -80,    61,   -80,   -15,
     -15,   -80,   -80,   -80,    24,    63,    24,   -80,   -80,   -80,
     -80,    77,   -80,    46,   -80,   -80,   -80,   -80,   -80,   -80,
     -80,   -80,   -80,   -80,   -80,    67,   -80,    69,     7,   -80,
     -80,    80,    93,   -80,    94,    -1,    24,   108,    24,   -80,
      79,   -80,   -38,   109,   -38,   -80,    79,   -80,   109,    24,
      24,   -80,   -80,   -80,   -80,   -80,   -80,   -80,   -80,   115,
     -80,   -80,   -80,   -80,   -80,   123,   -80,   -80,   -80,   -80,
      -1,   -80,   -80,   -80,    24,    24,   -80,   -80,   -80,   -38,
     -38,   -80,    28,    24,   -80,   121,   -80,   -80,    24,   -38,
     -80,   -80,   -80,   -80,   -23,    59,   -80,   -80,    24,   110,
     111,   112,   114,   -80,   -80,   -80,   127,    80,    59,   -80,
     -80,   -80,   -80,   -80,    80,   -80,   -80
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_int8 yydefact[] =
{
       3,     0,     0,     1,     0,    48,    46,     0,    41,    42,
      52,    53,    54,    55,     0,    37,     0,     9,    11,     0,
       0,     7,     0,    15,     0,     0,     0,    38,    19,    20,
       0,    24,     0,     0,     0,    27,    28,    29,     0,     6,
      31,    22,    43,     4,     5,     0,    34,     0,    30,     0,
       0,   121,   117,   116,     0,    50,    84,   119,    87,   120,
      39,     0,   111,   112,    10,    12,    13,    14,    16,    17,
      18,    21,    25,    26,    35,     0,   118,     0,     0,    89,
      91,   109,     0,    44,     0,     0,     0,    51,    77,    80,
      72,    83,     0,    49,    66,    69,    72,    47,    65,    85,
       0,    88,    40,   114,   115,   113,     8,    93,    92,     0,
      90,     2,   110,    94,    33,    23,    45,    62,    63,    64,
      36,    58,    61,    60,    78,     0,    81,    73,    82,    67,
       0,    70,    74,    86,   123,     0,    32,    59,    79,    68,
      76,    71,    75,    99,    95,    96,    99,    98,     0,     0,
       0,     0,     0,   102,    57,   103,     0,   109,    97,   104,
     105,   106,   107,   108,   109,   100,   101
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
     -80,   -80,   -80,   -80,   -80,   -80,   172,   -80,   -80,   -80,
     -80,   -80,   -80,   -80,   -80,    55,   -80,   -80,   139,   -54,
     -59,    81,   -80,   -80,   -65,   -79,   -80,   -31,   -80,   113,
     -80,   -80,   -80,    44,   -67,   -80,   -80,   -46,   -80,   -34,
     -36,   -80
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    43,    82,   115,    77,    45,    84,    46,
      50,    49,    47,   156,   120,   121,   122,    97,    93,    94,
      95,   128,   141,    87,    88,    89,    55,    56,    78,    79,
      80,   135,   144,   145,   113,    63,   106,    57,    81,    58,
      59,   111
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      74,    90,    51,    96,    96,     3,    52,  -122,    75,   126,
      53,    91,     5,     6,     7,     8,     9,    10,    11,    12,
      13,   124,   101,    99,    14,    15,    51,    51,   146,    52,
     147,    51,   140,    53,    60,   131,    48,    27,   129,    92,
      90,    61,    90,    52,    34,   126,    96,    53,    96,   103,
      91,   123,    91,   117,   118,   119,    76,   109,    62,   126,
     138,    64,    51,    65,    51,   101,    42,    52,    52,   133,
     131,    53,    53,   148,   149,   150,   139,    54,    90,    90,
     131,   151,   152,    96,    96,    67,   123,   104,    91,    91,
     165,    66,    90,    96,   105,    68,   142,   166,    69,   101,
     -56,   153,    91,    52,    70,    52,    71,    53,    72,    53,
      73,   157,    83,   154,   159,    86,    85,   100,   102,   155,
     107,     4,   108,   112,   157,     5,     6,     7,     8,     9,
      10,    11,    12,    13,   114,   116,   127,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,   134,   125,   130,   136,   161,   143,   160,   164,   162,
      38,   163,    39,    40,    44,   137,    75,   132,    41,    42,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    98,
     158,   110,    14,    15,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    27,     0,     0,     0,     0,
       0,     0,    34,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    76,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    42
};

static const yytype_int16 yycheck[] =
{
      34,    47,     3,    49,    50,     0,    44,     0,     1,    88,
      48,    47,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    86,    56,    54,    17,    18,     3,     3,    51,    44,
      53,     3,     4,    48,    41,    94,    53,    30,    92,    54,
      86,    48,    88,    44,    37,   124,    92,    48,    94,     3,
      86,    85,    88,    54,    55,    56,    49,    50,    48,   138,
     125,     3,     3,    57,     3,    99,    59,    44,    44,   100,
     129,    48,    48,    14,    15,    16,   130,    54,   124,   125,
     139,    22,    23,   129,   130,     3,   120,    41,   124,   125,
     157,    57,   138,   139,    48,    41,   132,   164,     3,   133,
      41,    42,   138,    44,     3,    44,     3,    48,     3,    48,
       3,   145,    48,    54,   148,    54,    41,    54,    41,    60,
      53,     1,    53,    43,   158,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    41,    41,    57,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    37,    38,    39,
      40,    46,    54,    54,    41,    54,    45,    57,    41,    57,
      50,    57,    52,    53,     2,   120,     1,    96,    58,    59,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    50,
     146,    78,    17,    18,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    30,    -1,    -1,    -1,    -1,
      -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    49,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    59
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_int8 yystos[] =
{
       0,    62,    63,     0,     1,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    50,    52,
      53,    58,    59,    64,    67,    68,    70,    73,    53,    72,
      71,     3,    44,    48,    54,    87,    88,    98,   100,   101,
      41,    48,    48,    96,     3,    57,    57,     3,    41,     3,
       3,     3,     3,     3,   100,     1,    49,    67,    89,    90,
      91,    99,    65,    48,    69,    41,    54,    84,    85,    86,
      98,   101,    54,    79,    80,    81,    98,    78,    79,    88,
      54,   100,    41,     3,    41,    48,    97,    53,    53,    50,
      90,   102,    43,    95,    41,    66,    41,    54,    55,    56,
      75,    76,    77,   100,    85,    54,    86,    57,    82,    80,
      54,    81,    82,    88,    46,    92,    41,    76,    85,    80,
       4,    83,   101,    45,    93,    94,    51,    53,    14,    15,
      16,    22,    23,    42,    54,    60,    74,   100,    94,   100,
      57,    54,    57,    57,    41,    95,    95
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_int8 yyr1[] =
{
       0,    61,    62,    63,    63,    64,    64,    64,    64,    64,
      64,    64,    64,    64,    64,    64,    64,    64,    64,    64,
      64,    64,    65,    64,    64,    64,    64,    64,    64,    64,
      64,    64,    66,    66,    67,    67,    67,    67,    67,    67,
      67,    68,    68,    69,    69,    67,    71,    70,    72,    70,
      70,    70,    73,    73,    73,    73,    74,    74,    75,    75,
      76,    76,    77,    77,    77,    78,    79,    79,    79,    80,
      80,    81,    82,    82,    83,    83,    83,    84,    84,    84,
      85,    85,    86,    86,    87,    87,    87,    88,    88,    89,
      89,    90,    90,    90,    92,    91,    93,    93,    93,    94,
      94,    94,    94,    94,    94,    94,    94,    94,    94,    95,
      95,    96,    97,    97,    97,    97,    98,    98,    99,   100,
     100,   101,   102,   102
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     4,     0,     2,     1,     1,     1,     3,     1,
       2,     1,     2,     2,     2,     1,     2,     2,     2,     1,
       1,     2,     0,     3,     1,     2,     2,     1,     1,     1,
       2,     1,     2,     1,     1,     2,     3,     1,     1,     2,
       3,     1,     1,     0,     1,     3,     0,     3,     0,     3,
       2,     2,     1,     1,     1,     1,     0,     1,     1,     2,
       1,     1,     1,     1,     1,     1,     1,     2,     3,     1,
       2,     3,     0,     1,     0,     1,     1,     1,     2,     3,
       1,     2,     2,     1,     1,     2,     3,     1,     2,     1,
       2,     1,     2,     2,     0,     5,     1,     3,     2,     0,
       3,     4,     2,     2,     3,     3,     3,     3,     3,     0,
       1,     1,     0,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     0,     2
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = GRAM_EMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == GRAM_EMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        YY_LAC_DISCARD ("YYBACKUP");                              \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (&yylloc, YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use GRAM_error or GRAM_UNDEF. */
#define YYERRCODE GRAM_UNDEF

/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)                                \
    do                                                                  \
      if (N)                                                            \
        {                                                               \
          (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;        \
          (Current).first_column = YYRHSLOC (Rhs, 1).first_column;      \
          (Current).last_line    = YYRHSLOC (Rhs, N).last_line;         \
          (Current).last_column  = YYRHSLOC (Rhs, N).last_column;       \
        }                                                               \
      else                                                              \
        {                                                               \
          (Current).first_line   = (Current).last_line   =              \
            YYRHSLOC (Rhs, 0).last_line;                                \
          (Current).first_column = (Current).last_column =              \
            YYRHSLOC (Rhs, 0).last_column;                              \
        }                                                               \
    while (0)
#endif

#define YYRHSLOC(Rhs, K) ((Rhs)[K])


/* Enable debugging if requested.  */
#if GRAM_DEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

# ifndef YY_LOCATION_PRINT
#  if defined GRAM_LTYPE_IS_TRIVIAL && GRAM_LTYPE_IS_TRIVIAL

/* Print *YYLOCP on YYO.  Private, do not rely on its existence. */

YY_ATTRIBUTE_UNUSED
static int
yy_location_print_ (FILE *yyo, YYLTYPE const * const yylocp)
{
  int res = 0;
  int end_col = 0 != yylocp->last_column ? yylocp->last_column - 1 : 0;
  if (0 <= yylocp->first_line)
    {
      res += YYFPRINTF (yyo, "%d", yylocp->first_line);
      if (0 <= yylocp->first_column)
        res += YYFPRINTF (yyo, ".%d", yylocp->first_column);
    }
  if (0 <= yylocp->last_line)
    {
      if (yylocp->first_line < yylocp->last_line)
        {
          res += YYFPRINTF (yyo, "-%d", yylocp->last_line);
          if (0 <= end_col)
            res += YYFPRINTF (yyo, ".%d", end_col);
        }
      else if (0 <= end_col && yylocp->first_column < end_col)
        res += YYFPRINTF (yyo, "-%d", end_col);
    }
  return res;
 }

#   define YY_LOCATION_PRINT(File, Loc)          \
  yy_location_print_ (File, &(Loc))

#  else
#   define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#  endif
# endif /* !defined YY_LOCATION_PRINT */


# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value, Location); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
{
  FILE *yyoutput = yyo;
  YYUSE (yyoutput);
  YYUSE (yylocationp);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yykind < YYNTOKENS)
    YYPRINT (yyo, yytoknum[yykind], *yyvaluep);
# endif
/* "%code pre-printer" blocks.  */
#line 207 "src/parse-gram.y"
tron (yyo);

#line 1064 "src/parse-gram.c"
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  switch (yykind)
    {
    case 3: /* "string"  */
#line 214 "src/parse-gram.y"
         { fputs (((*yyvaluep).STRING), yyo); }
#line 1071 "src/parse-gram.c"
        break;

    case 4: /* "translatable string"  */
#line 214 "src/parse-gram.y"
         { fputs (((*yyvaluep).TSTRING), yyo); }
#line 1077 "src/parse-gram.c"
        break;

    case 21: /* "%error-verbose"  */
#line 221 "src/parse-gram.y"
         { fputs (((*yyvaluep).PERCENT_ERROR_VERBOSE), yyo); }
#line 1083 "src/parse-gram.c"
        break;

    case 24: /* "%<flag>"  */
#line 224 "src/parse-gram.y"
         { fprintf (yyo, "%%%s", ((*yyvaluep).PERCENT_FLAG)); }
#line 1089 "src/parse-gram.c"
        break;

    case 25: /* "%file-prefix"  */
#line 221 "src/parse-gram.y"
         { fputs (((*yyvaluep).PERCENT_FILE_PREFIX), yyo); }
#line 1095 "src/parse-gram.c"
        break;

    case 29: /* "%name-prefix"  */
#line 221 "src/parse-gram.y"
         { fputs (((*yyvaluep).PERCENT_NAME_PREFIX), yyo); }
#line 1101 "src/parse-gram.c"
        break;

    case 34: /* "%pure-parser"  */
#line 221 "src/parse-gram.y"
         { fputs (((*yyvaluep).PERCENT_PURE_PARSER), yyo); }
#line 1107 "src/parse-gram.c"
        break;

    case 41: /* "{...}"  */
#line 214 "src/parse-gram.y"
         { fputs (((*yyvaluep).BRACED_CODE), yyo); }
#line 1113 "src/parse-gram.c"
        break;

    case 42: /* "%?{...}"  */
#line 214 "src/parse-gram.y"
         { fputs (((*yyvaluep).BRACED_PREDICATE), yyo); }
#line 1119 "src/parse-gram.c"
        break;

    case 43: /* "[identifier]"  */
#line 222 "src/parse-gram.y"
         { fprintf (yyo, "[%s]", ((*yyvaluep).BRACKETED_ID)); }
#line 1125 "src/parse-gram.c"
        break;

    case 44: /* "character literal"  */
#line 211 "src/parse-gram.y"
         { fputs (char_name (((*yyvaluep).CHAR)), yyo); }
#line 1131 "src/parse-gram.c"
        break;

    case 46: /* "epilogue"  */
#line 214 "src/parse-gram.y"
         { fputs (((*yyvaluep).EPILOGUE), yyo); }
#line 1137 "src/parse-gram.c"
        break;

    case 48: /* "identifier"  */
#line 221 "src/parse-gram.y"
         { fputs (((*yyvaluep).ID), yyo); }
#line 1143 "src/parse-gram.c"
        break;

    case 49: /* "identifier:"  */
#line 223 "src/parse-gram.y"
         { fprintf (yyo, "%s:", ((*yyvaluep).ID_COLON)); }
#line 1149 "src/parse-gram.c"
        break;

    case 52: /* "%{...%}"  */
#line 214 "src/parse-gram.y"
         { fputs (((*yyvaluep).PROLOGUE), yyo); }
#line 1155 "src/parse-gram.c"
        break;

    case 54: /* "<tag>"  */
#line 225 "src/parse-gram.y"
         { fprintf (yyo, "<%s>", ((*yyvaluep).TAG)); }
#line 1161 "src/parse-gram.c"
        break;

    case 57: /* "integer literal"  */
#line 228 "src/parse-gram.y"
         { fprintf (yyo, "%d", ((*yyvaluep).INT)); }
#line 1167 "src/parse-gram.c"
        break;

    case 58: /* "%param"  */
#line 267 "src/parse-gram.y"
{
  switch (((*yyvaluep).PERCENT_PARAM))
    {
#define CASE(In, Out)                                           \
      case param_ ## In: fputs ("%" #Out, yyo); break
      CASE (lex,   lex-param);
      CASE (parse, parse-param);
      CASE (both,  param);
#undef CASE
      case param_none: aver (false); break;
    }
}
#line 1184 "src/parse-gram.c"
        break;

    case 68: /* code_props_type  */
#line 404 "src/parse-gram.y"
         { fprintf (yyo, "%s", code_props_type_string (((*yyvaluep).code_props_type))); }
#line 1190 "src/parse-gram.c"
        break;

    case 74: /* tag.opt  */
#line 221 "src/parse-gram.y"
         { fputs (((*yyvaluep).yykind_74), yyo); }
#line 1196 "src/parse-gram.c"
        break;

    case 75: /* generic_symlist  */
#line 237 "src/parse-gram.y"
         { symbol_list_syms_print (((*yyvaluep).generic_symlist), yyo); }
#line 1202 "src/parse-gram.c"
        break;

    case 76: /* generic_symlist_item  */
#line 237 "src/parse-gram.y"
         { symbol_list_syms_print (((*yyvaluep).generic_symlist_item), yyo); }
#line 1208 "src/parse-gram.c"
        break;

    case 77: /* tag  */
#line 225 "src/parse-gram.y"
         { fprintf (yyo, "<%s>", ((*yyvaluep).tag)); }
#line 1214 "src/parse-gram.c"
        break;

    case 78: /* nterm_decls  */
#line 237 "src/parse-gram.y"
         { symbol_list_syms_print (((*yyvaluep).nterm_decls), yyo); }
#line 1220 "src/parse-gram.c"
        break;

    case 79: /* token_decls  */
#line 237 "src/parse-gram.y"
         { symbol_list_syms_print (((*yyvaluep).token_decls), yyo); }
#line 1226 "src/parse-gram.c"
        break;

    case 80: /* token_decl.1  */
#line 237 "src/parse-gram.y"
         { symbol_list_syms_print (((*yyvaluep).yykind_80), yyo); }
#line 1232 "src/parse-gram.c"
        break;

    case 81: /* token_decl  */
#line 231 "src/parse-gram.y"
         { fprintf (yyo, "%s", ((*yyvaluep).token_decl) ? ((*yyvaluep).token_decl)->tag : "<NULL>"); }
#line 1238 "src/parse-gram.c"
        break;

    case 82: /* int.opt  */
#line 228 "src/parse-gram.y"
         { fprintf (yyo, "%d", ((*yyvaluep).yykind_82)); }
#line 1244 "src/parse-gram.c"
        break;

    case 83: /* alias  */
#line 231 "src/parse-gram.y"
         { fprintf (yyo, "%s", ((*yyvaluep).alias) ? ((*yyvaluep).alias)->tag : "<NULL>"); }
#line 1250 "src/parse-gram.c"
        break;

    case 84: /* token_decls_for_prec  */
#line 237 "src/parse-gram.y"
         { symbol_list_syms_print (((*yyvaluep).token_decls_for_prec), yyo); }
#line 1256 "src/parse-gram.c"
        break;

    case 85: /* token_decl_for_prec.1  */
#line 237 "src/parse-gram.y"
         { symbol_list_syms_print (((*yyvaluep).yykind_85), yyo); }
#line 1262 "src/parse-gram.c"
        break;

    case 86: /* token_decl_for_prec  */
#line 231 "src/parse-gram.y"
         { fprintf (yyo, "%s", ((*yyvaluep).token_decl_for_prec) ? ((*yyvaluep).token_decl_for_prec)->tag : "<NULL>"); }
#line 1268 "src/parse-gram.c"
        break;

    case 87: /* symbol_decls  */
#line 237 "src/parse-gram.y"
         { symbol_list_syms_print (((*yyvaluep).symbol_decls), yyo); }
#line 1274 "src/parse-gram.c"
        break;

    case 88: /* symbol_decl.1  */
#line 237 "src/parse-gram.y"
         { symbol_list_syms_print (((*yyvaluep).yykind_88), yyo); }
#line 1280 "src/parse-gram.c"
        break;

    case 96: /* variable  */
#line 221 "src/parse-gram.y"
         { fputs (((*yyvaluep).variable), yyo); }
#line 1286 "src/parse-gram.c"
        break;

    case 97: /* value  */
#line 720 "src/parse-gram.y"
{
  switch (((*yyvaluep).value).kind)
    {
    case muscle_code:    fprintf (yyo,  "{%s}",  ((*yyvaluep).value).chars); break;
    case muscle_keyword: fprintf (yyo,   "%s",   ((*yyvaluep).value).chars); break;
    case muscle_string:  fprintf (yyo, "\"%s\"", ((*yyvaluep).value).chars); break;
    }
}
#line 1299 "src/parse-gram.c"
        break;

    case 98: /* id  */
#line 231 "src/parse-gram.y"
         { fprintf (yyo, "%s", ((*yyvaluep).id) ? ((*yyvaluep).id)->tag : "<NULL>"); }
#line 1305 "src/parse-gram.c"
        break;

    case 99: /* id_colon  */
#line 232 "src/parse-gram.y"
         { fprintf (yyo, "%s:", ((*yyvaluep).id_colon)->tag); }
#line 1311 "src/parse-gram.c"
        break;

    case 100: /* symbol  */
#line 231 "src/parse-gram.y"
         { fprintf (yyo, "%s", ((*yyvaluep).symbol) ? ((*yyvaluep).symbol)->tag : "<NULL>"); }
#line 1317 "src/parse-gram.c"
        break;

    case 101: /* string_as_id  */
#line 231 "src/parse-gram.y"
         { fprintf (yyo, "%s", ((*yyvaluep).string_as_id) ? ((*yyvaluep).string_as_id)->tag : "<NULL>"); }
#line 1323 "src/parse-gram.c"
        break;

      default:
        break;
    }
  YY_IGNORE_MAYBE_UNINITIALIZED_END
/* "%code post-printer" blocks.  */
#line 208 "src/parse-gram.y"
troff (yyo);

#line 1334 "src/parse-gram.c"
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  YY_LOCATION_PRINT (yyo, *yylocationp);
  YYFPRINTF (yyo, ": ");
  yy_symbol_value_print (yyo, yykind, yyvaluep, yylocationp);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp, YYLTYPE *yylsp,
                 int yyrule)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)],
                       &(yylsp[(yyi + 1) - (yynrhs)]));
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, yylsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !GRAM_DEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !GRAM_DEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


/* Given a state stack such that *YYBOTTOM is its bottom, such that
   *YYTOP is either its top or is YYTOP_EMPTY to indicate an empty
   stack, and such that *YYCAPACITY is the maximum number of elements it
   can hold without a reallocation, make sure there is enough room to
   store YYADD more elements.  If not, allocate a new stack using
   YYSTACK_ALLOC, copy the existing elements, and adjust *YYBOTTOM,
   *YYTOP, and *YYCAPACITY to reflect the new capacity and memory
   location.  If *YYBOTTOM != YYBOTTOM_NO_FREE, then free the old stack
   using YYSTACK_FREE.  Return 0 if successful or if no reallocation is
   required.  Return YYENOMEM if memory is exhausted.  */
static int
yy_lac_stack_realloc (YYPTRDIFF_T *yycapacity, YYPTRDIFF_T yyadd,
#if GRAM_DEBUG
                      char const *yydebug_prefix,
                      char const *yydebug_suffix,
#endif
                      yy_state_t **yybottom,
                      yy_state_t *yybottom_no_free,
                      yy_state_t **yytop, yy_state_t *yytop_empty)
{
  YYPTRDIFF_T yysize_old =
    *yytop == yytop_empty ? 0 : *yytop - *yybottom + 1;
  YYPTRDIFF_T yysize_new = yysize_old + yyadd;
  if (*yycapacity < yysize_new)
    {
      YYPTRDIFF_T yyalloc = 2 * yysize_new;
      yy_state_t *yybottom_new;
      /* Use YYMAXDEPTH for maximum stack size given that the stack
         should never need to grow larger than the main state stack
         needs to grow without LAC.  */
      if (YYMAXDEPTH < yysize_new)
        {
          YYDPRINTF ((stderr, "%smax size exceeded%s", yydebug_prefix,
                      yydebug_suffix));
          return YYENOMEM;
        }
      if (YYMAXDEPTH < yyalloc)
        yyalloc = YYMAXDEPTH;
      yybottom_new =
        YY_CAST (yy_state_t *,
                 YYSTACK_ALLOC (YY_CAST (YYSIZE_T,
                                         yyalloc * YYSIZEOF (*yybottom_new))));
      if (!yybottom_new)
        {
          YYDPRINTF ((stderr, "%srealloc failed%s", yydebug_prefix,
                      yydebug_suffix));
          return YYENOMEM;
        }
      if (*yytop != yytop_empty)
        {
          YYCOPY (yybottom_new, *yybottom, yysize_old);
          *yytop = yybottom_new + (yysize_old - 1);
        }
      if (*yybottom != yybottom_no_free)
        YYSTACK_FREE (*yybottom);
      *yybottom = yybottom_new;
      *yycapacity = yyalloc;
    }
  return 0;
}

/* Establish the initial context for the current lookahead if no initial
   context is currently established.

   We define a context as a snapshot of the parser stacks.  We define
   the initial context for a lookahead as the context in which the
   parser initially examines that lookahead in order to select a
   syntactic action.  Thus, if the lookahead eventually proves
   syntactically unacceptable (possibly in a later context reached via a
   series of reductions), the initial context can be used to determine
   the exact set of tokens that would be syntactically acceptable in the
   lookahead's place.  Moreover, it is the context after which any
   further semantic actions would be erroneous because they would be
   determined by a syntactically unacceptable token.

   YY_LAC_ESTABLISH should be invoked when a reduction is about to be
   performed in an inconsistent state (which, for the purposes of LAC,
   includes consistent states that don't know they're consistent because
   their default reductions have been disabled).  Iff there is a
   lookahead token, it should also be invoked before reporting a syntax
   error.  This latter case is for the sake of the debugging output.

   For parse.lac=full, the implementation of YY_LAC_ESTABLISH is as
   follows.  If no initial context is currently established for the
   current lookahead, then check if that lookahead can eventually be
   shifted if syntactic actions continue from the current context.
   Report a syntax error if it cannot.  */
#define YY_LAC_ESTABLISH                                                \
do {                                                                    \
  if (!yy_lac_established)                                              \
    {                                                                   \
      YYDPRINTF ((stderr,                                               \
                  "LAC: initial context established for %s\n",          \
                  yysymbol_name (yytoken)));                            \
      yy_lac_established = 1;                                           \
      switch (yy_lac (yyesa, &yyes, &yyes_capacity, yyssp, yytoken))    \
        {                                                               \
        case YYENOMEM:                                                  \
          goto yyexhaustedlab;                                          \
        case 1:                                                         \
          goto yyerrlab;                                                \
        }                                                               \
    }                                                                   \
} while (0)

/* Discard any previous initial lookahead context because of Event,
   which may be a lookahead change or an invalidation of the currently
   established initial context for the current lookahead.

   The most common example of a lookahead change is a shift.  An example
   of both cases is syntax error recovery.  That is, a syntax error
   occurs when the lookahead is syntactically erroneous for the
   currently established initial context, so error recovery manipulates
   the parser stacks to try to find a new initial context in which the
   current lookahead is syntactically acceptable.  If it fails to find
   such a context, it discards the lookahead.  */
#if GRAM_DEBUG
# define YY_LAC_DISCARD(Event)                                           \
do {                                                                     \
  if (yy_lac_established)                                                \
    {                                                                    \
      YYDPRINTF ((stderr, "LAC: initial context discarded due to "       \
                  Event "\n"));                                          \
      yy_lac_established = 0;                                            \
    }                                                                    \
} while (0)
#else
# define YY_LAC_DISCARD(Event) yy_lac_established = 0
#endif

/* Given the stack whose top is *YYSSP, return 0 iff YYTOKEN can
   eventually (after perhaps some reductions) be shifted, return 1 if
   not, or return YYENOMEM if memory is exhausted.  As preconditions and
   postconditions: *YYES_CAPACITY is the allocated size of the array to
   which *YYES points, and either *YYES = YYESA or *YYES points to an
   array allocated with YYSTACK_ALLOC.  yy_lac may overwrite the
   contents of either array, alter *YYES and *YYES_CAPACITY, and free
   any old *YYES other than YYESA.  */
static int
yy_lac (yy_state_t *yyesa, yy_state_t **yyes,
        YYPTRDIFF_T *yyes_capacity, yy_state_t *yyssp, yysymbol_kind_t yytoken)
{
  yy_state_t *yyes_prev = yyssp;
  yy_state_t *yyesp = yyes_prev;
  /* Reduce until we encounter a shift and thereby accept the token.  */
  YYDPRINTF ((stderr, "LAC: checking lookahead %s:", yysymbol_name (yytoken)));
  if (yytoken == YYSYMBOL_YYUNDEF)
    {
      YYDPRINTF ((stderr, " Always Err\n"));
      return 1;
    }
  while (1)
    {
      int yyrule = yypact[+*yyesp];
      if (yypact_value_is_default (yyrule)
          || (yyrule += yytoken) < 0 || YYLAST < yyrule
          || yycheck[yyrule] != yytoken)
        {
          /* Use the default action.  */
          yyrule = yydefact[+*yyesp];
          if (yyrule == 0)
            {
              YYDPRINTF ((stderr, " Err\n"));
              return 1;
            }
        }
      else
        {
          /* Use the action from yytable.  */
          yyrule = yytable[yyrule];
          if (yytable_value_is_error (yyrule))
            {
              YYDPRINTF ((stderr, " Err\n"));
              return 1;
            }
          if (0 < yyrule)
            {
              YYDPRINTF ((stderr, " S%d\n", yyrule));
              return 0;
            }
          yyrule = -yyrule;
        }
      /* By now we know we have to simulate a reduce.  */
      YYDPRINTF ((stderr, " R%d", yyrule - 1));
      {
        /* Pop the corresponding number of values from the stack.  */
        YYPTRDIFF_T yylen = yyr2[yyrule];
        /* First pop from the LAC stack as many tokens as possible.  */
        if (yyesp != yyes_prev)
          {
            YYPTRDIFF_T yysize = yyesp - *yyes + 1;
            if (yylen < yysize)
              {
                yyesp -= yylen;
                yylen = 0;
              }
            else
              {
                yyesp = yyes_prev;
                yylen -= yysize;
              }
          }
        /* Only afterwards look at the main stack.  */
        if (yylen)
          yyesp = yyes_prev -= yylen;
      }
      /* Push the resulting state of the reduction.  */
      {
        yy_state_fast_t yystate;
        {
          const int yylhs = yyr1[yyrule] - YYNTOKENS;
          const int yyi = yypgoto[yylhs] + *yyesp;
          yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyesp
                     ? yytable[yyi]
                     : yydefgoto[yylhs]);
        }
        if (yyesp == yyes_prev)
          {
            yyesp = *yyes;
            YY_IGNORE_USELESS_CAST_BEGIN
            *yyesp = YY_CAST (yy_state_t, yystate);
            YY_IGNORE_USELESS_CAST_END
          }
        else
          {
            if (yy_lac_stack_realloc (yyes_capacity, 1,
#if GRAM_DEBUG
                                      " (", ")",
#endif
                                      yyes, yyesa, &yyesp, yyes_prev))
              {
                YYDPRINTF ((stderr, "\n"));
                return YYENOMEM;
              }
            YY_IGNORE_USELESS_CAST_BEGIN
            *++yyesp = YY_CAST (yy_state_t, yystate);
            YY_IGNORE_USELESS_CAST_END
          }
        YYDPRINTF ((stderr, " G%d", yystate));
      }
    }
}

/* Context of a parse error.  */
typedef struct
{
  yy_state_t *yyssp;
  yy_state_t *yyesa;
  yy_state_t **yyes;
  YYPTRDIFF_T *yyes_capacity;
  yysymbol_kind_t yytoken;
  YYLTYPE *yylloc;
} yypcontext_t;

/* Put in YYARG at most YYARGN of the expected tokens given the
   current YYCTX, and return the number of tokens stored in YYARG.  If
   YYARG is null, return the number of expected tokens (guaranteed to
   be less than YYNTOKENS).  Return YYENOMEM on memory exhaustion.
   Return 0 if there are more than YYARGN expected tokens, yet fill
   YYARG up to YYARGN. */
static int
yypcontext_expected_tokens (const yypcontext_t *yyctx,
                            yysymbol_kind_t yyarg[], int yyargn)
{
  /* Actual size of YYARG. */
  int yycount = 0;

  int yyx;
  for (yyx = 0; yyx < YYNTOKENS; ++yyx)
    {
      yysymbol_kind_t yysym = YY_CAST (yysymbol_kind_t, yyx);
      if (yysym != YYSYMBOL_YYerror && yysym != YYSYMBOL_YYUNDEF)
        switch (yy_lac (yyctx->yyesa, yyctx->yyes, yyctx->yyes_capacity, yyctx->yyssp, yysym))
          {
          case YYENOMEM:
            return YYENOMEM;
          case 1:
            continue;
          default:
            if (!yyarg)
              ++yycount;
            else if (yycount == yyargn)
              return 0;
            else
              yyarg[yycount++] = yysym;
          }
    }
  if (yyarg && yycount == 0 && 0 < yyargn)
    yyarg[0] = YYSYMBOL_YYEMPTY;
  return yycount;
}




/* The kind of the lookahead of this context.  */
static yysymbol_kind_t
yypcontext_token (const yypcontext_t *yyctx) YY_ATTRIBUTE_UNUSED;

static yysymbol_kind_t
yypcontext_token (const yypcontext_t *yyctx)
{
  return yyctx->yytoken;
}

/* The location of the lookahead of this context.  */
static YYLTYPE *
yypcontext_location (const yypcontext_t *yyctx) YY_ATTRIBUTE_UNUSED;

static YYLTYPE *
yypcontext_location (const yypcontext_t *yyctx)
{
  return yyctx->yylloc;
}

/* User defined function to report a syntax error.  */
static int
yyreport_syntax_error (const yypcontext_t *yyctx);

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep, YYLTYPE *yylocationp)
{
  YYUSE (yyvaluep);
  YYUSE (yylocationp);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  switch (yykind)
    {
    case 75: /* generic_symlist  */
#line 236 "src/parse-gram.y"
            { symbol_list_free (((*yyvaluep).generic_symlist)); }
#line 1777 "src/parse-gram.c"
        break;

    case 76: /* generic_symlist_item  */
#line 236 "src/parse-gram.y"
            { symbol_list_free (((*yyvaluep).generic_symlist_item)); }
#line 1783 "src/parse-gram.c"
        break;

    case 78: /* nterm_decls  */
#line 236 "src/parse-gram.y"
            { symbol_list_free (((*yyvaluep).nterm_decls)); }
#line 1789 "src/parse-gram.c"
        break;

    case 79: /* token_decls  */
#line 236 "src/parse-gram.y"
            { symbol_list_free (((*yyvaluep).token_decls)); }
#line 1795 "src/parse-gram.c"
        break;

    case 80: /* token_decl.1  */
#line 236 "src/parse-gram.y"
            { symbol_list_free (((*yyvaluep).yykind_80)); }
#line 1801 "src/parse-gram.c"
        break;

    case 84: /* token_decls_for_prec  */
#line 236 "src/parse-gram.y"
            { symbol_list_free (((*yyvaluep).token_decls_for_prec)); }
#line 1807 "src/parse-gram.c"
        break;

    case 85: /* token_decl_for_prec.1  */
#line 236 "src/parse-gram.y"
            { symbol_list_free (((*yyvaluep).yykind_85)); }
#line 1813 "src/parse-gram.c"
        break;

    case 87: /* symbol_decls  */
#line 236 "src/parse-gram.y"
            { symbol_list_free (((*yyvaluep).symbol_decls)); }
#line 1819 "src/parse-gram.c"
        break;

    case 88: /* symbol_decl.1  */
#line 236 "src/parse-gram.y"
            { symbol_list_free (((*yyvaluep).yykind_88)); }
#line 1825 "src/parse-gram.c"
        break;

      default:
        break;
    }
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}






/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
/* The lookahead symbol.  */
int yychar;


/* The semantic value of the lookahead symbol.  */
/* Default value used for initialization, for pacifying older GCCs
   or non-GCC compilers.  */
YY_INITIAL_VALUE (static YYSTYPE yyval_default;)
YYSTYPE yylval YY_INITIAL_VALUE (= yyval_default);

/* Location data for the lookahead symbol.  */
static YYLTYPE yyloc_default
# if defined GRAM_LTYPE_IS_TRIVIAL && GRAM_LTYPE_IS_TRIVIAL
  = { 1, 1, 1, 1 }
# endif
;
YYLTYPE yylloc = yyloc_default;

    /* Number of syntax errors so far.  */
    int yynerrs;

    yy_state_fast_t yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.
       'yyls': related to locations.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize;

    /* The state stack.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss;
    yy_state_t *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    /* The location stack.  */
    YYLTYPE yylsa[YYINITDEPTH];
    YYLTYPE *yyls;
    YYLTYPE *yylsp;

    yy_state_t yyesa[20];
    yy_state_t *yyes;
    YYPTRDIFF_T yyes_capacity;

  /* Whether LAC context is established.  A Boolean.  */
  int yy_lac_established = 0;
  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;
  YYLTYPE yyloc;

  /* The locations where the error started and ended.  */
  YYLTYPE yyerror_range[3];



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N), yylsp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yynerrs = 0;
  yystate = 0;
  yyerrstatus = 0;

  yystacksize = YYINITDEPTH;
  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yylsp = yyls = yylsa;

  yyes = yyesa;
  yyes_capacity = 20;
  if (YYMAXDEPTH < yyes_capacity)
    yyes_capacity = YYMAXDEPTH;


  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = GRAM_EMPTY; /* Cause a token to be read.  */

/* User initialization code.  */
#line 136 "src/parse-gram.y"
{
  /* Bison's grammar can initial empty locations, hence a default
     location is needed. */
  boundary_set (&yylloc.start, grammar_file, 1, 1, 1);
  boundary_set (&yylloc.end, grammar_file, 1, 1, 1);
}

#line 1952 "src/parse-gram.c"

  yylsp[0] = yylloc;
  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    goto yyexhaustedlab;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;
        YYLTYPE *yyls1 = yyls;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yyls1, yysize * YYSIZEOF (*yylsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
        yyls = yyls1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
        YYSTACK_RELOCATE (yyls_alloc, yyls);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;
      yylsp = yyls + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == GRAM_EMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex (&yylval, &yylloc);
    }

  if (yychar <= GRAM_EOF)
    {
      yychar = GRAM_EOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == GRAM_error)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = GRAM_UNDEF;
      yytoken = YYSYMBOL_YYerror;
      yyerror_range[1] = yylloc;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    {
      YY_LAC_ESTABLISH;
      goto yydefault;
    }
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      YY_LAC_ESTABLISH;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END
  *++yylsp = yylloc;

  /* Discard the shifted token.  */
  yychar = GRAM_EMPTY;
  YY_LAC_DISCARD ("shift");
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];

  /* Default location. */
  YYLLOC_DEFAULT (yyloc, (yylsp - yylen), yylen);
  yyerror_range[1] = yyloc;
  YY_REDUCE_PRINT (yyn);
  {
    int yychar_backup = yychar;
    switch (yyn)
      {
  case 6:
#line 303 "src/parse-gram.y"
    {
      muscle_code_grow (union_seen ? "post_prologue" : "pre_prologue",
                        translate_code ((yyvsp[0].PROLOGUE), (yylsp[0]), true), (yylsp[0]));
      code_scanner_last_string_free ();
    }
#line 2175 "src/parse-gram.c"
    break;

  case 7:
#line 309 "src/parse-gram.y"
    {
      muscle_percent_define_ensure ((yyvsp[0].PERCENT_FLAG), (yylsp[0]), true);
    }
#line 2183 "src/parse-gram.c"
    break;

  case 8:
#line 313 "src/parse-gram.y"
    {
      muscle_percent_define_insert ((yyvsp[-1].variable), (yyloc), (yyvsp[0].value).kind, (yyvsp[0].value).chars,
                                    MUSCLE_PERCENT_DEFINE_GRAMMAR_FILE);
    }
#line 2192 "src/parse-gram.c"
    break;

  case 9:
#line 317 "src/parse-gram.y"
                                   { defines_flag = true; }
#line 2198 "src/parse-gram.c"
    break;

  case 10:
#line 319 "src/parse-gram.y"
    {
      defines_flag = true;
      spec_header_file = xstrdup ((yyvsp[0].STRING));
    }
#line 2207 "src/parse-gram.c"
    break;

  case 11:
#line 323 "src/parse-gram.y"
                                   { handle_error_verbose (&(yyloc), (yyvsp[0].PERCENT_ERROR_VERBOSE)); }
#line 2213 "src/parse-gram.c"
    break;

  case 12:
#line 324 "src/parse-gram.y"
                                   { expected_sr_conflicts = (yyvsp[0].INT); }
#line 2219 "src/parse-gram.c"
    break;

  case 13:
#line 325 "src/parse-gram.y"
                                   { expected_rr_conflicts = (yyvsp[0].INT); }
#line 2225 "src/parse-gram.c"
    break;

  case 14:
#line 326 "src/parse-gram.y"
                                   { handle_file_prefix (&(yyloc), &(yylsp[-1]), (yyvsp[-1].PERCENT_FILE_PREFIX), (yyvsp[0].STRING)); }
#line 2231 "src/parse-gram.c"
    break;

  case 15:
#line 328 "src/parse-gram.y"
    {
      nondeterministic_parser = true;
      glr_parser = true;
    }
#line 2240 "src/parse-gram.c"
    break;

  case 16:
#line 333 "src/parse-gram.y"
    {
      muscle_code_grow ("initial_action", translate_code ((yyvsp[0].BRACED_CODE), (yylsp[0]), false), (yylsp[0]));
      code_scanner_last_string_free ();
    }
#line 2249 "src/parse-gram.c"
    break;

  case 17:
#line 337 "src/parse-gram.y"
                                { language_argmatch ((yyvsp[0].STRING), grammar_prio, (yylsp[-1])); }
#line 2255 "src/parse-gram.c"
    break;

  case 18:
#line 338 "src/parse-gram.y"
                                { handle_name_prefix (&(yyloc), (yyvsp[-1].PERCENT_NAME_PREFIX), (yyvsp[0].STRING)); }
#line 2261 "src/parse-gram.c"
    break;

  case 19:
#line 339 "src/parse-gram.y"
                                { no_lines_flag = true; }
#line 2267 "src/parse-gram.c"
    break;

  case 20:
#line 340 "src/parse-gram.y"
                                { nondeterministic_parser = true; }
#line 2273 "src/parse-gram.c"
    break;

  case 21:
#line 341 "src/parse-gram.y"
                                { spec_outfile = (yyvsp[0].STRING); }
#line 2279 "src/parse-gram.c"
    break;

  case 22:
#line 342 "src/parse-gram.y"
           { current_param = (yyvsp[0].PERCENT_PARAM); }
#line 2285 "src/parse-gram.c"
    break;

  case 23:
#line 342 "src/parse-gram.y"
                                          { current_param = param_none; }
#line 2291 "src/parse-gram.c"
    break;

  case 24:
#line 343 "src/parse-gram.y"
                                { handle_pure_parser (&(yyloc), (yyvsp[0].PERCENT_PURE_PARSER)); }
#line 2297 "src/parse-gram.c"
    break;

  case 25:
#line 344 "src/parse-gram.y"
                                { handle_require (&(yylsp[0]), (yyvsp[0].STRING)); }
#line 2303 "src/parse-gram.c"
    break;

  case 26:
#line 345 "src/parse-gram.y"
                                { handle_skeleton (&(yylsp[0]), (yyvsp[0].STRING)); }
#line 2309 "src/parse-gram.c"
    break;

  case 27:
#line 346 "src/parse-gram.y"
                                { token_table_flag = true; }
#line 2315 "src/parse-gram.c"
    break;

  case 28:
#line 347 "src/parse-gram.y"
                                { report_flag |= report_states; }
#line 2321 "src/parse-gram.c"
    break;

  case 29:
#line 348 "src/parse-gram.y"
                                { handle_yacc (&(yyloc)); }
#line 2327 "src/parse-gram.c"
    break;

  case 30:
#line 349 "src/parse-gram.y"
                                { current_class = unknown_sym; yyerrok; }
#line 2333 "src/parse-gram.c"
    break;

  case 32:
#line 354 "src/parse-gram.y"
                   { add_param (current_param, (yyvsp[0].BRACED_CODE), (yylsp[0])); }
#line 2339 "src/parse-gram.c"
    break;

  case 33:
#line 355 "src/parse-gram.y"
                   { add_param (current_param, (yyvsp[0].BRACED_CODE), (yylsp[0])); }
#line 2345 "src/parse-gram.c"
    break;

  case 35:
#line 366 "src/parse-gram.y"
    {
      grammar_start_symbol_set ((yyvsp[0].symbol), (yylsp[0]));
    }
#line 2353 "src/parse-gram.c"
    break;

  case 36:
#line 370 "src/parse-gram.y"
    {
      code_props code;
      code_props_symbol_action_init (&code, (yyvsp[-1].BRACED_CODE), (yylsp[-1]));
      code_props_translate_code (&code);
      {
        for (symbol_list *list = (yyvsp[0].generic_symlist); list; list = list->next)
          symbol_list_code_props_set (list, (yyvsp[-2].code_props_type), &code);
        symbol_list_free ((yyvsp[0].generic_symlist));
      }
    }
#line 2368 "src/parse-gram.c"
    break;

  case 37:
#line 381 "src/parse-gram.y"
    {
      default_prec = true;
    }
#line 2376 "src/parse-gram.c"
    break;

  case 38:
#line 385 "src/parse-gram.y"
    {
      default_prec = false;
    }
#line 2384 "src/parse-gram.c"
    break;

  case 39:
#line 389 "src/parse-gram.y"
    {
      /* Do not invoke muscle_percent_code_grow here since it invokes
         muscle_user_name_list_grow.  */
      muscle_code_grow ("percent_code()",
                        translate_code_braceless ((yyvsp[0].BRACED_CODE), (yylsp[0])), (yylsp[0]));
      code_scanner_last_string_free ();
    }
#line 2396 "src/parse-gram.c"
    break;

  case 40:
#line 397 "src/parse-gram.y"
    {
      muscle_percent_code_grow ((yyvsp[-1].ID), (yylsp[-1]), translate_code_braceless ((yyvsp[0].BRACED_CODE), (yylsp[0])), (yylsp[0]));
      code_scanner_last_string_free ();
    }
#line 2405 "src/parse-gram.c"
    break;

  case 41:
#line 406 "src/parse-gram.y"
                 { (yyval.code_props_type) = destructor; }
#line 2411 "src/parse-gram.c"
    break;

  case 42:
#line 407 "src/parse-gram.y"
                 { (yyval.code_props_type) = printer; }
#line 2417 "src/parse-gram.c"
    break;

  case 43:
#line 417 "src/parse-gram.y"
         {}
#line 2423 "src/parse-gram.c"
    break;

  case 44:
#line 418 "src/parse-gram.y"
         { muscle_percent_define_insert ("api.value.union.name",
                                         (yylsp[0]), muscle_keyword, (yyvsp[0].ID),
                                         MUSCLE_PERCENT_DEFINE_GRAMMAR_FILE); }
#line 2431 "src/parse-gram.c"
    break;

  case 45:
#line 425 "src/parse-gram.y"
    {
      union_seen = true;
      muscle_code_grow ("union_members", translate_code_braceless ((yyvsp[0].BRACED_CODE), (yylsp[0])), (yylsp[0]));
      code_scanner_last_string_free ();
    }
#line 2441 "src/parse-gram.c"
    break;

  case 46:
#line 437 "src/parse-gram.y"
           { current_class = nterm_sym; }
#line 2447 "src/parse-gram.c"
    break;

  case 47:
#line 438 "src/parse-gram.y"
    {
      current_class = unknown_sym;
      symbol_list_free ((yyvsp[0].nterm_decls));
    }
#line 2456 "src/parse-gram.c"
    break;

  case 48:
#line 442 "src/parse-gram.y"
           { current_class = token_sym; }
#line 2462 "src/parse-gram.c"
    break;

  case 49:
#line 443 "src/parse-gram.y"
    {
      current_class = unknown_sym;
      symbol_list_free ((yyvsp[0].token_decls));
    }
#line 2471 "src/parse-gram.c"
    break;

  case 50:
#line 448 "src/parse-gram.y"
    {
      symbol_list_free ((yyvsp[0].symbol_decls));
    }
#line 2479 "src/parse-gram.c"
    break;

  case 51:
#line 452 "src/parse-gram.y"
    {
      ++current_prec;
      for (symbol_list *list = (yyvsp[0].token_decls_for_prec); list; list = list->next)
        symbol_precedence_set (list->content.sym, current_prec, (yyvsp[-1].precedence_declarator), (yylsp[-1]));
      symbol_list_free ((yyvsp[0].token_decls_for_prec));
    }
#line 2490 "src/parse-gram.c"
    break;

  case 52:
#line 461 "src/parse-gram.y"
                { (yyval.precedence_declarator) = left_assoc; }
#line 2496 "src/parse-gram.c"
    break;

  case 53:
#line 462 "src/parse-gram.y"
                { (yyval.precedence_declarator) = right_assoc; }
#line 2502 "src/parse-gram.c"
    break;

  case 54:
#line 463 "src/parse-gram.y"
                { (yyval.precedence_declarator) = non_assoc; }
#line 2508 "src/parse-gram.c"
    break;

  case 55:
#line 464 "src/parse-gram.y"
                { (yyval.precedence_declarator) = precedence_assoc; }
#line 2514 "src/parse-gram.c"
    break;

  case 56:
#line 468 "src/parse-gram.y"
         { (yyval.yykind_74) = NULL; }
#line 2520 "src/parse-gram.c"
    break;

  case 57:
#line 469 "src/parse-gram.y"
         { (yyval.yykind_74) = (yyvsp[0].TAG); }
#line 2526 "src/parse-gram.c"
    break;

  case 59:
#line 475 "src/parse-gram.y"
                                         { (yyval.generic_symlist) = symbol_list_append ((yyvsp[-1].generic_symlist), (yyvsp[0].generic_symlist_item)); }
#line 2532 "src/parse-gram.c"
    break;

  case 60:
#line 479 "src/parse-gram.y"
            { (yyval.generic_symlist_item) = symbol_list_sym_new ((yyvsp[0].symbol), (yylsp[0])); }
#line 2538 "src/parse-gram.c"
    break;

  case 61:
#line 480 "src/parse-gram.y"
            { (yyval.generic_symlist_item) = symbol_list_type_new ((yyvsp[0].tag), (yylsp[0])); }
#line 2544 "src/parse-gram.c"
    break;

  case 63:
#line 485 "src/parse-gram.y"
        { (yyval.tag) = uniqstr_new ("*"); }
#line 2550 "src/parse-gram.c"
    break;

  case 64:
#line 486 "src/parse-gram.y"
        { (yyval.tag) = uniqstr_new (""); }
#line 2556 "src/parse-gram.c"
    break;

  case 66:
#line 509 "src/parse-gram.y"
    {
      (yyval.token_decls) = (yyvsp[0].yykind_80);
    }
#line 2564 "src/parse-gram.c"
    break;

  case 67:
#line 513 "src/parse-gram.y"
    {
      (yyval.token_decls) = symbol_list_type_set ((yyvsp[0].yykind_80), (yyvsp[-1].TAG), (yylsp[-1]));
    }
#line 2572 "src/parse-gram.c"
    break;

  case 68:
#line 517 "src/parse-gram.y"
    {
      (yyval.token_decls) = symbol_list_append ((yyvsp[-2].token_decls), symbol_list_type_set ((yyvsp[0].yykind_80), (yyvsp[-1].TAG), (yylsp[-1])));
    }
#line 2580 "src/parse-gram.c"
    break;

  case 69:
#line 524 "src/parse-gram.y"
                            { (yyval.yykind_80) = symbol_list_sym_new ((yyvsp[0].token_decl), (yylsp[0])); }
#line 2586 "src/parse-gram.c"
    break;

  case 70:
#line 525 "src/parse-gram.y"
                            { (yyval.yykind_80) = symbol_list_append ((yyvsp[-1].yykind_80), symbol_list_sym_new ((yyvsp[0].token_decl), (yylsp[0]))); }
#line 2592 "src/parse-gram.c"
    break;

  case 71:
#line 530 "src/parse-gram.y"
    {
      (yyval.token_decl) = (yyvsp[-2].id);
      symbol_class_set ((yyvsp[-2].id), current_class, (yylsp[-2]), true);
      if (0 <= (yyvsp[-1].yykind_82))
        symbol_user_token_number_set ((yyvsp[-2].id), (yyvsp[-1].yykind_82), (yylsp[-1]));
      if ((yyvsp[0].alias))
        symbol_make_alias ((yyvsp[-2].id), (yyvsp[0].alias), (yylsp[0]));
    }
#line 2605 "src/parse-gram.c"
    break;

  case 72:
#line 542 "src/parse-gram.y"
          { (yyval.yykind_82) = -1; }
#line 2611 "src/parse-gram.c"
    break;

  case 74:
#line 548 "src/parse-gram.y"
                 { (yyval.alias) = NULL; }
#line 2617 "src/parse-gram.c"
    break;

  case 75:
#line 549 "src/parse-gram.y"
                 { (yyval.alias) = (yyvsp[0].string_as_id); }
#line 2623 "src/parse-gram.c"
    break;

  case 76:
#line 551 "src/parse-gram.y"
    {
      (yyval.alias) = symbol_get (quotearg_style (c_quoting_style, (yyvsp[0].TSTRING)), (yylsp[0]));
      symbol_class_set ((yyval.alias), token_sym, (yylsp[0]), false);
      (yyval.alias)->translatable = true;
    }
#line 2633 "src/parse-gram.c"
    break;

  case 77:
#line 569 "src/parse-gram.y"
    {
      (yyval.token_decls_for_prec) = (yyvsp[0].yykind_85);
    }
#line 2641 "src/parse-gram.c"
    break;

  case 78:
#line 573 "src/parse-gram.y"
    {
      (yyval.token_decls_for_prec) = symbol_list_type_set ((yyvsp[0].yykind_85), (yyvsp[-1].TAG), (yylsp[-1]));
    }
#line 2649 "src/parse-gram.c"
    break;

  case 79:
#line 577 "src/parse-gram.y"
    {
      (yyval.token_decls_for_prec) = symbol_list_append ((yyvsp[-2].token_decls_for_prec), symbol_list_type_set ((yyvsp[0].yykind_85), (yyvsp[-1].TAG), (yylsp[-1])));
    }
#line 2657 "src/parse-gram.c"
    break;

  case 80:
#line 585 "src/parse-gram.y"
    { (yyval.yykind_85) = symbol_list_sym_new ((yyvsp[0].token_decl_for_prec), (yylsp[0])); }
#line 2663 "src/parse-gram.c"
    break;

  case 81:
#line 587 "src/parse-gram.y"
    { (yyval.yykind_85) = symbol_list_append ((yyvsp[-1].yykind_85), symbol_list_sym_new ((yyvsp[0].token_decl_for_prec), (yylsp[0]))); }
#line 2669 "src/parse-gram.c"
    break;

  case 82:
#line 592 "src/parse-gram.y"
    {
      (yyval.token_decl_for_prec) = (yyvsp[-1].id);
      symbol_class_set ((yyvsp[-1].id), token_sym, (yylsp[-1]), false);
      if (0 <= (yyvsp[0].yykind_82))
        symbol_user_token_number_set ((yyvsp[-1].id), (yyvsp[0].yykind_82), (yylsp[0]));
    }
#line 2680 "src/parse-gram.c"
    break;

  case 84:
#line 609 "src/parse-gram.y"
    {
      (yyval.symbol_decls) = (yyvsp[0].yykind_88);
    }
#line 2688 "src/parse-gram.c"
    break;

  case 85:
#line 613 "src/parse-gram.y"
    {
      (yyval.symbol_decls) = symbol_list_type_set ((yyvsp[0].yykind_88), (yyvsp[-1].TAG), (yylsp[-1]));
    }
#line 2696 "src/parse-gram.c"
    break;

  case 86:
#line 617 "src/parse-gram.y"
    {
      (yyval.symbol_decls) = symbol_list_append ((yyvsp[-2].symbol_decls), symbol_list_type_set ((yyvsp[0].yykind_88), (yyvsp[-1].TAG), (yylsp[-1])));
    }
#line 2704 "src/parse-gram.c"
    break;

  case 87:
#line 625 "src/parse-gram.y"
    {
      symbol_class_set ((yyvsp[0].symbol), pct_type_sym, (yylsp[0]), false);
      (yyval.yykind_88) = symbol_list_sym_new ((yyvsp[0].symbol), (yylsp[0]));
    }
#line 2713 "src/parse-gram.c"
    break;

  case 88:
#line 630 "src/parse-gram.y"
    {
      symbol_class_set ((yyvsp[0].symbol), pct_type_sym, (yylsp[0]), false);
      (yyval.yykind_88) = symbol_list_append ((yyvsp[-1].yykind_88), symbol_list_sym_new ((yyvsp[0].symbol), (yylsp[0])));
    }
#line 2722 "src/parse-gram.c"
    break;

  case 93:
#line 651 "src/parse-gram.y"
    {
      yyerrok;
    }
#line 2730 "src/parse-gram.c"
    break;

  case 94:
#line 657 "src/parse-gram.y"
                         { current_lhs ((yyvsp[-1].id_colon), (yylsp[-1]), (yyvsp[0].yykind_95)); }
#line 2736 "src/parse-gram.c"
    break;

  case 95:
#line 658 "src/parse-gram.y"
    {
      /* Free the current lhs. */
      current_lhs (0, (yylsp[-4]), 0);
    }
#line 2745 "src/parse-gram.c"
    break;

  case 96:
#line 665 "src/parse-gram.y"
                     { grammar_current_rule_end ((yylsp[0])); }
#line 2751 "src/parse-gram.c"
    break;

  case 97:
#line 666 "src/parse-gram.y"
                     { grammar_current_rule_end ((yylsp[0])); }
#line 2757 "src/parse-gram.c"
    break;

  case 99:
#line 673 "src/parse-gram.y"
    { grammar_current_rule_begin (current_lhs_symbol, current_lhs_loc,
                                  current_lhs_named_ref); }
#line 2764 "src/parse-gram.c"
    break;

  case 100:
#line 676 "src/parse-gram.y"
    { grammar_current_rule_symbol_append ((yyvsp[-1].symbol), (yylsp[-1]), (yyvsp[0].yykind_95)); }
#line 2770 "src/parse-gram.c"
    break;

  case 101:
#line 678 "src/parse-gram.y"
    { grammar_current_rule_action_append ((yyvsp[-1].BRACED_CODE), (yylsp[-1]), (yyvsp[0].yykind_95), (yyvsp[-2].yykind_74)); }
#line 2776 "src/parse-gram.c"
    break;

  case 102:
#line 680 "src/parse-gram.y"
    { grammar_current_rule_predicate_append ((yyvsp[0].BRACED_PREDICATE), (yylsp[0])); }
#line 2782 "src/parse-gram.c"
    break;

  case 103:
#line 682 "src/parse-gram.y"
    { grammar_current_rule_empty_set ((yylsp[0])); }
#line 2788 "src/parse-gram.c"
    break;

  case 104:
#line 684 "src/parse-gram.y"
    { grammar_current_rule_prec_set ((yyvsp[0].symbol), (yylsp[0])); }
#line 2794 "src/parse-gram.c"
    break;

  case 105:
#line 686 "src/parse-gram.y"
    { grammar_current_rule_dprec_set ((yyvsp[0].INT), (yylsp[0])); }
#line 2800 "src/parse-gram.c"
    break;

  case 106:
#line 688 "src/parse-gram.y"
    { grammar_current_rule_merge_set ((yyvsp[0].TAG), (yylsp[0])); }
#line 2806 "src/parse-gram.c"
    break;

  case 107:
#line 690 "src/parse-gram.y"
    { grammar_current_rule_expect_sr ((yyvsp[0].INT), (yylsp[0])); }
#line 2812 "src/parse-gram.c"
    break;

  case 108:
#line 692 "src/parse-gram.y"
    { grammar_current_rule_expect_rr ((yyvsp[0].INT), (yylsp[0])); }
#line 2818 "src/parse-gram.c"
    break;

  case 109:
#line 696 "src/parse-gram.y"
                 { (yyval.yykind_95) = NULL; }
#line 2824 "src/parse-gram.c"
    break;

  case 110:
#line 697 "src/parse-gram.y"
                 { (yyval.yykind_95) = named_ref_new ((yyvsp[0].BRACKETED_ID), (yylsp[0])); }
#line 2830 "src/parse-gram.c"
    break;

  case 112:
#line 730 "src/parse-gram.y"
          { (yyval.value).kind = muscle_keyword; (yyval.value).chars = ""; }
#line 2836 "src/parse-gram.c"
    break;

  case 113:
#line 731 "src/parse-gram.y"
          { (yyval.value).kind = muscle_keyword; (yyval.value).chars = (yyvsp[0].ID); }
#line 2842 "src/parse-gram.c"
    break;

  case 114:
#line 732 "src/parse-gram.y"
          { (yyval.value).kind = muscle_string;  (yyval.value).chars = (yyvsp[0].STRING); }
#line 2848 "src/parse-gram.c"
    break;

  case 115:
#line 733 "src/parse-gram.y"
          { (yyval.value).kind = muscle_code;    (yyval.value).chars = strip_braces ((yyvsp[0].BRACED_CODE)); }
#line 2854 "src/parse-gram.c"
    break;

  case 116:
#line 746 "src/parse-gram.y"
    { (yyval.id) = symbol_from_uniqstr ((yyvsp[0].ID), (yylsp[0])); }
#line 2860 "src/parse-gram.c"
    break;

  case 117:
#line 748 "src/parse-gram.y"
    {
      const char *var = "api.token.raw";
      if (current_class == nterm_sym)
        {
          complain (&(yylsp[0]), complaint,
                    _("character literals cannot be nonterminals"));
          YYERROR;
        }
      if (muscle_percent_define_ifdef (var))
        {
          complain (&(yylsp[0]), complaint,
                    _("character literals cannot be used together"
                    " with %s"), var);
          location loc = muscle_percent_define_get_loc (var);
          subcomplain (&loc, complaint, _("definition of %s"), var);
        }
      (yyval.id) = symbol_get (char_name ((yyvsp[0].CHAR)), (yylsp[0]));
      symbol_class_set ((yyval.id), token_sym, (yylsp[0]), false);
      symbol_user_token_number_set ((yyval.id), (yyvsp[0].CHAR), (yylsp[0]));
    }
#line 2885 "src/parse-gram.c"
    break;

  case 118:
#line 771 "src/parse-gram.y"
           { (yyval.id_colon) = symbol_from_uniqstr ((yyvsp[0].ID_COLON), (yylsp[0])); }
#line 2891 "src/parse-gram.c"
    break;

  case 121:
#line 783 "src/parse-gram.y"
    {
      (yyval.string_as_id) = symbol_get (quotearg_style (c_quoting_style, (yyvsp[0].STRING)), (yylsp[0]));
      symbol_class_set ((yyval.string_as_id), token_sym, (yylsp[0]), false);
    }
#line 2900 "src/parse-gram.c"
    break;

  case 123:
#line 792 "src/parse-gram.y"
    {
      muscle_code_grow ("epilogue", translate_code ((yyvsp[0].EPILOGUE), (yylsp[0]), true), (yylsp[0]));
      code_scanner_last_string_free ();
    }
#line 2909 "src/parse-gram.c"
    break;


#line 2913 "src/parse-gram.c"

        default: break;
      }
    if (yychar_backup != yychar)
      YY_LAC_DISCARD ("yychar change");
  }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;
  *++yylsp = yyloc;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == GRAM_EMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      {
        yypcontext_t yyctx
          = {yyssp, yyesa, &yyes, &yyes_capacity, yytoken, &yylloc};
        if (yychar != GRAM_EMPTY)
          YY_LAC_ESTABLISH;
        if (yyreport_syntax_error (&yyctx) == 2)
          goto yyexhaustedlab;
      }
    }

  yyerror_range[1] = yylloc;
  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= GRAM_EOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == GRAM_EOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval, &yylloc);
          yychar = GRAM_EMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;

      yyerror_range[1] = *yylsp;
      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp, yylsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  /* If the stack popping above didn't lose the initial context for the
     current lookahead token, the shift below will for sure.  */
  YY_LAC_DISCARD ("error recovery");

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  yyerror_range[2] = yylloc;
  ++yylsp;
  YYLLOC_DEFAULT (*yylsp, yyerror_range, 2);

  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;


#if 1
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (&yylloc, YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif


/*-----------------------------------------------------.
| yyreturn -- parsing is finished, return the result.  |
`-----------------------------------------------------*/
yyreturn:
  if (yychar != GRAM_EMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval, &yylloc);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp, yylsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
  if (yyes != yyesa)
    YYSTACK_FREE (yyes);

  return yyresult;
}

#line 798 "src/parse-gram.y"


int
yyreport_syntax_error (const yypcontext_t *ctx)
{
  int res = 0;
  /* Arguments of format: reported tokens (one for the "unexpected",
     one per "expected"). */
  enum { ARGS_MAX = 5 };
  const char *argv[ARGS_MAX];
  int argc = 0;
  yysymbol_kind_t unexpected = yypcontext_token (ctx);
  if (unexpected != YYSYMBOL_YYEMPTY)
    {
      argv[argc++] = yysymbol_name (unexpected);
      yysymbol_kind_t expected[ARGS_MAX - 1];
      int nexpected = yypcontext_expected_tokens (ctx, expected, ARGS_MAX - 1);
      if (nexpected < 0)
        res = nexpected;
      else
        for (int i = 0; i < nexpected; ++i)
          argv[argc++] = yysymbol_name (expected[i]);
    }
  syntax_error (*yypcontext_location (ctx), argc, argv);
  return res;
}


/* Return the location of the left-hand side of a rule whose
   right-hand side is RHS[1] ... RHS[N].  Ignore empty nonterminals in
   the right-hand side, and return an empty location equal to the end
   boundary of RHS[0] if the right-hand side is empty.  */

static YYLTYPE
lloc_default (YYLTYPE const *rhs, int n)
{
  YYLTYPE loc;

  /* SGI MIPSpro 7.4.1m miscompiles "loc.start = loc.end = rhs[n].end;".
     The bug is fixed in 7.4.2m, but play it safe for now.  */
  loc.start = rhs[n].end;
  loc.end = rhs[n].end;

  /* Ignore empty nonterminals the start of the right-hand side.
     Do not bother to ignore them at the end of the right-hand side,
     since empty nonterminals have the same end as their predecessors.  */
  for (int i = 1; i <= n; i++)
    if (! equal_boundaries (rhs[i].start, rhs[i].end))
      {
        loc.start = rhs[i].start;
        break;
      }

  return loc;
}

static
char *strip_braces (char *code)
{
  code[strlen (code) - 1] = 0;
  return code + 1;
}

static
char const *
translate_code (char *code, location loc, bool plain)
{
  code_props plain_code;
  if (plain)
    code_props_plain_init (&plain_code, code, loc);
  else
    code_props_symbol_action_init (&plain_code, code, loc);
  code_props_translate_code (&plain_code);
  gram_scanner_last_string_free ();
  return plain_code.code;
}

static
char const *
translate_code_braceless (char *code, location loc)
{
  return translate_code (strip_braces (code), loc, true);
}

static void
add_param (param_type type, char *decl, location loc)
{
  static char const alphanum[26 + 26 + 1 + 10 + 1] =
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "_"
    "0123456789";

  char const *name_start = NULL;
  {
    char *p;
    /* Stop on last actual character.  */
    for (p = decl; p[1]; p++)
      if ((p == decl
           || ! memchr (alphanum, p[-1], sizeof alphanum - 1))
          && memchr (alphanum, p[0], sizeof alphanum - 10 - 1))
        name_start = p;

    /* Strip the surrounding '{' and '}', and any blanks just inside
       the braces.  */
    --p;
    while (c_isspace ((unsigned char) *p))
      --p;
    p[1] = '\0';
    ++decl;
    while (c_isspace ((unsigned char) *decl))
      ++decl;
  }

  if (! name_start)
    complain (&loc, complaint, _("missing identifier in parameter declaration"));
  else
    {
      char *name = xmemdup0 (name_start, strspn (name_start, alphanum));
      if (type & param_lex)
        muscle_pair_list_grow ("lex_param", decl, name);
      if (type & param_parse)
        muscle_pair_list_grow ("parse_param", decl, name);
      free (name);
    }

  gram_scanner_last_string_free ();
}


static void
handle_error_verbose (location const *loc, char const *directive)
{
  bison_directive (loc, directive);
  muscle_percent_define_insert (directive, *loc, muscle_keyword, "",
                                MUSCLE_PERCENT_DEFINE_GRAMMAR_FILE);
}


static void
handle_file_prefix (location const *loc,
                    location const *dir_loc,
                    char const *directive, char const *value)
{
  bison_directive (loc, directive);
  bool warned = false;

  if (location_empty (spec_file_prefix_loc))
    {
      spec_file_prefix_loc = *loc;
      spec_file_prefix = value;
    }
  else
    {
      duplicate_directive (directive, spec_file_prefix_loc, *loc);
      warned = true;
    }

  if (!warned
      && STRNEQ (directive, "%file-prefix"))
    deprecated_directive (dir_loc, directive, "%file-prefix");
}


static void
handle_name_prefix (location const *loc,
                    char const *directive, char const *value)
{
  bison_directive (loc, directive);

  char buf1[1024];
  size_t len1 = sizeof (buf1);
  char *old = asnprintf (buf1, &len1, "%s\"%s\"", directive, value);
  if (!old)
    xalloc_die ();

  if (location_empty (spec_name_prefix_loc))
    {
      spec_name_prefix = value;
      spec_name_prefix_loc = *loc;

      char buf2[1024];
      size_t len2 = sizeof (buf2);
      char *new = asnprintf (buf2, &len2, "%%define api.prefix {%s}", value);
      if (!new)
        xalloc_die ();
      deprecated_directive (loc, old, new);
      if (new != buf2)
        free (new);
    }
  else
    duplicate_directive (old, spec_file_prefix_loc, *loc);

  if (old != buf1)
    free (old);
}


static void
handle_pure_parser (location const *loc, char const *directive)
{
  bison_directive (loc, directive);
  deprecated_directive (loc, directive, "%define api.pure");
  muscle_percent_define_insert ("api.pure", *loc, muscle_keyword, "",
                                MUSCLE_PERCENT_DEFINE_GRAMMAR_FILE);
}


/* Convert VERSION into an int (MAJOR * 100 + MINOR).  Return -1 on
   errors.

   Changes of behavior are only on minor version changes, so "3.0.5"
   is the same as "3.0": 300. */
static int
str_to_version (char const *version)
{
  IGNORE_TYPE_LIMITS_BEGIN
  int res = 0;
  errno = 0;
  char *cp = NULL;
  long major = strtol (version, &cp, 10);
  if (errno || cp == version || *cp != '.' || major < 0
      || INT_MULTIPLY_WRAPV (major, 100, &res))
    return -1;

  ++cp;
  char *cp1 = NULL;
  long minor = strtol (cp, &cp1, 10);
  if (errno || cp1 == cp || (*cp1 != '\0' && *cp1 != '.')
      || ! (0 <= minor && minor < 100)
      || INT_ADD_WRAPV (minor, res, &res))
    return -1;

  IGNORE_TYPE_LIMITS_END
  return res;
}


static void
handle_require (location const *loc, char const *version)
{
  required_version = str_to_version (version);
  if (required_version == -1)
    {
      complain (loc, complaint, _("invalid version requirement: %s"),
                version);
      required_version = 0;
      return;
    }

  /* Pretend to be at least that version, to check features published
     in that version while developping it.  */
  const char* api_version = "3.6";
  const char* package_version =
    0 < strverscmp (api_version, PACKAGE_VERSION)
    ? api_version : PACKAGE_VERSION;
  if (0 < strverscmp (version, package_version))
    {
      complain (loc, complaint, _("require bison %s, but have %s"),
                version, package_version);
      exit (EX_MISMATCH);
    }
}

static void
handle_skeleton (location const *loc, char const *skel)
{
  char const *skeleton_user = skel;
  if (strchr (skeleton_user, '/'))
    {
      size_t dir_length = strlen (grammar_file);
      while (dir_length && grammar_file[dir_length - 1] != '/')
        --dir_length;
      while (dir_length && grammar_file[dir_length - 1] == '/')
        --dir_length;
      char *skeleton_build =
        xmalloc (dir_length + 1 + strlen (skeleton_user) + 1);
      if (dir_length > 0)
        {
          memcpy (skeleton_build, grammar_file, dir_length);
          skeleton_build[dir_length++] = '/';
        }
      strcpy (skeleton_build + dir_length, skeleton_user);
      skeleton_user = uniqstr_new (skeleton_build);
      free (skeleton_build);
    }
  skeleton_arg (skeleton_user, grammar_prio, *loc);
}


static void
handle_yacc (location const *loc)
{
  const char *directive = "%yacc";
  bison_directive (loc, directive);
  if (location_empty (yacc_loc))
    yacc_loc = *loc;
  else
    duplicate_directive (directive, yacc_loc, *loc);
}


static void
gram_error (location const *loc, char const *msg)
{
  complain (loc, complaint, "%s", msg);
}

static char const *
char_name (char c)
{
  if (c == '\'')
    return "'\\''";
  else
    {
      char buf[4];
      buf[0] = '\''; buf[1] = c; buf[2] = '\''; buf[3] = '\0';
      return quotearg_style (escape_quoting_style, buf);
    }
}

static
void
current_lhs (symbol *sym, location loc, named_ref *ref)
{
  current_lhs_symbol = sym;
  current_lhs_loc = loc;
  if (sym)
    symbol_location_as_lhs_set (sym, loc);
  /* In order to simplify memory management, named references for lhs
     are always assigned by deep copy into the current symbol_list
     node.  This is because a single named-ref in the grammar may
     result in several uses when the user factors lhs between several
     rules using "|".  Therefore free the parser's original copy.  */
  free (current_lhs_named_ref);
  current_lhs_named_ref = ref;
}

static void tron (FILE *yyo)
{
  begin_use_class ("value", yyo);
}

static void troff (FILE *yyo)
{
  end_use_class ("value", yyo);
}
