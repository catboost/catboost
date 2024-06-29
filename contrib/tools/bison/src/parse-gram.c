/* A Bison parser, made by GNU Bison 3.2.90.23-0bbcb-dirty.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2019 Free Software Foundation,
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

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Undocumented macros, especially those whose name start with YY_,
   are private implementation details.  Do not rely on them.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.2.90.23-0bbcb-dirty"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 2

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* "%code top" blocks.  */
#line 27 "src/parse-gram.y" /* yacc.c:315  */

  /* On column 0 to please syntax-check.  */
#include <config.h>

#line 72 "src/parse-gram.c" /* yacc.c:315  */
/* Substitute the type names.  */
#define YYSTYPE         GRAM_STYPE
#define YYLTYPE         GRAM_LTYPE
/* Substitute the variable and function names.  */
#define yyparse         gram_parse
#define yylex           gram_lex
#define yyerror         gram_error
#define yydebug         gram_debug
#define yynerrs         gram_nerrs



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

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 1
#endif

/* In a future release of Bison, this section will be replaced
   by #include "src/parse-gram.h".  */
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
#line 21 "src/parse-gram.y" /* yacc.c:352  */

  #include "symlist.h"
  #include "symtab.h"
#line 239 "src/parse-gram.y" /* yacc.c:352  */

  typedef enum
  {
    param_none   = 0,
    param_lex    = 1 << 0,
    param_parse  = 1 << 1,
    param_both   = param_lex | param_parse
  } param_type;
#line 682 "src/parse-gram.y" /* yacc.c:352  */

  #include "muscle-tab.h"
  typedef struct
  {
    char const *chars;
    muscle_kind kind;
  } value_type;

#line 147 "src/parse-gram.c" /* yacc.c:352  */

/* Token type.  */
#ifndef GRAM_TOKENTYPE
# define GRAM_TOKENTYPE
  enum gram_tokentype
  {
    GRAM_EOF = 0,
    STRING = 258,
    PERCENT_TOKEN = 259,
    PERCENT_NTERM = 260,
    PERCENT_TYPE = 261,
    PERCENT_DESTRUCTOR = 262,
    PERCENT_PRINTER = 263,
    PERCENT_LEFT = 264,
    PERCENT_RIGHT = 265,
    PERCENT_NONASSOC = 266,
    PERCENT_PRECEDENCE = 267,
    PERCENT_PREC = 268,
    PERCENT_DPREC = 269,
    PERCENT_MERGE = 270,
    PERCENT_CODE = 271,
    PERCENT_DEFAULT_PREC = 272,
    PERCENT_DEFINE = 273,
    PERCENT_DEFINES = 274,
    PERCENT_ERROR_VERBOSE = 275,
    PERCENT_EXPECT = 276,
    PERCENT_EXPECT_RR = 277,
    PERCENT_FLAG = 278,
    PERCENT_FILE_PREFIX = 279,
    PERCENT_GLR_PARSER = 280,
    PERCENT_INITIAL_ACTION = 281,
    PERCENT_LANGUAGE = 282,
    PERCENT_NAME_PREFIX = 283,
    PERCENT_NO_DEFAULT_PREC = 284,
    PERCENT_NO_LINES = 285,
    PERCENT_NONDETERMINISTIC_PARSER = 286,
    PERCENT_OUTPUT = 287,
    PERCENT_REQUIRE = 288,
    PERCENT_SKELETON = 289,
    PERCENT_START = 290,
    PERCENT_TOKEN_TABLE = 291,
    PERCENT_VERBOSE = 292,
    PERCENT_YACC = 293,
    BRACED_CODE = 294,
    BRACED_PREDICATE = 295,
    BRACKETED_ID = 296,
    CHAR = 297,
    EPILOGUE = 298,
    EQUAL = 299,
    ID = 300,
    ID_COLON = 301,
    PERCENT_PERCENT = 302,
    PIPE = 303,
    PROLOGUE = 304,
    SEMICOLON = 305,
    TAG = 306,
    TAG_ANY = 307,
    TAG_NONE = 308,
    INT = 309,
    PERCENT_PARAM = 310,
    PERCENT_UNION = 311,
    PERCENT_EMPTY = 312
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
  int yytype_79;
  /* named_ref.opt  */
  named_ref* yytype_91;
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
  symbol* yytype_98;
  /* generic_symlist  */
  symbol_list* generic_symlist;
  /* generic_symlist_item  */
  symbol_list* generic_symlist_item;
  /* nterm_decls  */
  symbol_list* nterm_decls;
  /* token_decls  */
  symbol_list* token_decls;
  /* token_decl.1  */
  symbol_list* yytype_77;
  /* token_decls_for_prec  */
  symbol_list* token_decls_for_prec;
  /* token_decl_for_prec.1  */
  symbol_list* yytype_81;
  /* symbol_decls  */
  symbol_list* symbol_decls;
  /* symbol_decl.1  */
  symbol_list* yytype_84;
  /* "%error-verbose"  */
  uniqstr PERCENT_ERROR_VERBOSE;
  /* "%<flag>"  */
  uniqstr PERCENT_FLAG;
  /* "%file-prefix"  */
  uniqstr PERCENT_FILE_PREFIX;
  /* "%name-prefix"  */
  uniqstr PERCENT_NAME_PREFIX;
  /* "%yacc"  */
  uniqstr PERCENT_YACC;
  /* "[identifier]"  */
  uniqstr BRACKETED_ID;
  /* "identifier"  */
  uniqstr ID;
  /* "identifier:"  */
  uniqstr ID_COLON;
  /* "<tag>"  */
  uniqstr TAG;
  /* tag.opt  */
  uniqstr yytype_71;
  /* tag  */
  uniqstr tag;
  /* variable  */
  uniqstr variable;
  /* "char"  */
  unsigned char CHAR;
  /* value  */
  value_type value;
#line 301 "src/parse-gram.c" /* yacc.c:352  */
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


/* Unqualified %code blocks.  */
#line 33 "src/parse-gram.y" /* yacc.c:355  */

  #include "system.h"
  #include <errno.h>

  #include "c-ctype.h"
  #include "complain.h"
  #include "conflicts.h"
  #include "files.h"
  #include "getargs.h"
  #include "gram.h"
  #include "named-ref.h"
  #include "quotearg.h"
  #include "reader.h"
  #include "scan-code.h"
  #include "scan-gram.h"
  #include "vasnprintf.h"
  #include "xmemdup0.h"

  static int current_prec = 0;
  static location current_lhs_location;
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

  /* Handle a %require directive.  */
  static void handle_require (location const *loc, char const *version);

  /* Handle a %skeleton directive.  */
  static void handle_skeleton (location const *loc, char const *skel);

  /* Handle a %yacc directive.  */
  static void handle_yacc (location const *loc, char const *directive);

  static void gram_error (location const *, char const *);

  /* A string that describes a char (e.g., 'a' -> "'a'").  */
  static char const *char_name (char);

  #define YYTYPE_INT16 int_fast16_t
  #define YYTYPE_INT8 int_fast8_t
  #define YYTYPE_UINT16 uint_fast16_t
  #define YYTYPE_UINT8 uint_fast8_t
#line 249 "src/parse-gram.y" /* yacc.c:355  */

  /** Add a lex-param and/or a parse-param.
   *
   * \param type  where to push this formal argument.
   * \param decl  the formal argument.  Destroyed.
   * \param loc   the location in the source.
   */
  static void add_param (param_type type, char *decl, location loc);
  static param_type current_param = param_none;

#line 425 "src/parse-gram.c" /* yacc.c:355  */

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

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

#ifndef YY_ATTRIBUTE
# if (defined __GNUC__                                               \
      && (2 < __GNUC__ || (__GNUC__ == 2 && 96 <= __GNUC_MINOR__)))  \
     || defined __SUNPRO_C && 0x5110 <= __SUNPRO_C
#  define YY_ATTRIBUTE(Spec) __attribute__(Spec)
# else
#  define YY_ATTRIBUTE(Spec) /* empty */
# endif
#endif

#ifndef YY_ATTRIBUTE_PURE
# define YY_ATTRIBUTE_PURE   YY_ATTRIBUTE ((__pure__))
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# define YY_ATTRIBUTE_UNUSED YY_ATTRIBUTE ((__unused__))
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && ! defined __ICC && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN \
    _Pragma ("GCC diagnostic push") \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")\
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END \
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
#endif


#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined GRAM_LTYPE_IS_TRIVIAL && GRAM_LTYPE_IS_TRIVIAL \
             && defined GRAM_STYPE_IS_TRIVIAL && GRAM_STYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
  YYLTYPE yyls_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE) + sizeof (YYLTYPE)) \
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
        YYSIZE_T yynewbytes;                                            \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / sizeof (*yyptr);                          \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
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
#define YYLAST   229

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  58
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  42
/* YYNRULES -- Number of rules.  */
#define YYNRULES  121
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  164

#define YYUNDEFTOK  2
#define YYMAXUTOK   312

/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                                \
  ((unsigned) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57
};

#if GRAM_DEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   281,   281,   290,   291,   295,   296,   302,   306,   311,
     312,   317,   318,   319,   320,   321,   326,   331,   332,   333,
     334,   335,   336,   336,   337,   338,   339,   340,   341,   342,
     343,   347,   348,   357,   358,   362,   373,   377,   381,   389,
     399,   400,   410,   411,   417,   430,   430,   435,   435,   440,
     444,   454,   455,   456,   457,   461,   462,   467,   468,   472,
     473,   477,   478,   479,   492,   501,   505,   509,   517,   518,
     522,   535,   536,   548,   552,   556,   564,   566,   571,   578,
     588,   592,   596,   604,   605,   613,   614,   620,   621,   622,
     629,   629,   637,   638,   639,   644,   647,   649,   651,   653,
     655,   657,   659,   661,   663,   668,   669,   678,   702,   703,
     704,   705,   717,   719,   734,   739,   740,   745,   754,   755,
     759,   760
};
#endif

#if GRAM_DEBUG || YYERROR_VERBOSE || 1
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "$undefined", "\"string\"", "\"%token\"",
  "\"%nterm\"", "\"%type\"", "\"%destructor\"", "\"%printer\"",
  "\"%left\"", "\"%right\"", "\"%nonassoc\"", "\"%precedence\"",
  "\"%prec\"", "\"%dprec\"", "\"%merge\"", "\"%code\"",
  "\"%default-prec\"", "\"%define\"", "\"%defines\"", "\"%error-verbose\"",
  "\"%expect\"", "\"%expect-rr\"", "\"%<flag>\"", "\"%file-prefix\"",
  "\"%glr-parser\"", "\"%initial-action\"", "\"%language\"",
  "\"%name-prefix\"", "\"%no-default-prec\"", "\"%no-lines\"",
  "\"%nondeterministic-parser\"", "\"%output\"", "\"%require\"",
  "\"%skeleton\"", "\"%start\"", "\"%token-table\"", "\"%verbose\"",
  "\"%yacc\"", "\"{...}\"", "\"%?{...}\"", "\"[identifier]\"", "\"char\"",
  "\"epilogue\"", "\"=\"", "\"identifier\"", "\"identifier:\"", "\"%%\"",
  "\"|\"", "\"%{...%}\"", "\";\"", "\"<tag>\"", "\"<*>\"", "\"<>\"",
  "\"integer\"", "\"%param\"", "\"%union\"", "\"%empty\"", "$accept",
  "input", "prologue_declarations", "prologue_declaration", "$@1",
  "params", "grammar_declaration", "code_props_type", "union_name",
  "symbol_declaration", "$@2", "$@3", "precedence_declarator", "tag.opt",
  "generic_symlist", "generic_symlist_item", "tag", "nterm_decls",
  "token_decls", "token_decl.1", "token_decl", "int.opt",
  "token_decls_for_prec", "token_decl_for_prec.1", "token_decl_for_prec",
  "symbol_decls", "symbol_decl.1", "grammar",
  "rules_or_grammar_declaration", "rules", "$@4", "rhses.1", "rhs",
  "named_ref.opt", "variable", "value", "id", "id_colon", "symbol",
  "string_as_id", "string_as_id.opt", "epilogue.opt", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312
};
# endif

#define YYPACT_NINF -123

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-123)))

#define YYTABLE_NINF -121

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
    -123,     6,   109,  -123,   -21,  -123,  -123,     2,  -123,  -123,
    -123,  -123,  -123,  -123,   -11,  -123,    -7,    46,  -123,     3,
      13,  -123,    77,  -123,    43,    87,    88,  -123,  -123,  -123,
      91,    92,    94,    32,  -123,  -123,  -123,   162,  -123,  -123,
    -123,    53,  -123,  -123,    61,  -123,    30,  -123,   -15,   -15,
    -123,  -123,  -123,    32,    50,    32,  -123,  -123,  -123,  -123,
      64,  -123,    23,  -123,  -123,  -123,  -123,  -123,  -123,  -123,
    -123,  -123,  -123,  -123,    54,  -123,    55,     8,  -123,  -123,
      65,    68,  -123,    69,    18,    32,    60,    32,  -123,    70,
    -123,    16,    72,    16,  -123,    70,  -123,    72,    32,    32,
    -123,  -123,  -123,  -123,  -123,  -123,  -123,  -123,   105,  -123,
    -123,  -123,  -123,  -123,   110,  -123,  -123,  -123,  -123,    18,
    -123,  -123,  -123,    32,    32,  -123,  -123,  -123,    16,    16,
    -123,   147,    32,  -123,  -123,  -123,  -123,    32,    16,  -123,
    -123,    37,   172,  -123,  -123,    32,    97,   101,    99,   100,
    -123,  -123,  -123,   116,    65,   172,  -123,  -123,  -123,  -123,
    -123,    65,  -123,  -123
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       3,     0,     0,     1,     0,    47,    45,     0,    40,    41,
      51,    52,    53,    54,     0,    36,     0,     9,    11,     0,
       0,     7,     0,    15,     0,     0,     0,    37,    19,    20,
       0,     0,     0,     0,    26,    27,    28,     0,     6,    30,
      22,    42,     4,     5,     0,    33,     0,    29,     0,     0,
     117,   113,   112,     0,    49,    80,   115,    83,   116,    38,
       0,   107,   108,    10,    12,    13,    14,    16,    17,    18,
      21,    24,    25,    34,     0,   114,     0,     0,    85,    87,
     105,     0,    43,     0,     0,     0,    50,    73,    76,    71,
      79,     0,    48,    65,    68,    71,    46,    64,    81,     0,
      84,    39,   110,   111,   109,     8,    89,    88,     0,    86,
       2,   106,    90,    32,    23,    44,    61,    62,    63,    35,
      57,    60,    59,    74,     0,    77,    72,    78,    66,     0,
      69,   118,    82,   121,    95,    31,    58,    75,    67,   119,
      70,    91,    92,    95,    94,     0,     0,     0,     0,     0,
      98,    56,    99,     0,   105,    93,   100,   101,   102,   103,
     104,   105,    96,    97
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -123,  -123,  -123,  -123,  -123,  -123,   155,  -123,  -123,  -123,
    -123,  -123,  -123,  -123,  -123,    41,  -123,  -123,   112,   -84,
     -62,    67,  -123,   -83,   -64,  -123,   -43,  -123,   103,  -123,
    -123,  -123,    33,  -122,  -123,  -123,   -45,  -123,   -33,   -35,
    -123,  -123
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    42,    81,   114,    76,    44,    83,    45,
      49,    48,    46,   153,   119,   120,   121,    96,    92,    93,
      94,   127,    86,    87,    88,    54,    55,    77,    78,    79,
     134,   141,   142,   112,    62,   105,    56,    80,    57,    58,
     140,   110
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      73,    89,   123,    95,    95,    50,     3,   128,  -120,    74,
      98,    90,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    50,   100,   125,    14,    15,   102,    51,    59,    47,
      52,   130,   162,    50,    60,    50,    91,    27,    61,   163,
      89,   137,    89,    33,    51,   138,    95,    52,    95,    63,
      90,   122,    90,    53,    75,   108,   132,    64,    51,   125,
      51,    52,   103,    52,    41,   100,   130,    65,   104,   116,
     117,   118,    51,   125,    51,    52,   130,    52,    89,    89,
      66,    85,    67,    95,    95,   143,   122,   144,    90,    90,
      68,    69,    89,    95,    70,    71,   139,    72,    82,   100,
      84,    99,    90,   101,   106,   107,   111,   113,   115,   154,
       4,   124,   156,     5,     6,     7,     8,     9,    10,    11,
      12,    13,   154,   129,   126,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,   133,   135,
      50,   157,   158,   159,   160,   161,    37,    43,    38,    39,
     136,    97,   131,    74,    40,    41,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    50,   155,     0,    14,    15,
     109,     0,     0,     0,     0,   145,   146,   147,     0,     0,
       0,    27,     0,   148,   149,     0,     0,    33,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    75,     0,
       0,   -55,   150,     0,    51,     0,     0,    52,    41,     0,
       0,     0,     0,   151,     0,     0,     0,     0,     0,   152
};

static const yytype_int16 yycheck[] =
{
      33,    46,    85,    48,    49,     3,     0,    91,     0,     1,
      53,    46,     4,     5,     6,     7,     8,     9,    10,    11,
      12,     3,    55,    87,    16,    17,     3,    42,    39,    50,
      45,    93,   154,     3,    45,     3,    51,    29,    45,   161,
      85,   124,    87,    35,    42,   129,    91,    45,    93,     3,
      85,    84,    87,    51,    46,    47,    99,    54,    42,   123,
      42,    45,    39,    45,    56,    98,   128,    54,    45,    51,
      52,    53,    42,   137,    42,    45,   138,    45,   123,   124,
       3,    51,    39,   128,   129,    48,   119,    50,   123,   124,
       3,     3,   137,   138,     3,     3,   131,     3,    45,   132,
      39,    51,   137,    39,    50,    50,    41,    39,    39,   142,
       1,    51,   145,     4,     5,     6,     7,     8,     9,    10,
      11,    12,   155,    51,    54,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    43,    39,
       3,    54,    51,    54,    54,    39,    47,     2,    49,    50,
     119,    49,    95,     1,    55,    56,     4,     5,     6,     7,
       8,     9,    10,    11,    12,     3,   143,    -1,    16,    17,
      77,    -1,    -1,    -1,    -1,    13,    14,    15,    -1,    -1,
      -1,    29,    -1,    21,    22,    -1,    -1,    35,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    46,    -1,
      -1,    39,    40,    -1,    42,    -1,    -1,    45,    56,    -1,
      -1,    -1,    -1,    51,    -1,    -1,    -1,    -1,    -1,    57
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    59,    60,     0,     1,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    47,    49,    50,
      55,    56,    61,    64,    65,    67,    70,    50,    69,    68,
       3,    42,    45,    51,    83,    84,    94,    96,    97,    39,
      45,    45,    92,     3,    54,    54,     3,    39,     3,     3,
       3,     3,     3,    96,     1,    46,    64,    85,    86,    87,
      95,    62,    45,    66,    39,    51,    80,    81,    82,    94,
      97,    51,    76,    77,    78,    94,    75,    76,    84,    51,
      96,    39,     3,    39,    45,    93,    50,    50,    47,    86,
      99,    41,    91,    39,    63,    39,    51,    52,    53,    72,
      73,    74,    96,    81,    51,    82,    54,    79,    77,    51,
      78,    79,    84,    43,    88,    39,    73,    81,    77,    97,
      98,    89,    90,    48,    50,    13,    14,    15,    21,    22,
      40,    51,    57,    71,    96,    90,    96,    54,    51,    54,
      54,    39,    91,    91
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    58,    59,    60,    60,    61,    61,    61,    61,    61,
      61,    61,    61,    61,    61,    61,    61,    61,    61,    61,
      61,    61,    62,    61,    61,    61,    61,    61,    61,    61,
      61,    63,    63,    64,    64,    64,    64,    64,    64,    64,
      65,    65,    66,    66,    64,    68,    67,    69,    67,    67,
      67,    70,    70,    70,    70,    71,    71,    72,    72,    73,
      73,    74,    74,    74,    75,    76,    76,    76,    77,    77,
      78,    79,    79,    80,    80,    80,    81,    81,    82,    82,
      83,    83,    83,    84,    84,    85,    85,    86,    86,    86,
      88,    87,    89,    89,    89,    90,    90,    90,    90,    90,
      90,    90,    90,    90,    90,    91,    91,    92,    93,    93,
      93,    93,    94,    94,    95,    96,    96,    97,    98,    98,
      99,    99
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     4,     0,     2,     1,     1,     1,     3,     1,
       2,     1,     2,     2,     2,     1,     2,     2,     2,     1,
       1,     2,     0,     3,     2,     2,     1,     1,     1,     2,
       1,     2,     1,     1,     2,     3,     1,     1,     2,     3,
       1,     1,     0,     1,     3,     0,     3,     0,     3,     2,
       2,     1,     1,     1,     1,     0,     1,     1,     2,     1,
       1,     1,     1,     1,     1,     1,     2,     3,     1,     2,
       3,     0,     1,     1,     2,     3,     1,     2,     2,     1,
       1,     2,     3,     1,     2,     1,     2,     1,     2,     2,
       0,     4,     1,     3,     2,     0,     3,     4,     2,     2,
       3,     3,     3,     3,     3,     0,     1,     1,     0,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     0,     1,
       0,     2
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
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

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256


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

#ifndef YY_LOCATION_PRINT
# if defined GRAM_LTYPE_IS_TRIVIAL && GRAM_LTYPE_IS_TRIVIAL

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

#  define YY_LOCATION_PRINT(File, Loc)          \
  yy_location_print_ (File, &(Loc))

# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value, Location); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
{
  FILE *yyoutput = yyo;
  YYUSE (yyoutput);
  YYUSE (yylocationp);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyo, yytoknum[yytype], *yyvaluep);
# endif
  switch (yytype)
    {
    case 3: /* "string"  */
#line 207 "src/parse-gram.y" /* yacc.c:680  */
      { fputs (quotearg_style (c_quoting_style, ((*yyvaluep).STRING)), yyo); }
#line 1108 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 20: /* "%error-verbose"  */
#line 215 "src/parse-gram.y" /* yacc.c:680  */
      { fputs (((*yyvaluep).PERCENT_ERROR_VERBOSE), yyo); }
#line 1114 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 23: /* "%<flag>"  */
#line 218 "src/parse-gram.y" /* yacc.c:680  */
      { fprintf (yyo, "%%%s", ((*yyvaluep).PERCENT_FLAG)); }
#line 1120 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 24: /* "%file-prefix"  */
#line 215 "src/parse-gram.y" /* yacc.c:680  */
      { fputs (((*yyvaluep).PERCENT_FILE_PREFIX), yyo); }
#line 1126 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 28: /* "%name-prefix"  */
#line 215 "src/parse-gram.y" /* yacc.c:680  */
      { fputs (((*yyvaluep).PERCENT_NAME_PREFIX), yyo); }
#line 1132 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 38: /* "%yacc"  */
#line 215 "src/parse-gram.y" /* yacc.c:680  */
      { fputs (((*yyvaluep).PERCENT_YACC), yyo); }
#line 1138 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 39: /* "{...}"  */
#line 208 "src/parse-gram.y" /* yacc.c:680  */
      { fprintf (yyo, "{\n%s\n}", ((*yyvaluep).BRACED_CODE)); }
#line 1144 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 40: /* "%?{...}"  */
#line 208 "src/parse-gram.y" /* yacc.c:680  */
      { fprintf (yyo, "{\n%s\n}", ((*yyvaluep).BRACED_PREDICATE)); }
#line 1150 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 41: /* "[identifier]"  */
#line 216 "src/parse-gram.y" /* yacc.c:680  */
      { fprintf (yyo, "[%s]", ((*yyvaluep).BRACKETED_ID)); }
#line 1156 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 42: /* "char"  */
#line 204 "src/parse-gram.y" /* yacc.c:680  */
      { fputs (char_name (((*yyvaluep).CHAR)), yyo); }
#line 1162 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 43: /* "epilogue"  */
#line 208 "src/parse-gram.y" /* yacc.c:680  */
      { fprintf (yyo, "{\n%s\n}", ((*yyvaluep).EPILOGUE)); }
#line 1168 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 45: /* "identifier"  */
#line 215 "src/parse-gram.y" /* yacc.c:680  */
      { fputs (((*yyvaluep).ID), yyo); }
#line 1174 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 46: /* "identifier:"  */
#line 217 "src/parse-gram.y" /* yacc.c:680  */
      { fprintf (yyo, "%s:", ((*yyvaluep).ID_COLON)); }
#line 1180 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 49: /* "%{...%}"  */
#line 208 "src/parse-gram.y" /* yacc.c:680  */
      { fprintf (yyo, "{\n%s\n}", ((*yyvaluep).PROLOGUE)); }
#line 1186 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 51: /* "<tag>"  */
#line 219 "src/parse-gram.y" /* yacc.c:680  */
      { fprintf (yyo, "<%s>", ((*yyvaluep).TAG)); }
#line 1192 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 54: /* "integer"  */
#line 222 "src/parse-gram.y" /* yacc.c:680  */
      { fprintf (yyo, "%d", ((*yyvaluep).INT)); }
#line 1198 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 55: /* "%param"  */
#line 261 "src/parse-gram.y" /* yacc.c:680  */
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
#line 1215 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 65: /* code_props_type  */
#line 397 "src/parse-gram.y" /* yacc.c:680  */
      { fprintf (yyo, "%s", code_props_type_string (((*yyvaluep).code_props_type))); }
#line 1221 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 71: /* tag.opt  */
#line 215 "src/parse-gram.y" /* yacc.c:680  */
      { fputs (((*yyvaluep).yytype_71), yyo); }
#line 1227 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 72: /* generic_symlist  */
#line 231 "src/parse-gram.y" /* yacc.c:680  */
      { symbol_list_syms_print (((*yyvaluep).generic_symlist), yyo); }
#line 1233 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 73: /* generic_symlist_item  */
#line 231 "src/parse-gram.y" /* yacc.c:680  */
      { symbol_list_syms_print (((*yyvaluep).generic_symlist_item), yyo); }
#line 1239 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 74: /* tag  */
#line 219 "src/parse-gram.y" /* yacc.c:680  */
      { fprintf (yyo, "<%s>", ((*yyvaluep).tag)); }
#line 1245 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 75: /* nterm_decls  */
#line 231 "src/parse-gram.y" /* yacc.c:680  */
      { symbol_list_syms_print (((*yyvaluep).nterm_decls), yyo); }
#line 1251 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 76: /* token_decls  */
#line 231 "src/parse-gram.y" /* yacc.c:680  */
      { symbol_list_syms_print (((*yyvaluep).token_decls), yyo); }
#line 1257 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 77: /* token_decl.1  */
#line 231 "src/parse-gram.y" /* yacc.c:680  */
      { symbol_list_syms_print (((*yyvaluep).yytype_77), yyo); }
#line 1263 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 78: /* token_decl  */
#line 225 "src/parse-gram.y" /* yacc.c:680  */
      { fprintf (yyo, "%s", ((*yyvaluep).token_decl) ? ((*yyvaluep).token_decl)->tag : "<NULL>"); }
#line 1269 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 79: /* int.opt  */
#line 222 "src/parse-gram.y" /* yacc.c:680  */
      { fprintf (yyo, "%d", ((*yyvaluep).yytype_79)); }
#line 1275 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 80: /* token_decls_for_prec  */
#line 231 "src/parse-gram.y" /* yacc.c:680  */
      { symbol_list_syms_print (((*yyvaluep).token_decls_for_prec), yyo); }
#line 1281 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 81: /* token_decl_for_prec.1  */
#line 231 "src/parse-gram.y" /* yacc.c:680  */
      { symbol_list_syms_print (((*yyvaluep).yytype_81), yyo); }
#line 1287 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 82: /* token_decl_for_prec  */
#line 225 "src/parse-gram.y" /* yacc.c:680  */
      { fprintf (yyo, "%s", ((*yyvaluep).token_decl_for_prec) ? ((*yyvaluep).token_decl_for_prec)->tag : "<NULL>"); }
#line 1293 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 83: /* symbol_decls  */
#line 231 "src/parse-gram.y" /* yacc.c:680  */
      { symbol_list_syms_print (((*yyvaluep).symbol_decls), yyo); }
#line 1299 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 84: /* symbol_decl.1  */
#line 231 "src/parse-gram.y" /* yacc.c:680  */
      { symbol_list_syms_print (((*yyvaluep).yytype_84), yyo); }
#line 1305 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 92: /* variable  */
#line 215 "src/parse-gram.y" /* yacc.c:680  */
      { fputs (((*yyvaluep).variable), yyo); }
#line 1311 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 93: /* value  */
#line 692 "src/parse-gram.y" /* yacc.c:680  */
      {
  switch (((*yyvaluep).value).kind)
    {
    case muscle_code:    fprintf (yyo,  "{%s}",  ((*yyvaluep).value).chars); break;
    case muscle_keyword: fprintf (yyo,   "%s",   ((*yyvaluep).value).chars); break;
    case muscle_string:  fprintf (yyo, "\"%s\"", ((*yyvaluep).value).chars); break;
    }
}
#line 1324 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 94: /* id  */
#line 225 "src/parse-gram.y" /* yacc.c:680  */
      { fprintf (yyo, "%s", ((*yyvaluep).id) ? ((*yyvaluep).id)->tag : "<NULL>"); }
#line 1330 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 95: /* id_colon  */
#line 226 "src/parse-gram.y" /* yacc.c:680  */
      { fprintf (yyo, "%s:", ((*yyvaluep).id_colon)->tag); }
#line 1336 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 96: /* symbol  */
#line 225 "src/parse-gram.y" /* yacc.c:680  */
      { fprintf (yyo, "%s", ((*yyvaluep).symbol) ? ((*yyvaluep).symbol)->tag : "<NULL>"); }
#line 1342 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 97: /* string_as_id  */
#line 225 "src/parse-gram.y" /* yacc.c:680  */
      { fprintf (yyo, "%s", ((*yyvaluep).string_as_id) ? ((*yyvaluep).string_as_id)->tag : "<NULL>"); }
#line 1348 "src/parse-gram.c" /* yacc.c:680  */
        break;

    case 98: /* string_as_id.opt  */
#line 225 "src/parse-gram.y" /* yacc.c:680  */
      { fprintf (yyo, "%s", ((*yyvaluep).yytype_98) ? ((*yyvaluep).yytype_98)->tag : "<NULL>"); }
#line 1354 "src/parse-gram.c" /* yacc.c:680  */
        break;

      default:
        break;
    }
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
{
  YYFPRINTF (yyo, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  YY_LOCATION_PRINT (yyo, *yylocationp);
  YYFPRINTF (yyo, ": ");
  yy_symbol_value_print (yyo, yytype, yyvaluep, yylocationp);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
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
yy_reduce_print (yytype_int16 *yyssp, YYSTYPE *yyvsp, YYLTYPE *yylsp, int yyrule)
{
  unsigned long yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[yyssp[yyi + 1 - yynrhs]],
                       &yyvsp[(yyi + 1) - (yynrhs)]
                       , &(yylsp[(yyi + 1) - (yynrhs)])                       );
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
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
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
   required.  Return 1 if memory is exhausted.  */
static int
yy_lac_stack_realloc (YYSIZE_T *yycapacity, YYSIZE_T yyadd,
#if GRAM_DEBUG
                      char const *yydebug_prefix,
                      char const *yydebug_suffix,
#endif
                      yytype_int16 **yybottom,
                      yytype_int16 *yybottom_no_free,
                      yytype_int16 **yytop, yytype_int16 *yytop_empty)
{
  YYSIZE_T yysize_old =
    (YYSIZE_T) (*yytop == yytop_empty ? 0 : *yytop - *yybottom + 1);
  YYSIZE_T yysize_new = yysize_old + yyadd;
  if (*yycapacity < yysize_new)
    {
      YYSIZE_T yyalloc = 2 * yysize_new;
      yytype_int16 *yybottom_new;
      /* Use YYMAXDEPTH for maximum stack size given that the stack
         should never need to grow larger than the main state stack
         needs to grow without LAC.  */
      if (YYMAXDEPTH < yysize_new)
        {
          YYDPRINTF ((stderr, "%smax size exceeded%s", yydebug_prefix,
                      yydebug_suffix));
          return 1;
        }
      if (YYMAXDEPTH < yyalloc)
        yyalloc = YYMAXDEPTH;
      yybottom_new =
        (yytype_int16*) YYSTACK_ALLOC (yyalloc * sizeof *yybottom_new);
      if (!yybottom_new)
        {
          YYDPRINTF ((stderr, "%srealloc failed%s", yydebug_prefix,
                      yydebug_suffix));
          return 1;
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
#define YY_LAC_ESTABLISH                                         \
do {                                                             \
  if (!yy_lac_established)                                       \
    {                                                            \
      YYDPRINTF ((stderr,                                        \
                  "LAC: initial context established for %s\n",   \
                  yytname[yytoken]));                            \
      yy_lac_established = 1;                                    \
      {                                                          \
        int yy_lac_status =                                      \
          yy_lac (yyesa, &yyes, &yyes_capacity, yyssp, yytoken); \
        if (yy_lac_status == 2)                                  \
          goto yyexhaustedlab;                                   \
        if (yy_lac_status == 1)                                  \
          goto yyerrlab;                                         \
      }                                                          \
    }                                                            \
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
      if (yydebug)                                                       \
        YYFPRINTF (stderr, "LAC: initial context discarded due to "      \
                   Event "\n");                                          \
      yy_lac_established = 0;                                            \
    }                                                                    \
} while (0)
#else
# define YY_LAC_DISCARD(Event) yy_lac_established = 0
#endif

/* Given the stack whose top is *YYSSP, return 0 iff YYTOKEN can
   eventually (after perhaps some reductions) be shifted, return 1 if
   not, or return 2 if memory is exhausted.  As preconditions and
   postconditions: *YYES_CAPACITY is the allocated size of the array to
   which *YYES points, and either *YYES = YYESA or *YYES points to an
   array allocated with YYSTACK_ALLOC.  yy_lac may overwrite the
   contents of either array, alter *YYES and *YYES_CAPACITY, and free
   any old *YYES other than YYESA.  */
static int
yy_lac (yytype_int16 *yyesa, yytype_int16 **yyes,
        YYSIZE_T *yyes_capacity, yytype_int16 *yyssp, int yytoken)
{
  yytype_int16 *yyes_prev = yyssp;
  yytype_int16 *yyesp = yyes_prev;
  YYDPRINTF ((stderr, "LAC: checking lookahead %s:", yytname[yytoken]));
  if (yytoken == YYUNDEFTOK)
    {
      YYDPRINTF ((stderr, " Always Err\n"));
      return 1;
    }
  while (1)
    {
      int yyrule = yypact[*yyesp];
      if (yypact_value_is_default (yyrule)
          || (yyrule += yytoken) < 0 || YYLAST < yyrule
          || yycheck[yyrule] != yytoken)
        {
          yyrule = yydefact[*yyesp];
          if (yyrule == 0)
            {
              YYDPRINTF ((stderr, " Err\n"));
              return 1;
            }
        }
      else
        {
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
      {
        YYSIZE_T yylen = yyr2[yyrule];
        YYDPRINTF ((stderr, " R%d", yyrule - 1));
        if (yyesp != yyes_prev)
          {
            YYSIZE_T yysize = (YYSIZE_T) (yyesp - *yyes + 1);
            if (yylen < yysize)
              {
                yyesp -= yylen;
                yylen = 0;
              }
            else
              {
                yylen -= yysize;
                yyesp = yyes_prev;
              }
          }
        if (yylen)
          yyesp = yyes_prev -= yylen;
      }
      {
        yytype_int16 yystate;
        {
          const int yylhs = yyr1[yyrule] - YYNTOKENS;
          const int yyi = yypgoto[yylhs] + *yyesp;
          yystate = ((yytype_int16)
                     (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyesp
                      ? yytable[yyi]
                      : yydefgoto[yylhs]));
        }
        if (yyesp == yyes_prev)
          {
            yyesp = *yyes;
            *yyesp = yystate;
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
                return 2;
              }
            *++yyesp = yystate;
          }
        YYDPRINTF ((stderr, " G%d", (int) yystate));
      }
    }
}


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
yystrlen (const char *yystr)
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            else
              goto append;

          append:
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return (YYSIZE_T) (yystpcpy (yyres, yystr) - yyres);
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.  In order to see if a particular token T is a
   valid looakhead, invoke yy_lac (YYESA, YYES, YYES_CAPACITY, YYSSP, T).

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store or if
   yy_lac returned 2.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyesa, yytype_int16 **yyes,
                YYSIZE_T *yyes_capacity, yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (YY_NULLPTR, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
       In the first two cases, it might appear that the current syntax
       error should have been detected in the previous state when yy_lac
       was invoked.  However, at that time, there might have been a
       different syntax error that discarded a different initial context
       during error recovery, leaving behind the current lookahead.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      YYDPRINTF ((stderr, "Constructing syntax error message\n"));
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          int yyx;

          for (yyx = 0; yyx < YYNTOKENS; ++yyx)
            if (yyx != YYTERROR && yyx != YYUNDEFTOK)
              {
                {
                  int yy_lac_status = yy_lac (yyesa, yyes, yyes_capacity,
                                              yyssp, yyx);
                  if (yy_lac_status == 2)
                    return 2;
                  if (yy_lac_status == 1)
                    continue;
                }
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                {
                  YYSIZE_T yysize1 = yysize + yytnamerr (YY_NULLPTR, yytname[yyx]);
                  if (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM)
                    yysize = yysize1;
                  else
                    return 2;
                }
              }
        }
# if GRAM_DEBUG
      else if (yydebug)
        YYFPRINTF (stderr, "No expected tokens.\n");
# endif
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
    default: /* Avoid compiler warnings. */
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  {
    YYSIZE_T yysize1 = yysize + yystrlen (yyformat);
    if (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM)
      yysize = yysize1;
    else
      return 2;
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep, YYLTYPE *yylocationp)
{
  YYUSE (yyvaluep);
  YYUSE (yylocationp);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  switch (yytype)
    {
    case 72: /* generic_symlist  */
#line 230 "src/parse-gram.y" /* yacc.c:1257  */
      { symbol_list_free (((*yyvaluep).generic_symlist)); }
#line 1944 "src/parse-gram.c" /* yacc.c:1257  */
        break;

    case 73: /* generic_symlist_item  */
#line 230 "src/parse-gram.y" /* yacc.c:1257  */
      { symbol_list_free (((*yyvaluep).generic_symlist_item)); }
#line 1950 "src/parse-gram.c" /* yacc.c:1257  */
        break;

    case 75: /* nterm_decls  */
#line 230 "src/parse-gram.y" /* yacc.c:1257  */
      { symbol_list_free (((*yyvaluep).nterm_decls)); }
#line 1956 "src/parse-gram.c" /* yacc.c:1257  */
        break;

    case 76: /* token_decls  */
#line 230 "src/parse-gram.y" /* yacc.c:1257  */
      { symbol_list_free (((*yyvaluep).token_decls)); }
#line 1962 "src/parse-gram.c" /* yacc.c:1257  */
        break;

    case 77: /* token_decl.1  */
#line 230 "src/parse-gram.y" /* yacc.c:1257  */
      { symbol_list_free (((*yyvaluep).yytype_77)); }
#line 1968 "src/parse-gram.c" /* yacc.c:1257  */
        break;

    case 80: /* token_decls_for_prec  */
#line 230 "src/parse-gram.y" /* yacc.c:1257  */
      { symbol_list_free (((*yyvaluep).token_decls_for_prec)); }
#line 1974 "src/parse-gram.c" /* yacc.c:1257  */
        break;

    case 81: /* token_decl_for_prec.1  */
#line 230 "src/parse-gram.y" /* yacc.c:1257  */
      { symbol_list_free (((*yyvaluep).yytype_81)); }
#line 1980 "src/parse-gram.c" /* yacc.c:1257  */
        break;

    case 83: /* symbol_decls  */
#line 230 "src/parse-gram.y" /* yacc.c:1257  */
      { symbol_list_free (((*yyvaluep).symbol_decls)); }
#line 1986 "src/parse-gram.c" /* yacc.c:1257  */
        break;

    case 84: /* symbol_decl.1  */
#line 230 "src/parse-gram.y" /* yacc.c:1257  */
      { symbol_list_free (((*yyvaluep).yytype_84)); }
#line 1992 "src/parse-gram.c" /* yacc.c:1257  */
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

    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.
       'yyls': related to locations.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    /* The location stack.  */
    YYLTYPE yylsa[YYINITDEPTH];
    YYLTYPE *yyls;
    YYLTYPE *yylsp;

    /* The locations where the error started and ended.  */
    YYLTYPE yyerror_range[3];

    YYSIZE_T yystacksize;

    yytype_int16 yyesa[20];
    yytype_int16 *yyes;
    YYSIZE_T yyes_capacity;

  int yy_lac_established = 0;
  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;
  YYLTYPE yyloc;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N), yylsp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yylsp = yyls = yylsa;
  yystacksize = YYINITDEPTH;

  yyes = yyesa;
  yyes_capacity = sizeof yyesa / sizeof *yyes;
  if (YYMAXDEPTH < yyes_capacity)
    yyes_capacity = YYMAXDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */

/* User initialization code.  */
#line 129 "src/parse-gram.y" /* yacc.c:1431  */
{
  /* Bison's grammar can initial empty locations, hence a default
     location is needed. */
  boundary_set (&yylloc.start, current_file, 1, 1);
  boundary_set (&yylloc.end, current_file, 1, 1);
}

#line 2117 "src/parse-gram.c" /* yacc.c:1431  */
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
| yynewstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  *yyssp = (yytype_int16) yystate;

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    goto yyexhaustedlab;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = (YYSIZE_T) (yyssp - yyss + 1);

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        YYSTYPE *yyvs1 = yyvs;
        yytype_int16 *yyss1 = yyss;
        YYLTYPE *yyls1 = yyls;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * sizeof (*yyssp),
                    &yyvs1, yysize * sizeof (*yyvsp),
                    &yyls1, yysize * sizeof (*yylsp),
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
        yytype_int16 *yyss1 = yyss;
        union yyalloc *yyptr =
          (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
        YYSTACK_RELOCATE (yyls_alloc, yyls);
# undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;
      yylsp = yyls + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
                  (unsigned long) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

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

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = yylex (&yylval, &yylloc);
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
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

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  YY_LAC_DISCARD ("shift");

  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END
  *++yylsp = yylloc;
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
#line 297 "src/parse-gram.y" /* yacc.c:1652  */
    {
      muscle_code_grow (union_seen ? "post_prologue" : "pre_prologue",
                        translate_code ((yyvsp[0].PROLOGUE), (yylsp[0]), true), (yylsp[0]));
      code_scanner_last_string_free ();
    }
#line 2322 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 7:
#line 303 "src/parse-gram.y" /* yacc.c:1652  */
    {
      muscle_percent_define_ensure ((yyvsp[0].PERCENT_FLAG), (yylsp[0]), true);
    }
#line 2330 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 8:
#line 307 "src/parse-gram.y" /* yacc.c:1652  */
    {
      muscle_percent_define_insert ((yyvsp[-1].variable), (yyloc), (yyvsp[0].value).kind, (yyvsp[0].value).chars,
                                    MUSCLE_PERCENT_DEFINE_GRAMMAR_FILE);
    }
#line 2339 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 9:
#line 311 "src/parse-gram.y" /* yacc.c:1652  */
    { defines_flag = true; }
#line 2345 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 10:
#line 313 "src/parse-gram.y" /* yacc.c:1652  */
    {
      defines_flag = true;
      spec_defines_file = xstrdup ((yyvsp[0].STRING));
    }
#line 2354 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 11:
#line 317 "src/parse-gram.y" /* yacc.c:1652  */
    { handle_error_verbose (&(yyloc), (yyvsp[0].PERCENT_ERROR_VERBOSE)); }
#line 2360 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 12:
#line 318 "src/parse-gram.y" /* yacc.c:1652  */
    { expected_sr_conflicts = (yyvsp[0].INT); }
#line 2366 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 13:
#line 319 "src/parse-gram.y" /* yacc.c:1652  */
    { expected_rr_conflicts = (yyvsp[0].INT); }
#line 2372 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 14:
#line 320 "src/parse-gram.y" /* yacc.c:1652  */
    { handle_file_prefix (&(yyloc), &(yylsp[-1]), (yyvsp[-1].PERCENT_FILE_PREFIX), (yyvsp[0].STRING)); }
#line 2378 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 15:
#line 322 "src/parse-gram.y" /* yacc.c:1652  */
    {
      nondeterministic_parser = true;
      glr_parser = true;
    }
#line 2387 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 16:
#line 327 "src/parse-gram.y" /* yacc.c:1652  */
    {
      muscle_code_grow ("initial_action", translate_code ((yyvsp[0].BRACED_CODE), (yylsp[0]), false), (yylsp[0]));
      code_scanner_last_string_free ();
    }
#line 2396 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 17:
#line 331 "src/parse-gram.y" /* yacc.c:1652  */
    { language_argmatch ((yyvsp[0].STRING), grammar_prio, (yylsp[-1])); }
#line 2402 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 18:
#line 332 "src/parse-gram.y" /* yacc.c:1652  */
    { handle_name_prefix (&(yyloc), (yyvsp[-1].PERCENT_NAME_PREFIX), (yyvsp[0].STRING)); }
#line 2408 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 19:
#line 333 "src/parse-gram.y" /* yacc.c:1652  */
    { no_lines_flag = true; }
#line 2414 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 20:
#line 334 "src/parse-gram.y" /* yacc.c:1652  */
    { nondeterministic_parser = true; }
#line 2420 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 21:
#line 335 "src/parse-gram.y" /* yacc.c:1652  */
    { spec_outfile = (yyvsp[0].STRING); }
#line 2426 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 22:
#line 336 "src/parse-gram.y" /* yacc.c:1652  */
    { current_param = (yyvsp[0].PERCENT_PARAM); }
#line 2432 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 23:
#line 336 "src/parse-gram.y" /* yacc.c:1652  */
    { current_param = param_none; }
#line 2438 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 24:
#line 337 "src/parse-gram.y" /* yacc.c:1652  */
    { handle_require (&(yylsp[0]), (yyvsp[0].STRING)); }
#line 2444 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 25:
#line 338 "src/parse-gram.y" /* yacc.c:1652  */
    { handle_skeleton (&(yylsp[0]), (yyvsp[0].STRING)); }
#line 2450 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 26:
#line 339 "src/parse-gram.y" /* yacc.c:1652  */
    { token_table_flag = true; }
#line 2456 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 27:
#line 340 "src/parse-gram.y" /* yacc.c:1652  */
    { report_flag |= report_states; }
#line 2462 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 28:
#line 341 "src/parse-gram.y" /* yacc.c:1652  */
    { handle_yacc (&(yyloc), (yyvsp[0].PERCENT_YACC)); }
#line 2468 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 29:
#line 342 "src/parse-gram.y" /* yacc.c:1652  */
    { current_class = unknown_sym; yyerrok; }
#line 2474 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 31:
#line 347 "src/parse-gram.y" /* yacc.c:1652  */
    { add_param (current_param, (yyvsp[0].BRACED_CODE), (yylsp[0])); }
#line 2480 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 32:
#line 348 "src/parse-gram.y" /* yacc.c:1652  */
    { add_param (current_param, (yyvsp[0].BRACED_CODE), (yylsp[0])); }
#line 2486 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 34:
#line 359 "src/parse-gram.y" /* yacc.c:1652  */
    {
      grammar_start_symbol_set ((yyvsp[0].symbol), (yylsp[0]));
    }
#line 2494 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 35:
#line 363 "src/parse-gram.y" /* yacc.c:1652  */
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
#line 2509 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 36:
#line 374 "src/parse-gram.y" /* yacc.c:1652  */
    {
      default_prec = true;
    }
#line 2517 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 37:
#line 378 "src/parse-gram.y" /* yacc.c:1652  */
    {
      default_prec = false;
    }
#line 2525 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 38:
#line 382 "src/parse-gram.y" /* yacc.c:1652  */
    {
      /* Do not invoke muscle_percent_code_grow here since it invokes
         muscle_user_name_list_grow.  */
      muscle_code_grow ("percent_code()",
                        translate_code_braceless ((yyvsp[0].BRACED_CODE), (yylsp[0])), (yylsp[0]));
      code_scanner_last_string_free ();
    }
#line 2537 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 39:
#line 390 "src/parse-gram.y" /* yacc.c:1652  */
    {
      muscle_percent_code_grow ((yyvsp[-1].ID), (yylsp[-1]), translate_code_braceless ((yyvsp[0].BRACED_CODE), (yylsp[0])), (yylsp[0]));
      code_scanner_last_string_free ();
    }
#line 2546 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 40:
#line 399 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.code_props_type) = destructor; }
#line 2552 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 41:
#line 400 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.code_props_type) = printer; }
#line 2558 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 42:
#line 410 "src/parse-gram.y" /* yacc.c:1652  */
    {}
#line 2564 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 43:
#line 411 "src/parse-gram.y" /* yacc.c:1652  */
    { muscle_percent_define_insert ("api.value.union.name",
                                         (yylsp[0]), muscle_keyword, (yyvsp[0].ID),
                                         MUSCLE_PERCENT_DEFINE_GRAMMAR_FILE); }
#line 2572 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 44:
#line 418 "src/parse-gram.y" /* yacc.c:1652  */
    {
      union_seen = true;
      muscle_code_grow ("union_members", translate_code_braceless ((yyvsp[0].BRACED_CODE), (yylsp[0])), (yylsp[0]));
      code_scanner_last_string_free ();
    }
#line 2582 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 45:
#line 430 "src/parse-gram.y" /* yacc.c:1652  */
    { current_class = nterm_sym; }
#line 2588 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 46:
#line 431 "src/parse-gram.y" /* yacc.c:1652  */
    {
      current_class = unknown_sym;
      symbol_list_free ((yyvsp[0].nterm_decls));
    }
#line 2597 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 47:
#line 435 "src/parse-gram.y" /* yacc.c:1652  */
    { current_class = token_sym; }
#line 2603 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 48:
#line 436 "src/parse-gram.y" /* yacc.c:1652  */
    {
      current_class = unknown_sym;
      symbol_list_free ((yyvsp[0].token_decls));
    }
#line 2612 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 49:
#line 441 "src/parse-gram.y" /* yacc.c:1652  */
    {
      symbol_list_free ((yyvsp[0].symbol_decls));
    }
#line 2620 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 50:
#line 445 "src/parse-gram.y" /* yacc.c:1652  */
    {
      ++current_prec;
      for (symbol_list *list = (yyvsp[0].token_decls_for_prec); list; list = list->next)
        symbol_precedence_set (list->content.sym, current_prec, (yyvsp[-1].precedence_declarator), (yylsp[-1]));
      symbol_list_free ((yyvsp[0].token_decls_for_prec));
    }
#line 2631 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 51:
#line 454 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.precedence_declarator) = left_assoc; }
#line 2637 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 52:
#line 455 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.precedence_declarator) = right_assoc; }
#line 2643 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 53:
#line 456 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.precedence_declarator) = non_assoc; }
#line 2649 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 54:
#line 457 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.precedence_declarator) = precedence_assoc; }
#line 2655 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 55:
#line 461 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.yytype_71) = NULL; }
#line 2661 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 56:
#line 462 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.yytype_71) = (yyvsp[0].TAG); }
#line 2667 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 58:
#line 468 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.generic_symlist) = symbol_list_append ((yyvsp[-1].generic_symlist), (yyvsp[0].generic_symlist_item)); }
#line 2673 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 59:
#line 472 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.generic_symlist_item) = symbol_list_sym_new ((yyvsp[0].symbol), (yylsp[0])); }
#line 2679 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 60:
#line 473 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.generic_symlist_item) = symbol_list_type_new ((yyvsp[0].tag), (yylsp[0])); }
#line 2685 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 62:
#line 478 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.tag) = uniqstr_new ("*"); }
#line 2691 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 63:
#line 479 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.tag) = uniqstr_new (""); }
#line 2697 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 65:
#line 502 "src/parse-gram.y" /* yacc.c:1652  */
    {
      (yyval.token_decls) = (yyvsp[0].yytype_77);
    }
#line 2705 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 66:
#line 506 "src/parse-gram.y" /* yacc.c:1652  */
    {
      (yyval.token_decls) = symbol_list_type_set ((yyvsp[0].yytype_77), (yyvsp[-1].TAG), (yylsp[-1]));
    }
#line 2713 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 67:
#line 510 "src/parse-gram.y" /* yacc.c:1652  */
    {
      (yyval.token_decls) = symbol_list_append ((yyvsp[-2].token_decls), symbol_list_type_set ((yyvsp[0].yytype_77), (yyvsp[-1].TAG), (yylsp[-1])));
    }
#line 2721 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 68:
#line 517 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.yytype_77) = symbol_list_sym_new ((yyvsp[0].token_decl), (yylsp[0])); }
#line 2727 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 69:
#line 518 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.yytype_77) = symbol_list_append ((yyvsp[-1].yytype_77), symbol_list_sym_new ((yyvsp[0].token_decl), (yylsp[0]))); }
#line 2733 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 70:
#line 523 "src/parse-gram.y" /* yacc.c:1652  */
    {
      (yyval.token_decl) = (yyvsp[-2].id);
      symbol_class_set ((yyvsp[-2].id), current_class, (yylsp[-2]), true);
      if (0 <= (yyvsp[-1].yytype_79))
        symbol_user_token_number_set ((yyvsp[-2].id), (yyvsp[-1].yytype_79), (yylsp[-1]));
      if ((yyvsp[0].yytype_98))
        symbol_make_alias ((yyvsp[-2].id), (yyvsp[0].yytype_98), (yylsp[0]));
    }
#line 2746 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 71:
#line 535 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.yytype_79) = -1; }
#line 2752 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 73:
#line 549 "src/parse-gram.y" /* yacc.c:1652  */
    {
      (yyval.token_decls_for_prec) = (yyvsp[0].yytype_81);
    }
#line 2760 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 74:
#line 553 "src/parse-gram.y" /* yacc.c:1652  */
    {
      (yyval.token_decls_for_prec) = symbol_list_type_set ((yyvsp[0].yytype_81), (yyvsp[-1].TAG), (yylsp[-1]));
    }
#line 2768 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 75:
#line 557 "src/parse-gram.y" /* yacc.c:1652  */
    {
      (yyval.token_decls_for_prec) = symbol_list_append ((yyvsp[-2].token_decls_for_prec), symbol_list_type_set ((yyvsp[0].yytype_81), (yyvsp[-1].TAG), (yylsp[-1])));
    }
#line 2776 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 76:
#line 565 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.yytype_81) = symbol_list_sym_new ((yyvsp[0].token_decl_for_prec), (yylsp[0])); }
#line 2782 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 77:
#line 567 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.yytype_81) = symbol_list_append ((yyvsp[-1].yytype_81), symbol_list_sym_new ((yyvsp[0].token_decl_for_prec), (yylsp[0]))); }
#line 2788 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 78:
#line 572 "src/parse-gram.y" /* yacc.c:1652  */
    {
      (yyval.token_decl_for_prec) = (yyvsp[-1].id);
      symbol_class_set ((yyvsp[-1].id), token_sym, (yylsp[-1]), false);
      if (0 <= (yyvsp[0].yytype_79))
        symbol_user_token_number_set ((yyvsp[-1].id), (yyvsp[0].yytype_79), (yylsp[0]));
    }
#line 2799 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 80:
#line 589 "src/parse-gram.y" /* yacc.c:1652  */
    {
      (yyval.symbol_decls) = (yyvsp[0].yytype_84);
    }
#line 2807 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 81:
#line 593 "src/parse-gram.y" /* yacc.c:1652  */
    {
      (yyval.symbol_decls) = symbol_list_type_set ((yyvsp[0].yytype_84), (yyvsp[-1].TAG), (yylsp[-1]));
    }
#line 2815 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 82:
#line 597 "src/parse-gram.y" /* yacc.c:1652  */
    {
      (yyval.symbol_decls) = symbol_list_append ((yyvsp[-2].symbol_decls), symbol_list_type_set ((yyvsp[0].yytype_84), (yyvsp[-1].TAG), (yylsp[-1])));
    }
#line 2823 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 83:
#line 604 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.yytype_84) = symbol_list_sym_new ((yyvsp[0].symbol), (yylsp[0])); }
#line 2829 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 84:
#line 605 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.yytype_84) = symbol_list_append ((yyvsp[-1].yytype_84), symbol_list_sym_new ((yyvsp[0].symbol), (yylsp[0]))); }
#line 2835 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 89:
#line 623 "src/parse-gram.y" /* yacc.c:1652  */
    {
      yyerrok;
    }
#line 2843 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 90:
#line 629 "src/parse-gram.y" /* yacc.c:1652  */
    { current_lhs ((yyvsp[-1].id_colon), (yylsp[-1]), (yyvsp[0].yytype_91)); }
#line 2849 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 91:
#line 630 "src/parse-gram.y" /* yacc.c:1652  */
    {
      /* Free the current lhs. */
      current_lhs (0, (yylsp[-3]), 0);
    }
#line 2858 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 92:
#line 637 "src/parse-gram.y" /* yacc.c:1652  */
    { grammar_current_rule_end ((yylsp[0])); }
#line 2864 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 93:
#line 638 "src/parse-gram.y" /* yacc.c:1652  */
    { grammar_current_rule_end ((yylsp[0])); }
#line 2870 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 95:
#line 645 "src/parse-gram.y" /* yacc.c:1652  */
    { grammar_current_rule_begin (current_lhs_symbol, current_lhs_location,
                                  current_lhs_named_ref); }
#line 2877 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 96:
#line 648 "src/parse-gram.y" /* yacc.c:1652  */
    { grammar_current_rule_symbol_append ((yyvsp[-1].symbol), (yylsp[-1]), (yyvsp[0].yytype_91)); }
#line 2883 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 97:
#line 650 "src/parse-gram.y" /* yacc.c:1652  */
    { grammar_current_rule_action_append ((yyvsp[-1].BRACED_CODE), (yylsp[-1]), (yyvsp[0].yytype_91), (yyvsp[-2].yytype_71)); }
#line 2889 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 98:
#line 652 "src/parse-gram.y" /* yacc.c:1652  */
    { grammar_current_rule_predicate_append ((yyvsp[0].BRACED_PREDICATE), (yylsp[0])); }
#line 2895 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 99:
#line 654 "src/parse-gram.y" /* yacc.c:1652  */
    { grammar_current_rule_empty_set ((yylsp[0])); }
#line 2901 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 100:
#line 656 "src/parse-gram.y" /* yacc.c:1652  */
    { grammar_current_rule_prec_set ((yyvsp[0].symbol), (yylsp[0])); }
#line 2907 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 101:
#line 658 "src/parse-gram.y" /* yacc.c:1652  */
    { grammar_current_rule_dprec_set ((yyvsp[0].INT), (yylsp[0])); }
#line 2913 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 102:
#line 660 "src/parse-gram.y" /* yacc.c:1652  */
    { grammar_current_rule_merge_set ((yyvsp[0].TAG), (yylsp[0])); }
#line 2919 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 103:
#line 662 "src/parse-gram.y" /* yacc.c:1652  */
    { grammar_current_rule_expect_sr ((yyvsp[0].INT), (yylsp[0])); }
#line 2925 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 104:
#line 664 "src/parse-gram.y" /* yacc.c:1652  */
    { grammar_current_rule_expect_rr ((yyvsp[0].INT), (yylsp[0])); }
#line 2931 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 105:
#line 668 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.yytype_91) = 0; }
#line 2937 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 106:
#line 669 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.yytype_91) = named_ref_new ((yyvsp[0].BRACKETED_ID), (yylsp[0])); }
#line 2943 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 108:
#line 702 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.value).kind = muscle_keyword; (yyval.value).chars = ""; }
#line 2949 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 109:
#line 703 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.value).kind = muscle_keyword; (yyval.value).chars = (yyvsp[0].ID); }
#line 2955 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 110:
#line 704 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.value).kind = muscle_string;  (yyval.value).chars = (yyvsp[0].STRING); }
#line 2961 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 111:
#line 705 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.value).kind = muscle_code;    (yyval.value).chars = strip_braces ((yyvsp[0].BRACED_CODE)); }
#line 2967 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 112:
#line 718 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.id) = symbol_from_uniqstr ((yyvsp[0].ID), (yylsp[0])); }
#line 2973 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 113:
#line 720 "src/parse-gram.y" /* yacc.c:1652  */
    {
      if (current_class == nterm_sym)
        {
          gram_error (&(yylsp[0]),
                      _("character literals cannot be nonterminals"));
          YYERROR;
        }
      (yyval.id) = symbol_get (char_name ((yyvsp[0].CHAR)), (yylsp[0]));
      symbol_class_set ((yyval.id), token_sym, (yylsp[0]), false);
      symbol_user_token_number_set ((yyval.id), (yyvsp[0].CHAR), (yylsp[0]));
    }
#line 2989 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 114:
#line 734 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.id_colon) = symbol_from_uniqstr ((yyvsp[0].ID_COLON), (yylsp[0])); }
#line 2995 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 117:
#line 746 "src/parse-gram.y" /* yacc.c:1652  */
    {
      (yyval.string_as_id) = symbol_get (quotearg_style (c_quoting_style, (yyvsp[0].STRING)), (yylsp[0]));
      symbol_class_set ((yyval.string_as_id), token_sym, (yylsp[0]), false);
    }
#line 3004 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 118:
#line 754 "src/parse-gram.y" /* yacc.c:1652  */
    { (yyval.yytype_98) = NULL; }
#line 3010 "src/parse-gram.c" /* yacc.c:1652  */
    break;

  case 121:
#line 761 "src/parse-gram.y" /* yacc.c:1652  */
    {
      muscle_code_grow ("epilogue", translate_code ((yyvsp[0].EPILOGUE), (yylsp[0]), true), (yylsp[0]));
      code_scanner_last_string_free ();
    }
#line 3019 "src/parse-gram.c" /* yacc.c:1652  */
    break;


#line 3023 "src/parse-gram.c" /* yacc.c:1652  */
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
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

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
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (&yylloc, YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyesa, &yyes, &yyes_capacity, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        if (yychar != YYEMPTY)
          YY_LAC_ESTABLISH;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (&yylloc, yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }

  yyerror_range[1] = yylloc;

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval, &yylloc);
          yychar = YYEMPTY;
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

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYTERROR;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
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
                  yystos[yystate], yyvsp, yylsp);
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
  /* Using YYLLOC is tempting, but would change the location of
     the lookahead.  YYLOC is available though.  */
  YYLLOC_DEFAULT (yyloc, yyerror_range, 2);
  *++yylsp = yyloc;

  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

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
  if (yychar != YYEMPTY)
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
                  yystos[*yyssp], yyvsp, yylsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
  if (yyes != yyesa)
    YYSTACK_FREE (yyes);
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  return yyresult;
}
#line 767 "src/parse-gram.y" /* yacc.c:1918  */


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
handle_require (location const *loc, char const *version)
{
  /* Changes of behavior are only on minor version changes, so "3.0.5"
     is the same as "3.0". */
  errno = 0;
  char* cp = NULL;
  unsigned long major = strtoul (version, &cp, 10);
  if (errno || *cp != '.')
    {
      complain (loc, complaint, _("invalid version requirement: %s"),
                version);
      return;
    }
  ++cp;
  unsigned long minor = strtoul (cp, NULL, 10);
  if (errno)
    {
      complain (loc, complaint, _("invalid version requirement: %s"),
                version);
      return;
    }
  required_version = major * 100 + minor;
  /* Pretend to be at least 3.2, even if we are only 3.1-211, as it
     allows us to check features published in 3.2 while developping
     3.2.  */
  const char* api_version = "3.2";
  const char* package_version =
    strverscmp (api_version, PACKAGE_VERSION) > 0
    ? api_version : PACKAGE_VERSION;
  if (strverscmp (version, package_version) > 0)
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
      size_t dir_length = strlen (current_file);
      while (dir_length && current_file[dir_length - 1] != '/')
        --dir_length;
      while (dir_length && current_file[dir_length - 1] == '/')
        --dir_length;
      char *skeleton_build =
        xmalloc (dir_length + 1 + strlen (skeleton_user) + 1);
      if (dir_length > 0)
        {
          memcpy (skeleton_build, current_file, dir_length);
          skeleton_build[dir_length++] = '/';
        }
      strcpy (skeleton_build + dir_length, skeleton_user);
      skeleton_user = uniqstr_new (skeleton_build);
      free (skeleton_build);
    }
  skeleton_arg (skeleton_user, grammar_prio, *loc);
}

static void
handle_yacc (location const *loc, char const *directive)
{
  bison_directive (loc, directive);
  bool warned = false;

  if (location_empty (yacc_loc))
    yacc_loc = *loc;
  else
    {
      duplicate_directive (directive, yacc_loc, *loc);
      warned = true;
    }

  if (!warned
      && STRNEQ (directive, "%fixed-output-files")
      && STRNEQ (directive, "%yacc"))
    deprecated_directive (loc, directive, "%fixed-output-files");
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
  current_lhs_location = loc;
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
