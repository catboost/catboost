/* A Bison parser, made by GNU Bison 3.5.4.  */

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
#define YYBISON_VERSION "3.5.4"

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

#line 72 "src/parse-gram.c"
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

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 1
#endif

#include "parse-gram.h"


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
#line 261 "src/parse-gram.y"

  /** Add a lex-param and/or a parse-param.
   *
   * \param type  where to push this formal argument.
   * \param decl  the formal argument.  Destroyed.
   * \param loc   the location in the source.
   */
  static void add_param (param_type type, char *decl, location loc);
  static param_type current_param = param_none;

#line 216 "src/parse-gram.c"

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
#endif


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
#define YYNTOKENS  60
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  42
/* YYNRULES -- Number of rules.  */
#define YYNRULES  122
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  166

#define YYUNDEFTOK  2
#define YYMAXUTOK   314


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX) (YYX)

#if GRAM_DEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,   293,   293,   302,   303,   307,   308,   314,   318,   323,
     324,   329,   330,   331,   332,   333,   338,   343,   344,   345,
     346,   347,   348,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   360,   361,   370,   371,   375,   386,   390,   394,
     402,   412,   413,   423,   424,   430,   443,   443,   448,   448,
     453,   457,   467,   468,   469,   470,   474,   475,   480,   481,
     485,   486,   490,   491,   492,   505,   514,   518,   522,   530,
     531,   535,   548,   549,   561,   565,   569,   577,   579,   584,
     591,   601,   605,   609,   617,   622,   634,   635,   641,   642,
     643,   650,   650,   658,   659,   660,   665,   668,   670,   672,
     674,   676,   678,   680,   682,   684,   689,   690,   699,   723,
     724,   725,   726,   738,   740,   767,   772,   773,   778,   787,
     788,   792,   793
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
  "\"%nondeterministic-parser\"", "\"%output\"", "\"%pure-parser\"",
  "\"%require\"", "\"%skeleton\"", "\"%start\"", "\"%token-table\"",
  "\"%verbose\"", "\"%yacc\"", "\"{...}\"", "\"%?{...}\"",
  "\"[identifier]\"", "\"character literal\"", "\":\"", "\"epilogue\"",
  "\"=\"", "\"identifier\"", "\"identifier:\"", "\"%%\"", "\"|\"",
  "\"%{...%}\"", "\";\"", "\"<tag>\"", "\"<*>\"", "\"<>\"", "\"integer\"",
  "\"%param\"", "\"%union\"", "\"%empty\"", "$accept", "input",
  "prologue_declarations", "prologue_declaration", "$@1", "params",
  "grammar_declaration", "code_props_type", "union_name",
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
static const yytype_int16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314
};
# endif

#define YYPACT_NINF (-130)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-122)

#define yytable_value_is_error(Yyn) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
    -130,    36,   110,  -130,   -22,  -130,  -130,     2,  -130,  -130,
    -130,  -130,  -130,  -130,   -19,  -130,    -9,    40,  -130,   -17,
      -2,  -130,    57,  -130,    21,    66,    77,  -130,  -130,  -130,
      78,  -130,    87,    92,    44,  -130,  -130,  -130,   165,  -130,
    -130,  -130,    50,  -130,  -130,    58,  -130,    29,  -130,    15,
      15,  -130,  -130,  -130,    44,    47,    44,  -130,  -130,  -130,
    -130,    61,  -130,    30,  -130,  -130,  -130,  -130,  -130,  -130,
    -130,  -130,  -130,  -130,  -130,    51,  -130,    53,     8,  -130,
    -130,    64,    67,  -130,    68,    20,    44,    56,    44,  -130,
      69,  -130,   -37,    59,   -37,  -130,    69,  -130,    59,    44,
      44,  -130,  -130,  -130,  -130,  -130,  -130,  -130,  -130,    79,
    -130,  -130,  -130,  -130,  -130,   111,  -130,  -130,  -130,  -130,
      20,  -130,  -130,  -130,    44,    44,  -130,  -130,  -130,   -37,
     -37,  -130,   147,    44,  -130,   108,  -130,  -130,    44,   -37,
    -130,  -130,  -130,   -21,   175,  -130,  -130,    44,    97,   101,
      99,   100,  -130,  -130,  -130,   117,    64,   175,  -130,  -130,
    -130,  -130,  -130,    64,  -130,  -130
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
       0,   118,   114,   113,     0,    50,    81,   116,    84,   117,
      39,     0,   108,   109,    10,    12,    13,    14,    16,    17,
      18,    21,    25,    26,    35,     0,   115,     0,     0,    86,
      88,   106,     0,    44,     0,     0,     0,    51,    74,    77,
      72,    80,     0,    49,    66,    69,    72,    47,    65,    82,
       0,    85,    40,   111,   112,   110,     8,    90,    89,     0,
      87,     2,   107,    91,    33,    23,    45,    62,    63,    64,
      36,    58,    61,    60,    75,     0,    78,    73,    79,    67,
       0,    70,   119,    83,   122,     0,    32,    59,    76,    68,
     120,    71,    96,    92,    93,    96,    95,     0,     0,     0,
       0,     0,    99,    57,   100,     0,   106,    94,   101,   102,
     103,   104,   105,   106,    97,    98
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -130,  -130,  -130,  -130,  -130,  -130,   156,  -130,  -130,  -130,
    -130,  -130,  -130,  -130,  -130,    43,  -130,  -130,   114,   -66,
     -35,    83,  -130,   -84,   -53,  -130,   -47,  -130,    82,  -130,
    -130,  -130,    35,  -129,  -130,  -130,   -46,  -130,   -34,   -36,
    -130,  -130
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    43,    82,   115,    77,    45,    84,    46,
      50,    49,    47,   155,   120,   121,   122,    97,    93,    94,
      95,   128,    87,    88,    89,    55,    56,    78,    79,    80,
     135,   143,   144,   113,    63,   106,    57,    81,    58,    59,
     141,   111
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      74,    90,   124,    96,    96,    51,    52,    99,  -121,    75,
      53,    91,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    60,   101,    51,    14,    15,   129,   164,    61,   145,
      48,   146,    51,   103,   165,   126,     3,    27,    62,    65,
      90,   138,    90,    64,    34,    52,    96,    51,    96,    53,
      91,   123,    91,   133,    66,    54,    76,   109,    52,   131,
      67,    68,    53,    52,   139,   101,    42,    53,    92,    69,
     104,   126,    52,   117,   118,   119,    53,   105,    90,    90,
      70,    71,    86,    96,    96,   126,   123,    52,    91,    91,
      72,    53,    90,    96,   131,    73,   140,    83,    85,   101,
     100,   102,    91,   107,   131,   108,   112,   114,   116,   125,
     156,     4,   130,   158,     5,     6,     7,     8,     9,    10,
      11,    12,    13,   156,   134,   127,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    37,
      51,   136,   142,   159,   160,   161,   162,   163,    44,    38,
     110,    39,    40,   137,    98,     0,    75,    41,    42,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    51,   132,
     157,    14,    15,     0,     0,     0,     0,     0,   147,   148,
     149,     0,     0,     0,    27,     0,   150,   151,     0,     0,
       0,    34,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    76,     0,   -56,   152,     0,    52,     0,
       0,     0,    53,    42,     0,     0,     0,     0,   153,     0,
       0,     0,     0,     0,   154
};

static const yytype_int16 yycheck[] =
{
      34,    47,    86,    49,    50,     3,    43,    54,     0,     1,
      47,    47,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    40,    56,     3,    16,    17,    92,   156,    47,    50,
      52,    52,     3,     3,   163,    88,     0,    29,    47,    56,
      86,   125,    88,     3,    36,    43,    92,     3,    94,    47,
      86,    85,    88,   100,    56,    53,    48,    49,    43,    94,
       3,    40,    47,    43,   130,    99,    58,    47,    53,     3,
      40,   124,    43,    53,    54,    55,    47,    47,   124,   125,
       3,     3,    53,   129,   130,   138,   120,    43,   124,   125,
       3,    47,   138,   139,   129,     3,   132,    47,    40,   133,
      53,    40,   138,    52,   139,    52,    42,    40,    40,    53,
     144,     1,    53,   147,     4,     5,     6,     7,     8,     9,
      10,    11,    12,   157,    45,    56,    16,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    37,    38,    39,
       3,    40,    44,    56,    53,    56,    56,    40,     2,    49,
      78,    51,    52,   120,    50,    -1,     1,    57,    58,     4,
       5,     6,     7,     8,     9,    10,    11,    12,     3,    96,
     145,    16,    17,    -1,    -1,    -1,    -1,    -1,    13,    14,
      15,    -1,    -1,    -1,    29,    -1,    21,    22,    -1,    -1,
      -1,    36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    48,    -1,    40,    41,    -1,    43,    -1,
      -1,    -1,    47,    58,    -1,    -1,    -1,    -1,    53,    -1,
      -1,    -1,    -1,    -1,    59
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_int8 yystos[] =
{
       0,    61,    62,     0,     1,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    49,    51,
      52,    57,    58,    63,    66,    67,    69,    72,    52,    71,
      70,     3,    43,    47,    53,    85,    86,    96,    98,    99,
      40,    47,    47,    94,     3,    56,    56,     3,    40,     3,
       3,     3,     3,     3,    98,     1,    48,    66,    87,    88,
      89,    97,    64,    47,    68,    40,    53,    82,    83,    84,
      96,    99,    53,    78,    79,    80,    96,    77,    78,    86,
      53,    98,    40,     3,    40,    47,    95,    52,    52,    49,
      88,   101,    42,    93,    40,    65,    40,    53,    54,    55,
      74,    75,    76,    98,    83,    53,    84,    56,    81,    79,
      53,    80,    81,    86,    45,    90,    40,    75,    83,    79,
      99,   100,    44,    91,    92,    50,    52,    13,    14,    15,
      21,    22,    41,    53,    59,    73,    98,    92,    98,    56,
      53,    56,    56,    40,    93,    93
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_int8 yyr1[] =
{
       0,    60,    61,    62,    62,    63,    63,    63,    63,    63,
      63,    63,    63,    63,    63,    63,    63,    63,    63,    63,
      63,    63,    64,    63,    63,    63,    63,    63,    63,    63,
      63,    63,    65,    65,    66,    66,    66,    66,    66,    66,
      66,    67,    67,    68,    68,    66,    70,    69,    71,    69,
      69,    69,    72,    72,    72,    72,    73,    73,    74,    74,
      75,    75,    76,    76,    76,    77,    78,    78,    78,    79,
      79,    80,    81,    81,    82,    82,    82,    83,    83,    84,
      84,    85,    85,    85,    86,    86,    87,    87,    88,    88,
      88,    90,    89,    91,    91,    91,    92,    92,    92,    92,
      92,    92,    92,    92,    92,    92,    93,    93,    94,    95,
      95,    95,    95,    96,    96,    97,    98,    98,    99,   100,
     100,   101,   101
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
       2,     3,     0,     1,     1,     2,     3,     1,     2,     2,
       1,     1,     2,     3,     1,     2,     1,     2,     1,     2,
       2,     0,     5,     1,     3,     2,     0,     3,     4,     2,
       2,     3,     3,     3,     3,     3,     0,     1,     1,     0,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     0,
       1,     0,     2
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
/* "%code pre-printer" blocks.  */
#line 213 "src/parse-gram.y"
tron (yyo);

#line 935 "src/parse-gram.c"
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  switch (yytype)
    {
    case 3: /* "string"  */
#line 220 "src/parse-gram.y"
         { fputs (((*yyvaluep).STRING), yyo); }
#line 942 "src/parse-gram.c"
        break;

    case 20: /* "%error-verbose"  */
#line 227 "src/parse-gram.y"
         { fputs (((*yyvaluep).PERCENT_ERROR_VERBOSE), yyo); }
#line 948 "src/parse-gram.c"
        break;

    case 23: /* "%<flag>"  */
#line 230 "src/parse-gram.y"
         { fprintf (yyo, "%%%s", ((*yyvaluep).PERCENT_FLAG)); }
#line 954 "src/parse-gram.c"
        break;

    case 24: /* "%file-prefix"  */
#line 227 "src/parse-gram.y"
         { fputs (((*yyvaluep).PERCENT_FILE_PREFIX), yyo); }
#line 960 "src/parse-gram.c"
        break;

    case 28: /* "%name-prefix"  */
#line 227 "src/parse-gram.y"
         { fputs (((*yyvaluep).PERCENT_NAME_PREFIX), yyo); }
#line 966 "src/parse-gram.c"
        break;

    case 33: /* "%pure-parser"  */
#line 227 "src/parse-gram.y"
         { fputs (((*yyvaluep).PERCENT_PURE_PARSER), yyo); }
#line 972 "src/parse-gram.c"
        break;

    case 40: /* "{...}"  */
#line 220 "src/parse-gram.y"
         { fputs (((*yyvaluep).BRACED_CODE), yyo); }
#line 978 "src/parse-gram.c"
        break;

    case 41: /* "%?{...}"  */
#line 220 "src/parse-gram.y"
         { fputs (((*yyvaluep).BRACED_PREDICATE), yyo); }
#line 984 "src/parse-gram.c"
        break;

    case 42: /* "[identifier]"  */
#line 228 "src/parse-gram.y"
         { fprintf (yyo, "[%s]", ((*yyvaluep).BRACKETED_ID)); }
#line 990 "src/parse-gram.c"
        break;

    case 43: /* "character literal"  */
#line 217 "src/parse-gram.y"
         { fputs (char_name (((*yyvaluep).CHAR)), yyo); }
#line 996 "src/parse-gram.c"
        break;

    case 45: /* "epilogue"  */
#line 220 "src/parse-gram.y"
         { fputs (((*yyvaluep).EPILOGUE), yyo); }
#line 1002 "src/parse-gram.c"
        break;

    case 47: /* "identifier"  */
#line 227 "src/parse-gram.y"
         { fputs (((*yyvaluep).ID), yyo); }
#line 1008 "src/parse-gram.c"
        break;

    case 48: /* "identifier:"  */
#line 229 "src/parse-gram.y"
         { fprintf (yyo, "%s:", ((*yyvaluep).ID_COLON)); }
#line 1014 "src/parse-gram.c"
        break;

    case 51: /* "%{...%}"  */
#line 220 "src/parse-gram.y"
         { fputs (((*yyvaluep).PROLOGUE), yyo); }
#line 1020 "src/parse-gram.c"
        break;

    case 53: /* "<tag>"  */
#line 231 "src/parse-gram.y"
         { fprintf (yyo, "<%s>", ((*yyvaluep).TAG)); }
#line 1026 "src/parse-gram.c"
        break;

    case 56: /* "integer"  */
#line 234 "src/parse-gram.y"
         { fprintf (yyo, "%d", ((*yyvaluep).INT)); }
#line 1032 "src/parse-gram.c"
        break;

    case 57: /* "%param"  */
#line 273 "src/parse-gram.y"
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
#line 1049 "src/parse-gram.c"
        break;

    case 67: /* code_props_type  */
#line 410 "src/parse-gram.y"
         { fprintf (yyo, "%s", code_props_type_string (((*yyvaluep).code_props_type))); }
#line 1055 "src/parse-gram.c"
        break;

    case 73: /* tag.opt  */
#line 227 "src/parse-gram.y"
         { fputs (((*yyvaluep).yytype_73), yyo); }
#line 1061 "src/parse-gram.c"
        break;

    case 74: /* generic_symlist  */
#line 243 "src/parse-gram.y"
         { symbol_list_syms_print (((*yyvaluep).generic_symlist), yyo); }
#line 1067 "src/parse-gram.c"
        break;

    case 75: /* generic_symlist_item  */
#line 243 "src/parse-gram.y"
         { symbol_list_syms_print (((*yyvaluep).generic_symlist_item), yyo); }
#line 1073 "src/parse-gram.c"
        break;

    case 76: /* tag  */
#line 231 "src/parse-gram.y"
         { fprintf (yyo, "<%s>", ((*yyvaluep).tag)); }
#line 1079 "src/parse-gram.c"
        break;

    case 77: /* nterm_decls  */
#line 243 "src/parse-gram.y"
         { symbol_list_syms_print (((*yyvaluep).nterm_decls), yyo); }
#line 1085 "src/parse-gram.c"
        break;

    case 78: /* token_decls  */
#line 243 "src/parse-gram.y"
         { symbol_list_syms_print (((*yyvaluep).token_decls), yyo); }
#line 1091 "src/parse-gram.c"
        break;

    case 79: /* token_decl.1  */
#line 243 "src/parse-gram.y"
         { symbol_list_syms_print (((*yyvaluep).yytype_79), yyo); }
#line 1097 "src/parse-gram.c"
        break;

    case 80: /* token_decl  */
#line 237 "src/parse-gram.y"
         { fprintf (yyo, "%s", ((*yyvaluep).token_decl) ? ((*yyvaluep).token_decl)->tag : "<NULL>"); }
#line 1103 "src/parse-gram.c"
        break;

    case 81: /* int.opt  */
#line 234 "src/parse-gram.y"
         { fprintf (yyo, "%d", ((*yyvaluep).yytype_81)); }
#line 1109 "src/parse-gram.c"
        break;

    case 82: /* token_decls_for_prec  */
#line 243 "src/parse-gram.y"
         { symbol_list_syms_print (((*yyvaluep).token_decls_for_prec), yyo); }
#line 1115 "src/parse-gram.c"
        break;

    case 83: /* token_decl_for_prec.1  */
#line 243 "src/parse-gram.y"
         { symbol_list_syms_print (((*yyvaluep).yytype_83), yyo); }
#line 1121 "src/parse-gram.c"
        break;

    case 84: /* token_decl_for_prec  */
#line 237 "src/parse-gram.y"
         { fprintf (yyo, "%s", ((*yyvaluep).token_decl_for_prec) ? ((*yyvaluep).token_decl_for_prec)->tag : "<NULL>"); }
#line 1127 "src/parse-gram.c"
        break;

    case 85: /* symbol_decls  */
#line 243 "src/parse-gram.y"
         { symbol_list_syms_print (((*yyvaluep).symbol_decls), yyo); }
#line 1133 "src/parse-gram.c"
        break;

    case 86: /* symbol_decl.1  */
#line 243 "src/parse-gram.y"
         { symbol_list_syms_print (((*yyvaluep).yytype_86), yyo); }
#line 1139 "src/parse-gram.c"
        break;

    case 94: /* variable  */
#line 227 "src/parse-gram.y"
         { fputs (((*yyvaluep).variable), yyo); }
#line 1145 "src/parse-gram.c"
        break;

    case 95: /* value  */
#line 713 "src/parse-gram.y"
{
  switch (((*yyvaluep).value).kind)
    {
    case muscle_code:    fprintf (yyo,  "{%s}",  ((*yyvaluep).value).chars); break;
    case muscle_keyword: fprintf (yyo,   "%s",   ((*yyvaluep).value).chars); break;
    case muscle_string:  fprintf (yyo, "\"%s\"", ((*yyvaluep).value).chars); break;
    }
}
#line 1158 "src/parse-gram.c"
        break;

    case 96: /* id  */
#line 237 "src/parse-gram.y"
         { fprintf (yyo, "%s", ((*yyvaluep).id) ? ((*yyvaluep).id)->tag : "<NULL>"); }
#line 1164 "src/parse-gram.c"
        break;

    case 97: /* id_colon  */
#line 238 "src/parse-gram.y"
         { fprintf (yyo, "%s:", ((*yyvaluep).id_colon)->tag); }
#line 1170 "src/parse-gram.c"
        break;

    case 98: /* symbol  */
#line 237 "src/parse-gram.y"
         { fprintf (yyo, "%s", ((*yyvaluep).symbol) ? ((*yyvaluep).symbol)->tag : "<NULL>"); }
#line 1176 "src/parse-gram.c"
        break;

    case 99: /* string_as_id  */
#line 237 "src/parse-gram.y"
         { fprintf (yyo, "%s", ((*yyvaluep).string_as_id) ? ((*yyvaluep).string_as_id)->tag : "<NULL>"); }
#line 1182 "src/parse-gram.c"
        break;

    case 100: /* string_as_id.opt  */
#line 237 "src/parse-gram.y"
         { fprintf (yyo, "%s", ((*yyvaluep).yytype_100) ? ((*yyvaluep).yytype_100)->tag : "<NULL>"); }
#line 1188 "src/parse-gram.c"
        break;

      default:
        break;
    }
  YY_IGNORE_MAYBE_UNINITIALIZED_END
/* "%code post-printer" blocks.  */
#line 214 "src/parse-gram.y"
troff (yyo);

#line 1199 "src/parse-gram.c"
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
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp, YYLTYPE *yylsp, int yyrule)
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
                       yystos[+yyssp[yyi + 1 - yynrhs]],
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
          return 1;
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
yy_lac (yy_state_t *yyesa, yy_state_t **yyes,
        YYPTRDIFF_T *yyes_capacity, yy_state_t *yyssp, int yytoken)
{
  yy_state_t *yyes_prev = yyssp;
  yy_state_t *yyesp = yyes_prev;
  YYDPRINTF ((stderr, "LAC: checking lookahead %s:", yytname[yytoken]));
  if (yytoken == YYUNDEFTOK)
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
          yyrule = yydefact[+*yyesp];
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
        YYPTRDIFF_T yylen = yyr2[yyrule];
        YYDPRINTF ((stderr, " R%d", yyrule - 1));
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
                yylen -= yysize;
                yyesp = yyes_prev;
              }
          }
        if (yylen)
          yyesp = yyes_prev -= yylen;
      }
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
                return 2;
              }
            YY_IGNORE_USELESS_CAST_BEGIN
            *++yyesp = YY_CAST (yy_state_t, yystate);
            YY_IGNORE_USELESS_CAST_END
          }
        YYDPRINTF ((stderr, " G%d", yystate));
      }
    }
}


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen(S) (YY_CAST (YYPTRDIFF_T, strlen (S)))
#  else
/* Return the length of YYSTR.  */
static YYPTRDIFF_T
yystrlen (const char *yystr)
{
  YYPTRDIFF_T yylen;
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
static YYPTRDIFF_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYPTRDIFF_T yyn = 0;
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

  if (yyres)
    return yystpcpy (yyres, yystr) - yyres;
  else
    return yystrlen (yystr);
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
yysyntax_error (YYPTRDIFF_T *yymsg_alloc, char **yymsg,
                yy_state_t *yyesa, yy_state_t **yyes,
                YYPTRDIFF_T *yyes_capacity, yy_state_t *yyssp, int yytoken)
{
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat: reported tokens (one for the "unexpected",
     one per "expected"). */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Actual size of YYARG. */
  int yycount = 0;
  /* Cumulated lengths of YYARG.  */
  YYPTRDIFF_T yysize = 0;

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
      int yyn = yypact[+*yyssp];
      YYPTRDIFF_T yysize0 = yytnamerr (YY_NULLPTR, yytname[yytoken]);
      yysize = yysize0;
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
                  YYPTRDIFF_T yysize1
                    = yysize + yytnamerr (YY_NULLPTR, yytname[yyx]);
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
    /* Don't count the "%s"s in the final size, but reserve room for
       the terminator.  */
    YYPTRDIFF_T yysize1 = yysize + (yystrlen (yyformat) - 2 * yycount) + 1;
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
          ++yyp;
          ++yyformat;
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
    case 74: /* generic_symlist  */
#line 242 "src/parse-gram.y"
            { symbol_list_free (((*yyvaluep).generic_symlist)); }
#line 1794 "src/parse-gram.c"
        break;

    case 75: /* generic_symlist_item  */
#line 242 "src/parse-gram.y"
            { symbol_list_free (((*yyvaluep).generic_symlist_item)); }
#line 1800 "src/parse-gram.c"
        break;

    case 77: /* nterm_decls  */
#line 242 "src/parse-gram.y"
            { symbol_list_free (((*yyvaluep).nterm_decls)); }
#line 1806 "src/parse-gram.c"
        break;

    case 78: /* token_decls  */
#line 242 "src/parse-gram.y"
            { symbol_list_free (((*yyvaluep).token_decls)); }
#line 1812 "src/parse-gram.c"
        break;

    case 79: /* token_decl.1  */
#line 242 "src/parse-gram.y"
            { symbol_list_free (((*yyvaluep).yytype_79)); }
#line 1818 "src/parse-gram.c"
        break;

    case 82: /* token_decls_for_prec  */
#line 242 "src/parse-gram.y"
            { symbol_list_free (((*yyvaluep).token_decls_for_prec)); }
#line 1824 "src/parse-gram.c"
        break;

    case 83: /* token_decl_for_prec.1  */
#line 242 "src/parse-gram.y"
            { symbol_list_free (((*yyvaluep).yytype_83)); }
#line 1830 "src/parse-gram.c"
        break;

    case 85: /* symbol_decls  */
#line 242 "src/parse-gram.y"
            { symbol_list_free (((*yyvaluep).symbol_decls)); }
#line 1836 "src/parse-gram.c"
        break;

    case 86: /* symbol_decl.1  */
#line 242 "src/parse-gram.y"
            { symbol_list_free (((*yyvaluep).yytype_86)); }
#line 1842 "src/parse-gram.c"
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

    /* The locations where the error started and ended.  */
    YYLTYPE yyerror_range[3];

    YYPTRDIFF_T yystacksize;

    yy_state_t yyesa[20];
    yy_state_t *yyes;
    YYPTRDIFF_T yyes_capacity;

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
  YYPTRDIFF_T yymsg_alloc = sizeof yymsgbuf;
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
  yyes_capacity = 20;
  if (YYMAXDEPTH < yyes_capacity)
    yyes_capacity = YYMAXDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */

/* User initialization code.  */
#line 136 "src/parse-gram.y"
{
  /* Bison's grammar can initial empty locations, hence a default
     location is needed. */
  boundary_set (&yylloc.start, grammar_file, 1, 1, 1);
  boundary_set (&yylloc.end, grammar_file, 1, 1, 1);
}

#line 1967 "src/parse-gram.c"

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
# undef YYSTACK_RELOCATE
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
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END
  *++yylsp = yylloc;

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
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
#line 309 "src/parse-gram.y"
    {
      muscle_code_grow (union_seen ? "post_prologue" : "pre_prologue",
                        translate_code ((yyvsp[0].PROLOGUE), (yylsp[0]), true), (yylsp[0]));
      code_scanner_last_string_free ();
    }
#line 2177 "src/parse-gram.c"
    break;

  case 7:
#line 315 "src/parse-gram.y"
    {
      muscle_percent_define_ensure ((yyvsp[0].PERCENT_FLAG), (yylsp[0]), true);
    }
#line 2185 "src/parse-gram.c"
    break;

  case 8:
#line 319 "src/parse-gram.y"
    {
      muscle_percent_define_insert ((yyvsp[-1].variable), (yyloc), (yyvsp[0].value).kind, (yyvsp[0].value).chars,
                                    MUSCLE_PERCENT_DEFINE_GRAMMAR_FILE);
    }
#line 2194 "src/parse-gram.c"
    break;

  case 9:
#line 323 "src/parse-gram.y"
                                   { defines_flag = true; }
#line 2200 "src/parse-gram.c"
    break;

  case 10:
#line 325 "src/parse-gram.y"
    {
      defines_flag = true;
      spec_header_file = xstrdup ((yyvsp[0].STRING));
    }
#line 2209 "src/parse-gram.c"
    break;

  case 11:
#line 329 "src/parse-gram.y"
                                   { handle_error_verbose (&(yyloc), (yyvsp[0].PERCENT_ERROR_VERBOSE)); }
#line 2215 "src/parse-gram.c"
    break;

  case 12:
#line 330 "src/parse-gram.y"
                                   { expected_sr_conflicts = (yyvsp[0].INT); }
#line 2221 "src/parse-gram.c"
    break;

  case 13:
#line 331 "src/parse-gram.y"
                                   { expected_rr_conflicts = (yyvsp[0].INT); }
#line 2227 "src/parse-gram.c"
    break;

  case 14:
#line 332 "src/parse-gram.y"
                                   { handle_file_prefix (&(yyloc), &(yylsp[-1]), (yyvsp[-1].PERCENT_FILE_PREFIX), (yyvsp[0].STRING)); }
#line 2233 "src/parse-gram.c"
    break;

  case 15:
#line 334 "src/parse-gram.y"
    {
      nondeterministic_parser = true;
      glr_parser = true;
    }
#line 2242 "src/parse-gram.c"
    break;

  case 16:
#line 339 "src/parse-gram.y"
    {
      muscle_code_grow ("initial_action", translate_code ((yyvsp[0].BRACED_CODE), (yylsp[0]), false), (yylsp[0]));
      code_scanner_last_string_free ();
    }
#line 2251 "src/parse-gram.c"
    break;

  case 17:
#line 343 "src/parse-gram.y"
                                { language_argmatch ((yyvsp[0].STRING), grammar_prio, (yylsp[-1])); }
#line 2257 "src/parse-gram.c"
    break;

  case 18:
#line 344 "src/parse-gram.y"
                                { handle_name_prefix (&(yyloc), (yyvsp[-1].PERCENT_NAME_PREFIX), (yyvsp[0].STRING)); }
#line 2263 "src/parse-gram.c"
    break;

  case 19:
#line 345 "src/parse-gram.y"
                                { no_lines_flag = true; }
#line 2269 "src/parse-gram.c"
    break;

  case 20:
#line 346 "src/parse-gram.y"
                                { nondeterministic_parser = true; }
#line 2275 "src/parse-gram.c"
    break;

  case 21:
#line 347 "src/parse-gram.y"
                                { spec_outfile = (yyvsp[0].STRING); }
#line 2281 "src/parse-gram.c"
    break;

  case 22:
#line 348 "src/parse-gram.y"
           { current_param = (yyvsp[0].PERCENT_PARAM); }
#line 2287 "src/parse-gram.c"
    break;

  case 23:
#line 348 "src/parse-gram.y"
                                          { current_param = param_none; }
#line 2293 "src/parse-gram.c"
    break;

  case 24:
#line 349 "src/parse-gram.y"
                                { handle_pure_parser (&(yyloc), (yyvsp[0].PERCENT_PURE_PARSER)); }
#line 2299 "src/parse-gram.c"
    break;

  case 25:
#line 350 "src/parse-gram.y"
                                { handle_require (&(yylsp[0]), (yyvsp[0].STRING)); }
#line 2305 "src/parse-gram.c"
    break;

  case 26:
#line 351 "src/parse-gram.y"
                                { handle_skeleton (&(yylsp[0]), (yyvsp[0].STRING)); }
#line 2311 "src/parse-gram.c"
    break;

  case 27:
#line 352 "src/parse-gram.y"
                                { token_table_flag = true; }
#line 2317 "src/parse-gram.c"
    break;

  case 28:
#line 353 "src/parse-gram.y"
                                { report_flag |= report_states; }
#line 2323 "src/parse-gram.c"
    break;

  case 29:
#line 354 "src/parse-gram.y"
                                { handle_yacc (&(yyloc)); }
#line 2329 "src/parse-gram.c"
    break;

  case 30:
#line 355 "src/parse-gram.y"
                                { current_class = unknown_sym; yyerrok; }
#line 2335 "src/parse-gram.c"
    break;

  case 32:
#line 360 "src/parse-gram.y"
                   { add_param (current_param, (yyvsp[0].BRACED_CODE), (yylsp[0])); }
#line 2341 "src/parse-gram.c"
    break;

  case 33:
#line 361 "src/parse-gram.y"
                   { add_param (current_param, (yyvsp[0].BRACED_CODE), (yylsp[0])); }
#line 2347 "src/parse-gram.c"
    break;

  case 35:
#line 372 "src/parse-gram.y"
    {
      grammar_start_symbol_set ((yyvsp[0].symbol), (yylsp[0]));
    }
#line 2355 "src/parse-gram.c"
    break;

  case 36:
#line 376 "src/parse-gram.y"
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
#line 2370 "src/parse-gram.c"
    break;

  case 37:
#line 387 "src/parse-gram.y"
    {
      default_prec = true;
    }
#line 2378 "src/parse-gram.c"
    break;

  case 38:
#line 391 "src/parse-gram.y"
    {
      default_prec = false;
    }
#line 2386 "src/parse-gram.c"
    break;

  case 39:
#line 395 "src/parse-gram.y"
    {
      /* Do not invoke muscle_percent_code_grow here since it invokes
         muscle_user_name_list_grow.  */
      muscle_code_grow ("percent_code()",
                        translate_code_braceless ((yyvsp[0].BRACED_CODE), (yylsp[0])), (yylsp[0]));
      code_scanner_last_string_free ();
    }
#line 2398 "src/parse-gram.c"
    break;

  case 40:
#line 403 "src/parse-gram.y"
    {
      muscle_percent_code_grow ((yyvsp[-1].ID), (yylsp[-1]), translate_code_braceless ((yyvsp[0].BRACED_CODE), (yylsp[0])), (yylsp[0]));
      code_scanner_last_string_free ();
    }
#line 2407 "src/parse-gram.c"
    break;

  case 41:
#line 412 "src/parse-gram.y"
                 { (yyval.code_props_type) = destructor; }
#line 2413 "src/parse-gram.c"
    break;

  case 42:
#line 413 "src/parse-gram.y"
                 { (yyval.code_props_type) = printer; }
#line 2419 "src/parse-gram.c"
    break;

  case 43:
#line 423 "src/parse-gram.y"
         {}
#line 2425 "src/parse-gram.c"
    break;

  case 44:
#line 424 "src/parse-gram.y"
         { muscle_percent_define_insert ("api.value.union.name",
                                         (yylsp[0]), muscle_keyword, (yyvsp[0].ID),
                                         MUSCLE_PERCENT_DEFINE_GRAMMAR_FILE); }
#line 2433 "src/parse-gram.c"
    break;

  case 45:
#line 431 "src/parse-gram.y"
    {
      union_seen = true;
      muscle_code_grow ("union_members", translate_code_braceless ((yyvsp[0].BRACED_CODE), (yylsp[0])), (yylsp[0]));
      code_scanner_last_string_free ();
    }
#line 2443 "src/parse-gram.c"
    break;

  case 46:
#line 443 "src/parse-gram.y"
           { current_class = nterm_sym; }
#line 2449 "src/parse-gram.c"
    break;

  case 47:
#line 444 "src/parse-gram.y"
    {
      current_class = unknown_sym;
      symbol_list_free ((yyvsp[0].nterm_decls));
    }
#line 2458 "src/parse-gram.c"
    break;

  case 48:
#line 448 "src/parse-gram.y"
           { current_class = token_sym; }
#line 2464 "src/parse-gram.c"
    break;

  case 49:
#line 449 "src/parse-gram.y"
    {
      current_class = unknown_sym;
      symbol_list_free ((yyvsp[0].token_decls));
    }
#line 2473 "src/parse-gram.c"
    break;

  case 50:
#line 454 "src/parse-gram.y"
    {
      symbol_list_free ((yyvsp[0].symbol_decls));
    }
#line 2481 "src/parse-gram.c"
    break;

  case 51:
#line 458 "src/parse-gram.y"
    {
      ++current_prec;
      for (symbol_list *list = (yyvsp[0].token_decls_for_prec); list; list = list->next)
        symbol_precedence_set (list->content.sym, current_prec, (yyvsp[-1].precedence_declarator), (yylsp[-1]));
      symbol_list_free ((yyvsp[0].token_decls_for_prec));
    }
#line 2492 "src/parse-gram.c"
    break;

  case 52:
#line 467 "src/parse-gram.y"
                { (yyval.precedence_declarator) = left_assoc; }
#line 2498 "src/parse-gram.c"
    break;

  case 53:
#line 468 "src/parse-gram.y"
                { (yyval.precedence_declarator) = right_assoc; }
#line 2504 "src/parse-gram.c"
    break;

  case 54:
#line 469 "src/parse-gram.y"
                { (yyval.precedence_declarator) = non_assoc; }
#line 2510 "src/parse-gram.c"
    break;

  case 55:
#line 470 "src/parse-gram.y"
                { (yyval.precedence_declarator) = precedence_assoc; }
#line 2516 "src/parse-gram.c"
    break;

  case 56:
#line 474 "src/parse-gram.y"
         { (yyval.yytype_73) = NULL; }
#line 2522 "src/parse-gram.c"
    break;

  case 57:
#line 475 "src/parse-gram.y"
         { (yyval.yytype_73) = (yyvsp[0].TAG); }
#line 2528 "src/parse-gram.c"
    break;

  case 59:
#line 481 "src/parse-gram.y"
                                         { (yyval.generic_symlist) = symbol_list_append ((yyvsp[-1].generic_symlist), (yyvsp[0].generic_symlist_item)); }
#line 2534 "src/parse-gram.c"
    break;

  case 60:
#line 485 "src/parse-gram.y"
            { (yyval.generic_symlist_item) = symbol_list_sym_new ((yyvsp[0].symbol), (yylsp[0])); }
#line 2540 "src/parse-gram.c"
    break;

  case 61:
#line 486 "src/parse-gram.y"
            { (yyval.generic_symlist_item) = symbol_list_type_new ((yyvsp[0].tag), (yylsp[0])); }
#line 2546 "src/parse-gram.c"
    break;

  case 63:
#line 491 "src/parse-gram.y"
        { (yyval.tag) = uniqstr_new ("*"); }
#line 2552 "src/parse-gram.c"
    break;

  case 64:
#line 492 "src/parse-gram.y"
        { (yyval.tag) = uniqstr_new (""); }
#line 2558 "src/parse-gram.c"
    break;

  case 66:
#line 515 "src/parse-gram.y"
    {
      (yyval.token_decls) = (yyvsp[0].yytype_79);
    }
#line 2566 "src/parse-gram.c"
    break;

  case 67:
#line 519 "src/parse-gram.y"
    {
      (yyval.token_decls) = symbol_list_type_set ((yyvsp[0].yytype_79), (yyvsp[-1].TAG), (yylsp[-1]));
    }
#line 2574 "src/parse-gram.c"
    break;

  case 68:
#line 523 "src/parse-gram.y"
    {
      (yyval.token_decls) = symbol_list_append ((yyvsp[-2].token_decls), symbol_list_type_set ((yyvsp[0].yytype_79), (yyvsp[-1].TAG), (yylsp[-1])));
    }
#line 2582 "src/parse-gram.c"
    break;

  case 69:
#line 530 "src/parse-gram.y"
                            { (yyval.yytype_79) = symbol_list_sym_new ((yyvsp[0].token_decl), (yylsp[0])); }
#line 2588 "src/parse-gram.c"
    break;

  case 70:
#line 531 "src/parse-gram.y"
                            { (yyval.yytype_79) = symbol_list_append ((yyvsp[-1].yytype_79), symbol_list_sym_new ((yyvsp[0].token_decl), (yylsp[0]))); }
#line 2594 "src/parse-gram.c"
    break;

  case 71:
#line 536 "src/parse-gram.y"
    {
      (yyval.token_decl) = (yyvsp[-2].id);
      symbol_class_set ((yyvsp[-2].id), current_class, (yylsp[-2]), true);
      if (0 <= (yyvsp[-1].yytype_81))
        symbol_user_token_number_set ((yyvsp[-2].id), (yyvsp[-1].yytype_81), (yylsp[-1]));
      if ((yyvsp[0].yytype_100))
        symbol_make_alias ((yyvsp[-2].id), (yyvsp[0].yytype_100), (yylsp[0]));
    }
#line 2607 "src/parse-gram.c"
    break;

  case 72:
#line 548 "src/parse-gram.y"
          { (yyval.yytype_81) = -1; }
#line 2613 "src/parse-gram.c"
    break;

  case 74:
#line 562 "src/parse-gram.y"
    {
      (yyval.token_decls_for_prec) = (yyvsp[0].yytype_83);
    }
#line 2621 "src/parse-gram.c"
    break;

  case 75:
#line 566 "src/parse-gram.y"
    {
      (yyval.token_decls_for_prec) = symbol_list_type_set ((yyvsp[0].yytype_83), (yyvsp[-1].TAG), (yylsp[-1]));
    }
#line 2629 "src/parse-gram.c"
    break;

  case 76:
#line 570 "src/parse-gram.y"
    {
      (yyval.token_decls_for_prec) = symbol_list_append ((yyvsp[-2].token_decls_for_prec), symbol_list_type_set ((yyvsp[0].yytype_83), (yyvsp[-1].TAG), (yylsp[-1])));
    }
#line 2637 "src/parse-gram.c"
    break;

  case 77:
#line 578 "src/parse-gram.y"
    { (yyval.yytype_83) = symbol_list_sym_new ((yyvsp[0].token_decl_for_prec), (yylsp[0])); }
#line 2643 "src/parse-gram.c"
    break;

  case 78:
#line 580 "src/parse-gram.y"
    { (yyval.yytype_83) = symbol_list_append ((yyvsp[-1].yytype_83), symbol_list_sym_new ((yyvsp[0].token_decl_for_prec), (yylsp[0]))); }
#line 2649 "src/parse-gram.c"
    break;

  case 79:
#line 585 "src/parse-gram.y"
    {
      (yyval.token_decl_for_prec) = (yyvsp[-1].id);
      symbol_class_set ((yyvsp[-1].id), token_sym, (yylsp[-1]), false);
      if (0 <= (yyvsp[0].yytype_81))
        symbol_user_token_number_set ((yyvsp[-1].id), (yyvsp[0].yytype_81), (yylsp[0]));
    }
#line 2660 "src/parse-gram.c"
    break;

  case 81:
#line 602 "src/parse-gram.y"
    {
      (yyval.symbol_decls) = (yyvsp[0].yytype_86);
    }
#line 2668 "src/parse-gram.c"
    break;

  case 82:
#line 606 "src/parse-gram.y"
    {
      (yyval.symbol_decls) = symbol_list_type_set ((yyvsp[0].yytype_86), (yyvsp[-1].TAG), (yylsp[-1]));
    }
#line 2676 "src/parse-gram.c"
    break;

  case 83:
#line 610 "src/parse-gram.y"
    {
      (yyval.symbol_decls) = symbol_list_append ((yyvsp[-2].symbol_decls), symbol_list_type_set ((yyvsp[0].yytype_86), (yyvsp[-1].TAG), (yylsp[-1])));
    }
#line 2684 "src/parse-gram.c"
    break;

  case 84:
#line 618 "src/parse-gram.y"
    {
      symbol_class_set ((yyvsp[0].symbol), pct_type_sym, (yylsp[0]), false);
      (yyval.yytype_86) = symbol_list_sym_new ((yyvsp[0].symbol), (yylsp[0]));
    }
#line 2693 "src/parse-gram.c"
    break;

  case 85:
#line 623 "src/parse-gram.y"
    {
      symbol_class_set ((yyvsp[0].symbol), pct_type_sym, (yylsp[0]), false);
      (yyval.yytype_86) = symbol_list_append ((yyvsp[-1].yytype_86), symbol_list_sym_new ((yyvsp[0].symbol), (yylsp[0])));
    }
#line 2702 "src/parse-gram.c"
    break;

  case 90:
#line 644 "src/parse-gram.y"
    {
      yyerrok;
    }
#line 2710 "src/parse-gram.c"
    break;

  case 91:
#line 650 "src/parse-gram.y"
                         { current_lhs ((yyvsp[-1].id_colon), (yylsp[-1]), (yyvsp[0].yytype_93)); }
#line 2716 "src/parse-gram.c"
    break;

  case 92:
#line 651 "src/parse-gram.y"
    {
      /* Free the current lhs. */
      current_lhs (0, (yylsp[-4]), 0);
    }
#line 2725 "src/parse-gram.c"
    break;

  case 93:
#line 658 "src/parse-gram.y"
                     { grammar_current_rule_end ((yylsp[0])); }
#line 2731 "src/parse-gram.c"
    break;

  case 94:
#line 659 "src/parse-gram.y"
                     { grammar_current_rule_end ((yylsp[0])); }
#line 2737 "src/parse-gram.c"
    break;

  case 96:
#line 666 "src/parse-gram.y"
    { grammar_current_rule_begin (current_lhs_symbol, current_lhs_loc,
                                  current_lhs_named_ref); }
#line 2744 "src/parse-gram.c"
    break;

  case 97:
#line 669 "src/parse-gram.y"
    { grammar_current_rule_symbol_append ((yyvsp[-1].symbol), (yylsp[-1]), (yyvsp[0].yytype_93)); }
#line 2750 "src/parse-gram.c"
    break;

  case 98:
#line 671 "src/parse-gram.y"
    { grammar_current_rule_action_append ((yyvsp[-1].BRACED_CODE), (yylsp[-1]), (yyvsp[0].yytype_93), (yyvsp[-2].yytype_73)); }
#line 2756 "src/parse-gram.c"
    break;

  case 99:
#line 673 "src/parse-gram.y"
    { grammar_current_rule_predicate_append ((yyvsp[0].BRACED_PREDICATE), (yylsp[0])); }
#line 2762 "src/parse-gram.c"
    break;

  case 100:
#line 675 "src/parse-gram.y"
    { grammar_current_rule_empty_set ((yylsp[0])); }
#line 2768 "src/parse-gram.c"
    break;

  case 101:
#line 677 "src/parse-gram.y"
    { grammar_current_rule_prec_set ((yyvsp[0].symbol), (yylsp[0])); }
#line 2774 "src/parse-gram.c"
    break;

  case 102:
#line 679 "src/parse-gram.y"
    { grammar_current_rule_dprec_set ((yyvsp[0].INT), (yylsp[0])); }
#line 2780 "src/parse-gram.c"
    break;

  case 103:
#line 681 "src/parse-gram.y"
    { grammar_current_rule_merge_set ((yyvsp[0].TAG), (yylsp[0])); }
#line 2786 "src/parse-gram.c"
    break;

  case 104:
#line 683 "src/parse-gram.y"
    { grammar_current_rule_expect_sr ((yyvsp[0].INT), (yylsp[0])); }
#line 2792 "src/parse-gram.c"
    break;

  case 105:
#line 685 "src/parse-gram.y"
    { grammar_current_rule_expect_rr ((yyvsp[0].INT), (yylsp[0])); }
#line 2798 "src/parse-gram.c"
    break;

  case 106:
#line 689 "src/parse-gram.y"
                 { (yyval.yytype_93) = NULL; }
#line 2804 "src/parse-gram.c"
    break;

  case 107:
#line 690 "src/parse-gram.y"
                 { (yyval.yytype_93) = named_ref_new ((yyvsp[0].BRACKETED_ID), (yylsp[0])); }
#line 2810 "src/parse-gram.c"
    break;

  case 109:
#line 723 "src/parse-gram.y"
          { (yyval.value).kind = muscle_keyword; (yyval.value).chars = ""; }
#line 2816 "src/parse-gram.c"
    break;

  case 110:
#line 724 "src/parse-gram.y"
          { (yyval.value).kind = muscle_keyword; (yyval.value).chars = (yyvsp[0].ID); }
#line 2822 "src/parse-gram.c"
    break;

  case 111:
#line 725 "src/parse-gram.y"
          { (yyval.value).kind = muscle_string;  (yyval.value).chars = (yyvsp[0].STRING); }
#line 2828 "src/parse-gram.c"
    break;

  case 112:
#line 726 "src/parse-gram.y"
          { (yyval.value).kind = muscle_code;    (yyval.value).chars = strip_braces ((yyvsp[0].BRACED_CODE)); }
#line 2834 "src/parse-gram.c"
    break;

  case 113:
#line 739 "src/parse-gram.y"
    { (yyval.id) = symbol_from_uniqstr ((yyvsp[0].ID), (yylsp[0])); }
#line 2840 "src/parse-gram.c"
    break;

  case 114:
#line 741 "src/parse-gram.y"
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
          int indent = 0;
          complain_indent (&(yylsp[0]), complaint, &indent,
                           _("character literals cannot be used together"
                             " with %s"), var);
          indent += SUB_INDENT;
          location loc = muscle_percent_define_get_loc (var);
          complain_indent (&loc, complaint, &indent,
                           _("definition of %s"), var);
        }
      (yyval.id) = symbol_get (char_name ((yyvsp[0].CHAR)), (yylsp[0]));
      symbol_class_set ((yyval.id), token_sym, (yylsp[0]), false);
      symbol_user_token_number_set ((yyval.id), (yyvsp[0].CHAR), (yylsp[0]));
    }
#line 2868 "src/parse-gram.c"
    break;

  case 115:
#line 767 "src/parse-gram.y"
           { (yyval.id_colon) = symbol_from_uniqstr ((yyvsp[0].ID_COLON), (yylsp[0])); }
#line 2874 "src/parse-gram.c"
    break;

  case 118:
#line 779 "src/parse-gram.y"
    {
      (yyval.string_as_id) = symbol_get (quotearg_style (c_quoting_style, (yyvsp[0].STRING)), (yylsp[0]));
      symbol_class_set ((yyval.string_as_id), token_sym, (yylsp[0]), false);
    }
#line 2883 "src/parse-gram.c"
    break;

  case 119:
#line 787 "src/parse-gram.y"
                     { (yyval.yytype_100) = NULL; }
#line 2889 "src/parse-gram.c"
    break;

  case 122:
#line 794 "src/parse-gram.y"
    {
      muscle_code_grow ("epilogue", translate_code ((yyvsp[0].EPILOGUE), (yylsp[0]), true), (yylsp[0]));
      code_scanner_last_string_free ();
    }
#line 2898 "src/parse-gram.c"
    break;


#line 2902 "src/parse-gram.c"

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
            yymsg = YY_CAST (char *, YYSTACK_ALLOC (YY_CAST (YYSIZE_T, yymsg_alloc)));
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
                  yystos[+*yyssp], yyvsp, yylsp);
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
#line 800 "src/parse-gram.y"


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

  /* Pretend to be at least 3.5, to check features published in that
     version while developping it.  */
  const char* api_version = "3.5";
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
