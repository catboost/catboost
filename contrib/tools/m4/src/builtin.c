/* GNU m4 -- A simple macro processor

   Copyright (C) 1989-1994, 2000, 2004, 2006-2013 Free Software
   Foundation, Inc.

   This file is part of GNU M4.

   GNU M4 is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   GNU M4 is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/* Code for all builtin macros, initialization of symbol table, and
   expansion of user defined macros.  */

#include "m4.h"

#include "execute.h"
#include "memchr2.h"
#include "progname.h"
#include "regex.h"
#include "spawn-pipe.h"
#include "wait-process.h"

#define ARG(i) (argc > (i) ? TOKEN_DATA_TEXT (argv[i]) : "")

/* Initialization of builtin and predefined macros.  The table
   "builtin_tab" is both used for initialization, and by the "builtin"
   builtin.  */

#define DECLARE(name) \
  static void name (struct obstack *, int, token_data **)

DECLARE (m4___file__);
DECLARE (m4___line__);
DECLARE (m4___program__);
DECLARE (m4_builtin);
DECLARE (m4_changecom);
DECLARE (m4_changequote);
#ifdef ENABLE_CHANGEWORD
DECLARE (m4_changeword);
#endif
DECLARE (m4_debugmode);
DECLARE (m4_debugfile);
DECLARE (m4_decr);
DECLARE (m4_define);
DECLARE (m4_defn);
DECLARE (m4_divert);
DECLARE (m4_divnum);
DECLARE (m4_dnl);
DECLARE (m4_dumpdef);
DECLARE (m4_errprint);
DECLARE (m4_esyscmd);
DECLARE (m4_eval);
DECLARE (m4_format);
DECLARE (m4_ifdef);
DECLARE (m4_ifelse);
DECLARE (m4_include);
DECLARE (m4_incr);
DECLARE (m4_index);
DECLARE (m4_indir);
DECLARE (m4_len);
DECLARE (m4_m4exit);
DECLARE (m4_m4wrap);
DECLARE (m4_maketemp);
DECLARE (m4_mkstemp);
DECLARE (m4_patsubst);
DECLARE (m4_popdef);
DECLARE (m4_pushdef);
DECLARE (m4_regexp);
DECLARE (m4_shift);
DECLARE (m4_sinclude);
DECLARE (m4_substr);
DECLARE (m4_syscmd);
DECLARE (m4_sysval);
DECLARE (m4_traceoff);
DECLARE (m4_traceon);
DECLARE (m4_translit);
DECLARE (m4_undefine);
DECLARE (m4_undivert);

#undef DECLARE

static builtin const builtin_tab[] =
{

  /* name               GNUext  macros  blind   function */

  { "__file__",         true,   false,  false,  m4___file__ },
  { "__line__",         true,   false,  false,  m4___line__ },
  { "__program__",      true,   false,  false,  m4___program__ },
  { "builtin",          true,   true,   true,   m4_builtin },
  { "changecom",        false,  false,  false,  m4_changecom },
  { "changequote",      false,  false,  false,  m4_changequote },
#ifdef ENABLE_CHANGEWORD
  { "changeword",       true,   false,  true,   m4_changeword },
#endif
  { "debugmode",        true,   false,  false,  m4_debugmode },
  { "debugfile",        true,   false,  false,  m4_debugfile },
  { "decr",             false,  false,  true,   m4_decr },
  { "define",           false,  true,   true,   m4_define },
  { "defn",             false,  false,  true,   m4_defn },
  { "divert",           false,  false,  false,  m4_divert },
  { "divnum",           false,  false,  false,  m4_divnum },
  { "dnl",              false,  false,  false,  m4_dnl },
  { "dumpdef",          false,  false,  false,  m4_dumpdef },
  { "errprint",         false,  false,  true,   m4_errprint },
  { "esyscmd",          true,   false,  true,   m4_esyscmd },
  { "eval",             false,  false,  true,   m4_eval },
  { "format",           true,   false,  true,   m4_format },
  { "ifdef",            false,  false,  true,   m4_ifdef },
  { "ifelse",           false,  false,  true,   m4_ifelse },
  { "include",          false,  false,  true,   m4_include },
  { "incr",             false,  false,  true,   m4_incr },
  { "index",            false,  false,  true,   m4_index },
  { "indir",            true,   true,   true,   m4_indir },
  { "len",              false,  false,  true,   m4_len },
  { "m4exit",           false,  false,  false,  m4_m4exit },
  { "m4wrap",           false,  false,  true,   m4_m4wrap },
  { "maketemp",         false,  false,  true,   m4_maketemp },
  { "mkstemp",          false,  false,  true,   m4_mkstemp },
  { "patsubst",         true,   false,  true,   m4_patsubst },
  { "popdef",           false,  false,  true,   m4_popdef },
  { "pushdef",          false,  true,   true,   m4_pushdef },
  { "regexp",           true,   false,  true,   m4_regexp },
  { "shift",            false,  false,  true,   m4_shift },
  { "sinclude",         false,  false,  true,   m4_sinclude },
  { "substr",           false,  false,  true,   m4_substr },
  { "syscmd",           false,  false,  true,   m4_syscmd },
  { "sysval",           false,  false,  false,  m4_sysval },
  { "traceoff",         false,  false,  false,  m4_traceoff },
  { "traceon",          false,  false,  false,  m4_traceon },
  { "translit",         false,  false,  true,   m4_translit },
  { "undefine",         false,  false,  true,   m4_undefine },
  { "undivert",         false,  false,  false,  m4_undivert },

  { 0,                  false,  false,  false,  0 },

  /* placeholder is intentionally stuck after the table end delimiter,
     so that we can easily find it, while not treating it as a real
     builtin.  */
  { "placeholder",      true,   false,  false,  m4_placeholder },
};

static predefined const predefined_tab[] =
{
#if UNIX
  { "unix",     "__unix__",     "" },
#endif
#if W32_NATIVE
  { "windows",  "__windows__",  "" },
#endif
#if OS2
  { "os2",      "__os2__",      "" },
#endif
#if !UNIX && !W32_NATIVE && !OS2
# warning Platform macro not provided
#endif
  { NULL,       "__gnu__",      "" },

  { NULL,       NULL,           NULL },
};

/*----------------------------------------.
| Find the builtin, which lives on ADDR.  |
`----------------------------------------*/

const builtin * M4_GNUC_PURE
find_builtin_by_addr (builtin_func *func)
{
  const builtin *bp;

  for (bp = &builtin_tab[0]; bp->name != NULL; bp++)
    if (bp->func == func)
      return bp;
  if (func == m4_placeholder)
    return bp + 1;
  return NULL;
}

/*----------------------------------------------------------.
| Find the builtin, which has NAME.  On failure, return the |
| placeholder builtin.                                      |
`----------------------------------------------------------*/

const builtin * M4_GNUC_PURE
find_builtin_by_name (const char *name)
{
  const builtin *bp;

  for (bp = &builtin_tab[0]; bp->name != NULL; bp++)
    if (STREQ (bp->name, name))
      return bp;
  return bp + 1;
}

/*----------------------------------------------------------------.
| Install a builtin macro with name NAME, bound to the C function |
| given in BP.  MODE is SYMBOL_INSERT or SYMBOL_PUSHDEF.          |
`----------------------------------------------------------------*/

void
define_builtin (const char *name, const builtin *bp, symbol_lookup mode)
{
  symbol *sym;

  sym = lookup_symbol (name, mode);
  SYMBOL_TYPE (sym) = TOKEN_FUNC;
  SYMBOL_MACRO_ARGS (sym) = bp->groks_macro_args;
  SYMBOL_BLIND_NO_ARGS (sym) = bp->blind_if_no_args;
  SYMBOL_FUNC (sym) = bp->func;
}

/* Storage for the compiled regular expression of
   --warn-macro-sequence.  */
static struct re_pattern_buffer macro_sequence_buf;

/* Storage for the matches of --warn-macro-sequence.  */
static struct re_registers macro_sequence_regs;

/* True if --warn-macro-sequence is in effect.  */
static bool macro_sequence_inuse;

/*----------------------------------------.
| Clean up regular expression variables.  |
`----------------------------------------*/

static void
free_pattern_buffer (struct re_pattern_buffer *buf, struct re_registers *regs)
{
  regfree (buf);
  free (regs->start);
  free (regs->end);
}

/*-----------------------------------------------------------------.
| Set the regular expression of --warn-macro-sequence that will be |
| checked during define and pushdef.  Exit on failure.             |
`-----------------------------------------------------------------*/
void
set_macro_sequence (const char *regexp)
{
  const char *msg;

  if (! regexp)
    regexp = DEFAULT_MACRO_SEQUENCE;
  else if (regexp[0] == '\0')
    {
      macro_sequence_inuse = false;
      return;
    }

  msg = re_compile_pattern (regexp, strlen (regexp), &macro_sequence_buf);
  if (msg != NULL)
    {
      M4ERROR ((EXIT_FAILURE, 0,
                "--warn-macro-sequence: bad regular expression `%s': %s",
                regexp, msg));
    }
  re_set_registers (&macro_sequence_buf, &macro_sequence_regs,
                    macro_sequence_regs.num_regs,
                    macro_sequence_regs.start, macro_sequence_regs.end);
  macro_sequence_inuse = true;
}

/*-----------------------------------------------------------.
| Free dynamic memory utilized by the macro sequence regular |
| expression during the define builtin.                      |
`-----------------------------------------------------------*/
void
free_macro_sequence (void)
{
  free_pattern_buffer (&macro_sequence_buf, &macro_sequence_regs);
}

/*-----------------------------------------------------------------.
| Define a predefined or user-defined macro, with name NAME, and   |
| expansion TEXT.  MODE destinguishes between the "define" and the |
| "pushdef" case.  It is also used from main.                      |
`-----------------------------------------------------------------*/

void
define_user_macro (const char *name, const char *text, symbol_lookup mode)
{
  symbol *s;
  char *defn = xstrdup (text ? text : "");

  s = lookup_symbol (name, mode);
  if (SYMBOL_TYPE (s) == TOKEN_TEXT)
    free (SYMBOL_TEXT (s));

  SYMBOL_TYPE (s) = TOKEN_TEXT;
  SYMBOL_TEXT (s) = defn;

  /* Implement --warn-macro-sequence.  */
  if (macro_sequence_inuse && text)
    {
      regoff_t offset = 0;
      size_t len = strlen (defn);

      while ((offset = re_search (&macro_sequence_buf, defn, len, offset,
                                  len - offset, &macro_sequence_regs)) >= 0)
        {
          /* Skip empty matches.  */
          if (macro_sequence_regs.start[0] == macro_sequence_regs.end[0])
            offset++;
          else
            {
              char tmp;
              offset = macro_sequence_regs.end[0];
              tmp = defn[offset];
              defn[offset] = '\0';
              M4ERROR ((warning_status, 0,
                        "Warning: definition of `%s' contains sequence `%s'",
                        name, defn + macro_sequence_regs.start[0]));
              defn[offset] = tmp;
            }
        }
      if (offset == -2)
        M4ERROR ((warning_status, 0,
                  "error checking --warn-macro-sequence for macro `%s'",
                  name));
    }
}

/*-----------------------------------------------.
| Initialize all builtin and predefined macros.  |
`-----------------------------------------------*/

void
builtin_init (void)
{
  const builtin *bp;
  const predefined *pp;
  char *string;

  for (bp = &builtin_tab[0]; bp->name != NULL; bp++)
    if (!no_gnu_extensions || !bp->gnu_extension)
      {
        if (prefix_all_builtins)
          {
            string = (char *) xmalloc (strlen (bp->name) + 4);
            strcpy (string, "m4_");
            strcat (string, bp->name);
            define_builtin (string, bp, SYMBOL_INSERT);
            free (string);
          }
        else
          define_builtin (bp->name, bp, SYMBOL_INSERT);
      }

  for (pp = &predefined_tab[0]; pp->func != NULL; pp++)
    if (no_gnu_extensions)
      {
        if (pp->unix_name != NULL)
          define_user_macro (pp->unix_name, pp->func, SYMBOL_INSERT);
      }
    else
      {
        if (pp->gnu_name != NULL)
          define_user_macro (pp->gnu_name, pp->func, SYMBOL_INSERT);
      }
}

/*-------------------------------------------------------------------.
| Give friendly warnings if a builtin macro is passed an             |
| inappropriate number of arguments.  NAME is the macro name for     |
| messages, ARGC is actual number of arguments, MIN is the minimum   |
| number of acceptable arguments, negative if not applicable, MAX is |
| the maximum number, negative if not applicable.                    |
`-------------------------------------------------------------------*/

static bool
bad_argc (token_data *name, int argc, int min, int max)
{
  bool isbad = false;

  if (min > 0 && argc < min)
    {
      if (!suppress_warnings)
        M4ERROR ((warning_status, 0,
                  "Warning: too few arguments to builtin `%s'",
                  TOKEN_DATA_TEXT (name)));
      isbad = true;
    }
  else if (max > 0 && argc > max && !suppress_warnings)
    M4ERROR ((warning_status, 0,
              "Warning: excess arguments to builtin `%s' ignored",
              TOKEN_DATA_TEXT (name)));

  return isbad;
}

/*-----------------------------------------------------------------.
| The function numeric_arg () converts ARG to an int pointed to by |
| VALUEP.  If the conversion fails, print error message for macro  |
| MACRO.  Return true iff conversion succeeds.                     |
`-----------------------------------------------------------------*/

static bool
numeric_arg (token_data *macro, const char *arg, int *valuep)
{
  char *endp;

  if (*arg == '\0')
    {
      *valuep = 0;
      M4ERROR ((warning_status, 0,
                "empty string treated as 0 in builtin `%s'",
                TOKEN_DATA_TEXT (macro)));
    }
  else
    {
      errno = 0;
      *valuep = strtol (arg, &endp, 10);
      if (*endp != '\0')
        {
          M4ERROR ((warning_status, 0,
                    "non-numeric argument to builtin `%s'",
                    TOKEN_DATA_TEXT (macro)));
          return false;
        }
      if (isspace (to_uchar (*arg)))
        M4ERROR ((warning_status, 0,
                  "leading whitespace ignored in builtin `%s'",
                  TOKEN_DATA_TEXT (macro)));
      else if (errno == ERANGE)
        M4ERROR ((warning_status, 0,
                  "numeric overflow detected in builtin `%s'",
                  TOKEN_DATA_TEXT (macro)));
    }
  return true;
}

/*------------------------------------------------------.
| The function ntoa () converts VALUE to a signed ASCII |
| representation in radix RADIX.                        |
`------------------------------------------------------*/

/* Digits for number to ASCII conversions.  */
static char const digits[] = "0123456789abcdefghijklmnopqrstuvwxyz";

const char *
ntoa (int32_t value, int radix)
{
  bool negative;
  uint32_t uvalue;
  static char str[256];
  char *s = &str[sizeof str];

  *--s = '\0';

  if (value < 0)
    {
      negative = true;
      uvalue = -(uint32_t) value;
    }
  else
    {
      negative = false;
      uvalue = (uint32_t) value;
    }

  do
    {
      *--s = digits[uvalue % radix];
      uvalue /= radix;
    }
  while (uvalue > 0);

  if (negative)
    *--s = '-';
  return s;
}

/*---------------------------------------------------------------.
| Format an int VAL, and stuff it into an obstack OBS.  Used for |
| macros expanding to numbers.                                   |
`---------------------------------------------------------------*/

static void
shipout_int (struct obstack *obs, int val)
{
  const char *s;

  s = ntoa ((int32_t) val, 10);
  obstack_grow (obs, s, strlen (s));
}

/*-------------------------------------------------------------------.
| Print ARGC arguments from the table ARGV to obstack OBS, separated |
| by SEP, and quoted by the current quotes if QUOTED is true.        |
`-------------------------------------------------------------------*/

static void
dump_args (struct obstack *obs, int argc, token_data **argv,
           const char *sep, bool quoted)
{
  int i;
  size_t len = strlen (sep);

  for (i = 1; i < argc; i++)
    {
      if (i > 1)
        obstack_grow (obs, sep, len);
      if (quoted)
        obstack_grow (obs, lquote.string, lquote.length);
      obstack_grow (obs, TOKEN_DATA_TEXT (argv[i]),
                    strlen (TOKEN_DATA_TEXT (argv[i])));
      if (quoted)
        obstack_grow (obs, rquote.string, rquote.length);
    }
}

/* The rest of this file is code for builtins and expansion of user
   defined macros.  All the functions for builtins have a prototype as:

        void m4_MACRONAME (struct obstack *obs, int argc, char *argv[]);

   The function are expected to leave their expansion on the obstack OBS,
   as an unfinished object.  ARGV is a table of ARGC pointers to the
   individual arguments to the macro.  Please note that in general
   argv[argc] != NULL.  */

/* The first section are macros for definining, undefining, examining,
   changing, ... other macros.  */

/*-------------------------------------------------------------------.
| The function define_macro is common for the builtins "define",     |
| "undefine", "pushdef" and "popdef".  ARGC and ARGV is as for the   |
| caller, and MODE argument determines how the macro name is entered |
| into the symbol table.                                             |
`-------------------------------------------------------------------*/

static void
define_macro (int argc, token_data **argv, symbol_lookup mode)
{
  const builtin *bp;

  if (bad_argc (argv[0], argc, 2, 3))
    return;

  if (TOKEN_DATA_TYPE (argv[1]) != TOKEN_TEXT)
    {
      M4ERROR ((warning_status, 0,
                "Warning: %s: invalid macro name ignored", ARG (0)));
      return;
    }

  if (argc == 2)
    {
      define_user_macro (ARG (1), "", mode);
      return;
    }

  switch (TOKEN_DATA_TYPE (argv[2]))
    {
    case TOKEN_TEXT:
      define_user_macro (ARG (1), ARG (2), mode);
      break;

    case TOKEN_FUNC:
      bp = find_builtin_by_addr (TOKEN_DATA_FUNC (argv[2]));
      if (bp == NULL)
        return;
      else
        define_builtin (ARG (1), bp, mode);
      break;

    case TOKEN_VOID:
    default:
      M4ERROR ((warning_status, 0,
                "INTERNAL ERROR: bad token data type in define_macro ()"));
      abort ();
    }
}

static void
m4_define (struct obstack *obs M4_GNUC_UNUSED, int argc, token_data **argv)
{
  define_macro (argc, argv, SYMBOL_INSERT);
}

static void
m4_undefine (struct obstack *obs M4_GNUC_UNUSED, int argc, token_data **argv)
{
  int i;
  if (bad_argc (argv[0], argc, 2, -1))
    return;
  for (i = 1; i < argc; i++)
    lookup_symbol (ARG (i), SYMBOL_DELETE);
}

static void
m4_pushdef (struct obstack *obs M4_GNUC_UNUSED, int argc, token_data **argv)
{
  define_macro (argc, argv,  SYMBOL_PUSHDEF);
}

static void
m4_popdef (struct obstack *obs M4_GNUC_UNUSED, int argc, token_data **argv)
{
  int i;
  if (bad_argc (argv[0], argc, 2, -1))
    return;
  for (i = 1; i < argc; i++)
    lookup_symbol (ARG (i), SYMBOL_POPDEF);
}

/*---------------------.
| Conditionals of m4.  |
`---------------------*/

static void
m4_ifdef (struct obstack *obs, int argc, token_data **argv)
{
  symbol *s;
  const char *result;

  if (bad_argc (argv[0], argc, 3, 4))
    return;
  s = lookup_symbol (ARG (1), SYMBOL_LOOKUP);

  if (s != NULL && SYMBOL_TYPE (s) != TOKEN_VOID)
    result = ARG (2);
  else if (argc >= 4)
    result = ARG (3);
  else
    result = NULL;

  if (result != NULL)
    obstack_grow (obs, result, strlen (result));
}

static void
m4_ifelse (struct obstack *obs, int argc, token_data **argv)
{
  const char *result;
  token_data *me = argv[0];

  if (argc == 2)
    return;

  if (bad_argc (me, argc, 4, -1))
    return;
  else
    /* Diagnose excess arguments if 5, 8, 11, etc., actual arguments.  */
    bad_argc (me, (argc + 2) % 3, -1, 1);

  argv++;
  argc--;

  result = NULL;
  while (result == NULL)

    if (STREQ (ARG (0), ARG (1)))
      result = ARG (2);

    else
      switch (argc)
        {
        case 3:
          return;

        case 4:
        case 5:
          result = ARG (3);
          break;

        default:
          argc -= 3;
          argv += 3;
        }

  obstack_grow (obs, result, strlen (result));
}

/*-------------------------------------------------------------------.
| The function dump_symbol () is for use by "dumpdef".  It builds up |
| a table of all defined, un-shadowed, symbols.                      |
`-------------------------------------------------------------------*/

/* The structure dump_symbol_data is used to pass the information needed
   from call to call to dump_symbol.  */

struct dump_symbol_data
{
  struct obstack *obs;          /* obstack for table */
  symbol **base;                /* base of table */
  int size;                     /* size of table */
};

static void
dump_symbol (symbol *sym, void *arg)
{
  struct dump_symbol_data *data = (struct dump_symbol_data *) arg;
  if (!SYMBOL_SHADOWED (sym) && SYMBOL_TYPE (sym) != TOKEN_VOID)
    {
      obstack_blank (data->obs, sizeof (symbol *));
      data->base = (symbol **) obstack_base (data->obs);
      data->base[data->size++] = sym;
    }
}

/*------------------------------------------------------------------------.
| qsort comparison routine, for sorting the table made in m4_dumpdef ().  |
`------------------------------------------------------------------------*/

static int
dumpdef_cmp (const void *s1, const void *s2)
{
  return strcmp (SYMBOL_NAME (* (symbol *const *) s1),
                 SYMBOL_NAME (* (symbol *const *) s2));
}

/*-------------------------------------------------------------.
| Implementation of "dumpdef" itself.  It builds up a table of |
| pointers to symbols, sorts it and prints the sorted table.   |
`-------------------------------------------------------------*/

static void
m4_dumpdef (struct obstack *obs, int argc, token_data **argv)
{
  symbol *s;
  int i;
  struct dump_symbol_data data;
  const builtin *bp;

  data.obs = obs;
  data.base = (symbol **) obstack_base (obs);
  data.size = 0;

  if (argc == 1)
    {
      hack_all_symbols (dump_symbol, &data);
    }
  else
    {
      for (i = 1; i < argc; i++)
        {
          s = lookup_symbol (TOKEN_DATA_TEXT (argv[i]), SYMBOL_LOOKUP);
          if (s != NULL && SYMBOL_TYPE (s) != TOKEN_VOID)
            dump_symbol (s, &data);
          else
            M4ERROR ((warning_status, 0,
                      "undefined macro `%s'", TOKEN_DATA_TEXT (argv[i])));
        }
    }

  /* Make table of symbols invisible to expand_macro ().  */

  obstack_finish (obs);

  qsort (data.base, data.size, sizeof (symbol *), dumpdef_cmp);

  for (; data.size > 0; --data.size, data.base++)
    {
      DEBUG_PRINT1 ("%s:\t", SYMBOL_NAME (data.base[0]));

      switch (SYMBOL_TYPE (data.base[0]))
        {
        case TOKEN_TEXT:
          if (debug_level & DEBUG_TRACE_QUOTE)
            DEBUG_PRINT3 ("%s%s%s\n",
                          lquote.string, SYMBOL_TEXT (data.base[0]), rquote.string);
          else
            DEBUG_PRINT1 ("%s\n", SYMBOL_TEXT (data.base[0]));
          break;

        case TOKEN_FUNC:
          bp = find_builtin_by_addr (SYMBOL_FUNC (data.base[0]));
          if (bp == NULL)
            {
              M4ERROR ((warning_status, 0, "\
INTERNAL ERROR: builtin not found in builtin table"));
              abort ();
            }
          DEBUG_PRINT1 ("<%s>\n", bp->name);
          break;

        case TOKEN_VOID:
        default:
          M4ERROR ((warning_status, 0,
                    "INTERNAL ERROR: bad token data type in m4_dumpdef ()"));
          abort ();
          break;
        }
    }
}

/*-----------------------------------------------------------------.
| The builtin "builtin" allows calls to builtin macros, even if    |
| their definition has been overridden or shadowed.  It is thus    |
| possible to redefine builtins, and still access their original   |
| definition.  This macro is not available in compatibility mode.  |
`-----------------------------------------------------------------*/

static void
m4_builtin (struct obstack *obs, int argc, token_data **argv)
{
  const builtin *bp;
  const char *name;

  if (bad_argc (argv[0], argc, 2, -1))
    return;
  if (TOKEN_DATA_TYPE (argv[1]) != TOKEN_TEXT)
    {
      M4ERROR ((warning_status, 0,
                "Warning: %s: invalid macro name ignored", ARG (0)));
      return;
    }

  name = ARG (1);
  bp = find_builtin_by_name (name);
  if (bp->func == m4_placeholder)
    M4ERROR ((warning_status, 0,
              "undefined builtin `%s'", name));
  else
    {
      int i;
      if (! bp->groks_macro_args)
        for (i = 2; i < argc; i++)
          if (TOKEN_DATA_TYPE (argv[i]) != TOKEN_TEXT)
            {
              TOKEN_DATA_TYPE (argv[i]) = TOKEN_TEXT;
              TOKEN_DATA_TEXT (argv[i]) = (char *) "";
            }
      bp->func (obs, argc - 1, argv + 1);
    }
}

/*-------------------------------------------------------------------.
| The builtin "indir" allows indirect calls to macros, even if their |
| name is not a proper macro name.  It is thus possible to define    |
| macros with ill-formed names for internal use in larger macro      |
| packages.  This macro is not available in compatibility mode.      |
`-------------------------------------------------------------------*/

static void
m4_indir (struct obstack *obs, int argc, token_data **argv)
{
  symbol *s;
  const char *name;

  if (bad_argc (argv[0], argc, 2, -1))
    return;
  if (TOKEN_DATA_TYPE (argv[1]) != TOKEN_TEXT)
    {
      M4ERROR ((warning_status, 0,
                "Warning: %s: invalid macro name ignored", ARG (0)));
      return;
    }

  name = ARG (1);
  s = lookup_symbol (name, SYMBOL_LOOKUP);
  if (s == NULL || SYMBOL_TYPE (s) == TOKEN_VOID)
    M4ERROR ((warning_status, 0,
              "undefined macro `%s'", name));
  else
    {
      int i;
      if (! SYMBOL_MACRO_ARGS (s))
        for (i = 2; i < argc; i++)
          if (TOKEN_DATA_TYPE (argv[i]) != TOKEN_TEXT)
            {
              TOKEN_DATA_TYPE (argv[i]) = TOKEN_TEXT;
              TOKEN_DATA_TEXT (argv[i]) = (char *) "";
            }
      call_macro (s, argc - 1, argv + 1, obs);
    }
}

/*------------------------------------------------------------------.
| The macro "defn" returns the quoted definition of the macro named |
| by the first argument.  If the macro is builtin, it will push a   |
| special macro-definition token on the input stack.                |
`------------------------------------------------------------------*/

static void
m4_defn (struct obstack *obs, int argc, token_data **argv)
{
  symbol *s;
  builtin_func *b;
  unsigned int i;

  if (bad_argc (argv[0], argc, 2, -1))
    return;

  assert (0 < argc);
  for (i = 1; i < (unsigned) argc; i++)
    {
      const char *arg = ARG((int) i);
      s = lookup_symbol (arg, SYMBOL_LOOKUP);
      if (s == NULL)
        continue;

      switch (SYMBOL_TYPE (s))
        {
        case TOKEN_TEXT:
          obstack_grow (obs, lquote.string, lquote.length);
          obstack_grow (obs, SYMBOL_TEXT (s), strlen (SYMBOL_TEXT (s)));
          obstack_grow (obs, rquote.string, rquote.length);
          break;

        case TOKEN_FUNC:
          b = SYMBOL_FUNC (s);
          if (b == m4_placeholder)
            M4ERROR ((warning_status, 0, "\
builtin `%s' requested by frozen file is not supported", arg));
          else if (argc != 2)
            M4ERROR ((warning_status, 0,
                      "Warning: cannot concatenate builtin `%s'",
                      arg));
          else
            push_macro (b);
          break;

        case TOKEN_VOID:
          /* Nothing to do for traced but undefined macro.  */
          break;

        default:
          M4ERROR ((warning_status, 0,
                    "INTERNAL ERROR: bad symbol type in m4_defn ()"));
          abort ();
        }
    }
}

/*--------------------------------------------------------------.
| This section contains macros to handle the builtins "syscmd", |
| "esyscmd" and "sysval".  "esyscmd" is GNU specific.           |
`--------------------------------------------------------------*/

/* Exit code from last "syscmd" command.  */
static int sysval;

static void
m4_syscmd (struct obstack *obs M4_GNUC_UNUSED, int argc, token_data **argv)
{
  const char *cmd = ARG (1);
  int status;
  int sig_status;
  const char *prog_args[4] = { "sh", "-c" };
  if (bad_argc (argv[0], argc, 2, 2) || !*cmd)
    {
      /* The empty command is successful.  */
      sysval = 0;
      return;
    }

  debug_flush_files ();
#if W32_NATIVE
  if (strstr (SYSCMD_SHELL, "cmd"))
    {
      prog_args[0] = "cmd";
      prog_args[1] = "/c";
    }
#endif
  prog_args[2] = cmd;
  errno = 0;
  status = execute (ARG (0), SYSCMD_SHELL, (char **) prog_args, false,
                    false, false, false, true, false, &sig_status);
  if (sig_status)
    {
      assert (status == 127);
      sysval = sig_status << 8;
    }
  else
    {
      if (status == 127 && errno)
        M4ERROR ((warning_status, errno, "cannot run command `%s'", cmd));
      sysval = status;
    }
}

static void
m4_esyscmd (struct obstack *obs, int argc, token_data **argv)
{
  const char *cmd = ARG (1);
  const char *prog_args[4] = { "sh", "-c" };
  pid_t child;
  int fd;
  FILE *pin;
  int status;
  int sig_status;

  if (bad_argc (argv[0], argc, 2, 2) || !*cmd)
    {
      /* The empty command is successful.  */
      sysval = 0;
      return;
    }

  debug_flush_files ();
#if W32_NATIVE
  if (strstr (SYSCMD_SHELL, "cmd"))
    {
      prog_args[0] = "cmd";
      prog_args[1] = "/c";
    }
#endif
  prog_args[2] = cmd;
  errno = 0;
  child = create_pipe_in (ARG (0), SYSCMD_SHELL, (char **) prog_args,
                          NULL, false, true, false, &fd);
  if (child == -1)
    {
      M4ERROR ((warning_status, errno, "cannot run command `%s'", cmd));
      sysval = 127;
      return;
    }
  pin = fdopen (fd, "r");
  if (pin == NULL)
    {
      M4ERROR ((warning_status, errno, "cannot run command `%s'", cmd));
      sysval = 127;
      close (fd);
      return;
    }
  while (1)
    {
      size_t avail = obstack_room (obs);
      size_t len;
      if (!avail)
        {
          int ch = getc (pin);
          if (ch == EOF)
            break;
          obstack_1grow (obs, ch);
          continue;
        }
      len = fread (obstack_next_free (obs), 1, avail, pin);
      if (len <= 0)
        break;
      obstack_blank_fast (obs, len);
    }
  if (ferror (pin) || fclose (pin))
    M4ERROR ((EXIT_FAILURE, errno, "cannot read pipe"));
  errno = 0;
  status = wait_subprocess (child, ARG (0), false, true, true, false,
                            &sig_status);
  if (sig_status)
    {
      assert (status == 127);
      sysval = sig_status << 8;
    }
  else
    {
      if (status == 127 && errno)
        M4ERROR ((warning_status, errno, "cannot run command `%s'", cmd));
      sysval = status;
    }
}

static void
m4_sysval (struct obstack *obs, int argc M4_GNUC_UNUSED,
           token_data **argv M4_GNUC_UNUSED)
{
  shipout_int (obs, sysval);
}

/*------------------------------------------------------------------.
| This section contains the top level code for the "eval" builtin.  |
| The actual work is done in the function evaluate (), which lives  |
| in eval.c.                                                        |
`------------------------------------------------------------------*/

static void
m4_eval (struct obstack *obs, int argc, token_data **argv)
{
  int32_t value = 0;
  int radix = 10;
  int min = 1;
  const char *s;

  if (bad_argc (argv[0], argc, 2, 4))
    return;

  if (*ARG (2) && !numeric_arg (argv[0], ARG (2), &radix))
    return;

  if (radix < 1 || radix > (int) strlen (digits))
    {
      M4ERROR ((warning_status, 0,
                "radix %d in builtin `%s' out of range",
                radix, ARG (0)));
      return;
    }

  if (argc >= 4 && !numeric_arg (argv[0], ARG (3), &min))
    return;
  if (min < 0)
    {
      M4ERROR ((warning_status, 0,
                "negative width to builtin `%s'", ARG (0)));
      return;
    }

  if (!*ARG (1))
    M4ERROR ((warning_status, 0,
              "empty string treated as 0 in builtin `%s'", ARG (0)));
  else if (evaluate (ARG (1), &value))
    return;

  if (radix == 1)
    {
      if (value < 0)
        {
          obstack_1grow (obs, '-');
          value = -value;
        }
      /* This assumes 2's-complement for correctly handling INT_MIN.  */
      while (min-- - value > 0)
        obstack_1grow (obs, '0');
      while (value-- != 0)
        obstack_1grow (obs, '1');
      obstack_1grow (obs, '\0');
      return;
    }

  s = ntoa (value, radix);

  if (*s == '-')
    {
      obstack_1grow (obs, '-');
      s++;
    }
  for (min -= strlen (s); --min >= 0;)
    obstack_1grow (obs, '0');

  obstack_grow (obs, s, strlen (s));
}

static void
m4_incr (struct obstack *obs, int argc, token_data **argv)
{
  int value;

  if (bad_argc (argv[0], argc, 2, 2))
    return;

  if (!numeric_arg (argv[0], ARG (1), &value))
    return;

  shipout_int (obs, value + 1);
}

static void
m4_decr (struct obstack *obs, int argc, token_data **argv)
{
  int value;

  if (bad_argc (argv[0], argc, 2, 2))
    return;

  if (!numeric_arg (argv[0], ARG (1), &value))
    return;

  shipout_int (obs, value - 1);
}

/* This section contains the macros "divert", "undivert" and "divnum" for
   handling diversion.  The utility functions used lives in output.c.  */

/*-----------------------------------------------------------------.
| Divert further output to the diversion given by ARGV[1].  Out of |
| range means discard further output.                              |
`-----------------------------------------------------------------*/

static void
m4_divert (struct obstack *obs M4_GNUC_UNUSED, int argc, token_data **argv)
{
  int i = 0;

  if (bad_argc (argv[0], argc, 1, 2))
    return;

  if (argc >= 2 && !numeric_arg (argv[0], ARG (1), &i))
    return;

  make_diversion (i);
}

/*-----------------------------------------------------.
| Expand to the current diversion number, -1 if none.  |
`-----------------------------------------------------*/

static void
m4_divnum (struct obstack *obs, int argc, token_data **argv)
{
  if (bad_argc (argv[0], argc, 1, 1))
    return;
  shipout_int (obs, current_diversion);
}

/*------------------------------------------------------------------.
| Bring back the diversion given by the argument list.  If none is  |
| specified, bring back all diversions.  GNU specific is the option |
| of undiverting named files, by passing a non-numeric argument to  |
| undivert ().                                                      |
`------------------------------------------------------------------*/

static void
m4_undivert (struct obstack *obs M4_GNUC_UNUSED, int argc, token_data **argv)
{
  int i, file;
  FILE *fp;
  char *endp;

  if (argc == 1)
    undivert_all ();
  else
    for (i = 1; i < argc; i++)
      {
        file = strtol (ARG (i), &endp, 10);
        if (*endp == '\0' && !isspace (to_uchar (*ARG (i))))
          insert_diversion (file);
        else if (no_gnu_extensions)
          M4ERROR ((warning_status, 0,
                    "non-numeric argument to builtin `%s'", ARG (0)));
        else
          {
            fp = m4_path_search (ARG (i), NULL);
            if (fp != NULL)
              {
                insert_file (fp);
                if (fclose (fp) == EOF)
                  M4ERROR ((warning_status, errno,
                            "error undiverting `%s'", ARG (i)));
              }
            else
              M4ERROR ((warning_status, errno,
                        "cannot undivert `%s'", ARG (i)));
          }
      }
}

/* This section contains various macros, which does not fall into any
   specific group.  These are "dnl", "shift", "changequote", "changecom"
   and "changeword".  */

/*-----------------------------------------------------------.
| Delete all subsequent whitespace from input.  The function |
| skip_line () lives in input.c.                             |
`-----------------------------------------------------------*/

static void
m4_dnl (struct obstack *obs M4_GNUC_UNUSED, int argc, token_data **argv)
{
  if (bad_argc (argv[0], argc, 1, 1))
    return;

  skip_line ();
}

/*--------------------------------------------------------------------.
| Shift all arguments one to the left, discarding the first           |
| argument.  Each output argument is quoted with the current quotes.  |
`--------------------------------------------------------------------*/

static void
m4_shift (struct obstack *obs, int argc, token_data **argv)
{
  if (bad_argc (argv[0], argc, 2, -1))
    return;
  dump_args (obs, argc - 1, argv + 1, ",", true);
}

/*--------------------------------------------------------------------------.
| Change the current quotes.  The function set_quotes () lives in input.c.  |
`--------------------------------------------------------------------------*/

static void
m4_changequote (struct obstack *obs M4_GNUC_UNUSED, int argc,
                token_data **argv)
{
  if (bad_argc (argv[0], argc, 1, 3))
    return;

  /* Explicit NULL distinguishes between empty and missing argument.  */
  set_quotes ((argc >= 2) ? TOKEN_DATA_TEXT (argv[1]) : NULL,
             (argc >= 3) ? TOKEN_DATA_TEXT (argv[2]) : NULL);
}

/*-----------------------------------------------------------------.
| Change the current comment delimiters.  The function set_comment |
| () lives in input.c.                                             |
`-----------------------------------------------------------------*/

static void
m4_changecom (struct obstack *obs M4_GNUC_UNUSED, int argc, token_data **argv)
{
  if (bad_argc (argv[0], argc, 1, 3))
    return;

  /* Explicit NULL distinguishes between empty and missing argument.  */
  set_comment ((argc >= 2) ? TOKEN_DATA_TEXT (argv[1]) : NULL,
               (argc >= 3) ? TOKEN_DATA_TEXT (argv[2]) : NULL);
}

#ifdef ENABLE_CHANGEWORD

/*---------------------------------------------------------------.
| Change the regular expression used for breaking the input into |
| words.  The function set_word_regexp () lives in input.c.      |
`---------------------------------------------------------------*/

static void
m4_changeword (struct obstack *obs M4_GNUC_UNUSED, int argc, token_data **argv)
{
  if (bad_argc (argv[0], argc, 2, 2))
    return;

  set_word_regexp (TOKEN_DATA_TEXT (argv[1]));
}

#endif /* ENABLE_CHANGEWORD */

/* This section contains macros for inclusion of other files -- "include"
   and "sinclude".  This differs from bringing back diversions, in that
   the input is scanned before being copied to the output.  */

/*---------------------------------------------------------------.
| Generic include function.  Include the file given by the first |
| argument, if it exists.  Complain about inaccessible files iff |
| SILENT is false.                                               |
`---------------------------------------------------------------*/

static void
include (int argc, token_data **argv, bool silent)
{
  FILE *fp;
  char *name;

  if (bad_argc (argv[0], argc, 2, 2))
    return;

  fp = m4_path_search (ARG (1), &name);
  if (fp == NULL)
    {
      if (!silent)
        {
          M4ERROR ((warning_status, errno, "cannot open `%s'", ARG (1)));
          retcode = EXIT_FAILURE;
        }
      return;
    }

  push_file (fp, name, true);
  free (name);
}

/*------------------------------------------------.
| Include a file, complaining in case of errors.  |
`------------------------------------------------*/

static void
m4_include (struct obstack *obs M4_GNUC_UNUSED, int argc, token_data **argv)
{
  include (argc, argv, false);
}

/*----------------------------------.
| Include a file, ignoring errors.  |
`----------------------------------*/

static void
m4_sinclude (struct obstack *obs M4_GNUC_UNUSED, int argc, token_data **argv)
{
  include (argc, argv, true);
}

/* More miscellaneous builtins -- "maketemp", "errprint", "__file__",
   "__line__", and "__program__".  The last three are GNU specific.  */

/*------------------------------------------------------------------.
| Use the first argument as at template for a temporary file name.  |
`------------------------------------------------------------------*/

/* Add trailing 'X' to PATTERN of length LEN as necessary, then
   securely create the file, and place the quoted new file name on
   OBS.  Report errors on behalf of ME.  */
static void
mkstemp_helper (struct obstack *obs, const char *me, const char *pattern,
                size_t len)
{
  int fd;
  size_t i;
  char *name;

  /* Guarantee that there are six trailing 'X' characters, even if the
     user forgot to supply them.  Output must be quoted if
     successful.  */
  obstack_grow (obs, lquote.string, lquote.length);
  obstack_grow (obs, pattern, len);
  for (i = 0; len > 0 && i < 6; i++)
    if (pattern[len - i - 1] != 'X')
      break;
  obstack_grow0 (obs, "XXXXXX", 6 - i);
  name = (char *) obstack_base (obs) + lquote.length;

  errno = 0;
  fd = mkstemp (name);
  if (fd < 0)
    {
      M4ERROR ((0, errno, "%s: cannot create tempfile `%s'", me, pattern));
      obstack_free (obs, obstack_finish (obs));
    }
  else
    {
      close (fd);
      /* Remove NUL, then finish quote.  */
      obstack_blank (obs, -1);
      obstack_grow (obs, rquote.string, rquote.length);
    }
}

static void
m4_maketemp (struct obstack *obs, int argc, token_data **argv)
{
  if (bad_argc (argv[0], argc, 2, 2))
    return;
  if (no_gnu_extensions)
    {
      /* POSIX states "any trailing 'X' characters [are] replaced with
         the current process ID as a string", without referencing the
         file system.  Horribly insecure, but we have to do it when we
         are in traditional mode.

         For reference, Solaris m4 does:
           maketemp() -> `'
           maketemp(X) -> `X'
           maketemp(XX) -> `Xn', where n is last digit of pid
           maketemp(XXXXXXXX) -> `X00nnnnn', where nnnnn is 16-bit pid
      */
      const char *str = ARG (1);
      int len = strlen (str);
      int i;
      int len2;

      M4ERROR ((warning_status, 0, "recommend using mkstemp instead"));
      for (i = len; i > 1; i--)
        if (str[i - 1] != 'X')
          break;
      obstack_grow (obs, str, i);
      str = ntoa ((int32_t) getpid (), 10);
      len2 = strlen (str);
      if (len2 > len - i)
        obstack_grow0 (obs, str + len2 - (len - i), len - i);
      else
        {
          while (i++ < len - len2)
            obstack_1grow (obs, '0');
          obstack_grow0 (obs, str, len2);
        }
    }
  else
    mkstemp_helper (obs, ARG (0), ARG (1), strlen (ARG (1)));
}

static void
m4_mkstemp (struct obstack *obs, int argc, token_data **argv)
{
  if (bad_argc (argv[0], argc, 2, 2))
    return;
  mkstemp_helper (obs, ARG (0), ARG (1), strlen (ARG (1)));
}

/*----------------------------------------.
| Print all arguments on standard error.  |
`----------------------------------------*/

static void
m4_errprint (struct obstack *obs, int argc, token_data **argv)
{
  if (bad_argc (argv[0], argc, 2, -1))
    return;
  dump_args (obs, argc, argv, " ", false);
  obstack_1grow (obs, '\0');
  debug_flush_files ();
  xfprintf (stderr, "%s", (char *) obstack_finish (obs));
  fflush (stderr);
}

static void
m4___file__ (struct obstack *obs, int argc, token_data **argv)
{
  if (bad_argc (argv[0], argc, 1, 1))
    return;
  obstack_grow (obs, lquote.string, lquote.length);
  obstack_grow (obs, current_file, strlen (current_file));
  obstack_grow (obs, rquote.string, rquote.length);
}

static void
m4___line__ (struct obstack *obs, int argc, token_data **argv)
{
  if (bad_argc (argv[0], argc, 1, 1))
    return;
  shipout_int (obs, current_line);
}

static void
m4___program__ (struct obstack *obs, int argc, token_data **argv)
{
  if (bad_argc (argv[0], argc, 1, 1))
    return;
  obstack_grow (obs, lquote.string, lquote.length);
  obstack_grow (obs, program_name, strlen (program_name));
  obstack_grow (obs, rquote.string, rquote.length);
}

/* This section contains various macros for exiting, saving input until
   EOF is seen, and tracing macro calls.  That is: "m4exit", "m4wrap",
   "traceon" and "traceoff".  */

/*----------------------------------------------------------.
| Exit immediately, with exit status specified by the first |
| argument, or 0 if no arguments are present.               |
`----------------------------------------------------------*/

static void M4_GNUC_NORETURN
m4_m4exit (struct obstack *obs M4_GNUC_UNUSED, int argc, token_data **argv)
{
  int exit_code = EXIT_SUCCESS;

  /* Warn on bad arguments, but still exit.  */
  bad_argc (argv[0], argc, 1, 2);
  if (argc >= 2 && !numeric_arg (argv[0], ARG (1), &exit_code))
    exit_code = EXIT_FAILURE;
  if (exit_code < 0 || exit_code > 255)
    {
      M4ERROR ((warning_status, 0,
                "exit status out of range: `%d'", exit_code));
      exit_code = EXIT_FAILURE;
    }
  /* Change debug stream back to stderr, to force flushing debug stream and
     detect any errors it might have encountered.  */
  debug_set_output (NULL);
  debug_flush_files ();
  if (exit_code == EXIT_SUCCESS && retcode != EXIT_SUCCESS)
    exit_code = retcode;
  /* Propagate non-zero status to atexit handlers.  */
  if (exit_code != EXIT_SUCCESS)
    exit_failure = exit_code;
  exit (exit_code);
}

/*------------------------------------------------------------------.
| Save the argument text until EOF has been seen, allowing for user |
| specified cleanup action.  GNU version saves all arguments, the   |
| standard version only the first.                                  |
`------------------------------------------------------------------*/

static void
m4_m4wrap (struct obstack *obs, int argc, token_data **argv)
{
  if (bad_argc (argv[0], argc, 2, -1))
    return;
  if (no_gnu_extensions)
    obstack_grow (obs, ARG (1), strlen (ARG (1)));
  else
    dump_args (obs, argc, argv, " ", false);
  obstack_1grow (obs, '\0');
  push_wrapup ((char *) obstack_finish (obs));
}

/* Enable tracing of all specified macros, or all, if none is specified.
   Tracing is disabled by default, when a macro is defined.  This can be
   overridden by the "t" debug flag.  */

/*------------------------------------------------------------------.
| Set_trace () is used by "traceon" and "traceoff" to enable and    |
| disable tracing of a macro.  It disables tracing if DATA is NULL, |
| otherwise it enables tracing.                                     |
`------------------------------------------------------------------*/

static void
set_trace (symbol *sym, void *data)
{
  SYMBOL_TRACED (sym) = data != NULL;
  /* Remove placeholder from table if macro is undefined and untraced.  */
  if (SYMBOL_TYPE (sym) == TOKEN_VOID && data == NULL)
    lookup_symbol (SYMBOL_NAME (sym), SYMBOL_POPDEF);
}

static void
m4_traceon (struct obstack *obs, int argc, token_data **argv)
{
  symbol *s;
  int i;

  if (argc == 1)
    hack_all_symbols (set_trace, obs);
  else
    for (i = 1; i < argc; i++)
      {
        s = lookup_symbol (ARG (i), SYMBOL_LOOKUP);
        if (!s)
          s = lookup_symbol (ARG (i), SYMBOL_INSERT);
        set_trace (s, obs);
      }
}

/*------------------------------------------------------------------------.
| Disable tracing of all specified macros, or all, if none is specified.  |
`------------------------------------------------------------------------*/

static void
m4_traceoff (struct obstack *obs M4_GNUC_UNUSED, int argc, token_data **argv)
{
  symbol *s;
  int i;

  if (argc == 1)
    hack_all_symbols (set_trace, NULL);
  else
    for (i = 1; i < argc; i++)
      {
        s = lookup_symbol (TOKEN_DATA_TEXT (argv[i]), SYMBOL_LOOKUP);
        if (s != NULL)
          set_trace (s, NULL);
      }
}

/*------------------------------------------------------------------.
| On-the-fly control of the format of the tracing output.  It takes |
| one argument, which is a character string like given to the -d    |
| option, or none in which case the debug_level is zeroed.          |
`------------------------------------------------------------------*/

static void
m4_debugmode (struct obstack *obs M4_GNUC_UNUSED, int argc, token_data **argv)
{
  int new_debug_level;
  int change_flag;

  if (bad_argc (argv[0], argc, 1, 2))
    return;

  if (argc == 1)
    debug_level = 0;
  else
    {
      if (ARG (1)[0] == '+' || ARG (1)[0] == '-')
        {
          change_flag = ARG (1)[0];
          new_debug_level = debug_decode (ARG (1) + 1);
        }
      else
        {
          change_flag = 0;
          new_debug_level = debug_decode (ARG (1));
        }

      if (new_debug_level < 0)
        M4ERROR ((warning_status, 0,
                  "Debugmode: bad debug flags: `%s'", ARG (1)));
      else
        {
          switch (change_flag)
            {
            case 0:
              debug_level = new_debug_level;
              break;

            case '+':
              debug_level |= new_debug_level;
              break;

            case '-':
              debug_level &= ~new_debug_level;
              break;

            default:
              M4ERROR ((warning_status, 0,
                        "INTERNAL ERROR: bad flag in m4_debugmode ()"));
              abort ();
            }
        }
    }
}

/*-------------------------------------------------------------------------.
| Specify the destination of the debugging output.  With one argument, the |
| argument is taken as a file name, with no arguments, revert to stderr.   |
`-------------------------------------------------------------------------*/

static void
m4_debugfile (struct obstack *obs M4_GNUC_UNUSED, int argc, token_data **argv)
{
  if (bad_argc (argv[0], argc, 1, 2))
    return;

  if (argc == 1)
    debug_set_output (NULL);
  else if (!debug_set_output (ARG (1)))
    M4ERROR ((warning_status, errno,
              "cannot set debug file `%s'", ARG (1)));
}

/* This section contains text processing macros: "len", "index",
   "substr", "translit", "format", "regexp" and "patsubst".  The last
   three are GNU specific.  */

/*---------------------------------------------.
| Expand to the length of the first argument.  |
`---------------------------------------------*/

static void
m4_len (struct obstack *obs, int argc, token_data **argv)
{
  if (bad_argc (argv[0], argc, 2, 2))
    return;
  shipout_int (obs, strlen (ARG (1)));
}

/*-------------------------------------------------------------------.
| The macro expands to the first index of the second argument in the |
| first argument.                                                    |
`-------------------------------------------------------------------*/

static void
m4_index (struct obstack *obs, int argc, token_data **argv)
{
  const char *haystack;
  const char *result;
  int retval;

  if (bad_argc (argv[0], argc, 3, 3))
    {
      /* builtin(`index') is blank, but index(`abc') is 0.  */
      if (argc == 2)
        shipout_int (obs, 0);
      return;
    }

  haystack = ARG (1);
  result = strstr (haystack, ARG (2));
  retval = result ? result - haystack : -1;

  shipout_int (obs, retval);
}

/*-----------------------------------------------------------------.
| The macro "substr" extracts substrings from the first argument,  |
| starting from the index given by the second argument, extending  |
| for a length given by the third argument.  If the third argument |
| is missing, the substring extends to the end of the first        |
| argument.                                                        |
`-----------------------------------------------------------------*/

static void
m4_substr (struct obstack *obs, int argc, token_data **argv)
{
  int start = 0;
  int length, avail;

  if (bad_argc (argv[0], argc, 3, 4))
    {
      /* builtin(`substr') is blank, but substr(`abc') is abc.  */
      if (argc == 2)
        obstack_grow (obs, ARG (1), strlen (ARG (1)));
      return;
    }

  length = avail = strlen (ARG (1));
  if (!numeric_arg (argv[0], ARG (2), &start))
    return;

  if (argc >= 4 && !numeric_arg (argv[0], ARG (3), &length))
    return;

  if (start < 0 || length <= 0 || start >= avail)
    return;

  if (start + length > avail)
    length = avail - start;
  obstack_grow (obs, ARG (1) + start, length);
}

/*------------------------------------------------------------------.
| For "translit", ranges are allowed in the second and third        |
| argument.  They are expanded in the following function, and the   |
| expanded strings, without any ranges left, are used to translate  |
| the characters of the first argument.  A single - (dash) can be   |
| included in the strings by being the first or the last character  |
| in the string.  If the first character in a range is after the    |
| first in the character set, the range is made backwards, thus 9-0 |
| is the string 9876543210.                                         |
`------------------------------------------------------------------*/

static const char *
expand_ranges (const char *s, struct obstack *obs)
{
  unsigned char from;
  unsigned char to;

  for (from = '\0'; *s != '\0'; from = to_uchar (*s++))
    {
      if (*s == '-' && from != '\0')
        {
          to = to_uchar (*++s);
          if (to == '\0')
            {
              /* trailing dash */
              obstack_1grow (obs, '-');
              break;
            }
          else if (from <= to)
            {
              while (from++ < to)
                obstack_1grow (obs, from);
            }
          else
            {
              while (--from >= to)
                obstack_1grow (obs, from);
            }
        }
      else
        obstack_1grow (obs, *s);
    }
  obstack_1grow (obs, '\0');
  return (char *) obstack_finish (obs);
}

/*-----------------------------------------------------------------.
| The macro "translit" translates all characters in the first      |
| argument, which are present in the second argument, into the     |
| corresponding character from the third argument.  If the third   |
| argument is shorter than the second, the extra characters in the |
| second argument are deleted from the first.                      |
`-----------------------------------------------------------------*/

static void
m4_translit (struct obstack *obs, int argc, token_data **argv)
{
  const char *data = ARG (1);
  const char *from = ARG (2);
  const char *to;
  char map[UCHAR_MAX + 1];
  char found[UCHAR_MAX + 1];
  unsigned char ch;

  if (bad_argc (argv[0], argc, 3, 4) || !*data || !*from)
    {
      /* builtin(`translit') is blank, but translit(`abc') is abc.  */
      if (2 <= argc)
        obstack_grow (obs, data, strlen (data));
      return;
    }

  to = ARG (3);
  if (strchr (to, '-') != NULL)
    {
      to = expand_ranges (to, obs);
      assert (to && *to);
    }

  /* If there are only one or two bytes to replace, it is faster to
     use memchr2.  Using expand_ranges does nothing unless there are
     at least three bytes.  */
  if (!from[1] || !from[2])
    {
      const char *p;
      size_t len = strlen (data);
      while ((p = (char *) memchr2 (data, from[0], from[1], len)))
        {
          obstack_grow (obs, data, p - data);
          len -= p - data;
          if (!len)
            return;
          data = p + 1;
          len--;
          if (*p == from[0] && to[0])
            obstack_1grow (obs, to[0]);
          else if (*p == from[1] && to[0] && to[1])
            obstack_1grow (obs, to[1]);
        }
      obstack_grow (obs, data, len);
      return;
    }

  if (strchr (from, '-') != NULL)
    {
      from = expand_ranges (from, obs);
      assert (from && *from);
    }

  /* Calling strchr(from) for each character in data is quadratic,
     since both strings can be arbitrarily long.  Instead, create a
     from-to mapping in one pass of from, then use that map in one
     pass of data, for linear behavior.  Traditional behavior is that
     only the first instance of a character in from is consulted,
     hence the found map.  */
  memset (map, 0, sizeof map);
  memset (found, 0, sizeof found);
  for ( ; (ch = *from) != '\0'; from++)
    {
      if (! found[ch])
        {
          found[ch] = 1;
          map[ch] = *to;
        }
      if (*to != '\0')
        to++;
    }

  for (data = ARG (1); (ch = *data) != '\0'; data++)
    {
      if (! found[ch])
        obstack_1grow (obs, ch);
      else if (map[ch])
        obstack_1grow (obs, map[ch]);
    }
}

/*-------------------------------------------------------------------.
| Frontend for printf like formatting.  The function format () lives |
| in the file format.c.                                              |
`-------------------------------------------------------------------*/

static void
m4_format (struct obstack *obs, int argc, token_data **argv)
{
  if (bad_argc (argv[0], argc, 2, -1))
    return;
  expand_format (obs, argc - 1, argv + 1);
}

/*------------------------------------------------------------------.
| Function to perform substitution by regular expressions.  Used by |
| the builtins regexp and patsubst.  The changed text is placed on  |
| the obstack.  The substitution is REPL, with \& substituted by    |
| this part of VICTIM matched by the last whole regular expression, |
| taken from REGS[0], and \N substituted by the text matched by the |
| Nth parenthesized sub-expression, taken from REGS[N].             |
`------------------------------------------------------------------*/

static int substitute_warned = 0;

static void
substitute (struct obstack *obs, const char *victim, const char *repl,
            struct re_registers *regs)
{
  int ch;
  __re_size_t ind;
  while (1)
    {
      const char *backslash = strchr (repl, '\\');
      if (!backslash)
        {
          obstack_grow (obs, repl, strlen (repl));
          return;
        }
      obstack_grow (obs, repl, backslash - repl);
      repl = backslash;
      ch = *++repl;
      switch (ch)
        {
        case '0':
          if (!substitute_warned)
            {
              M4ERROR ((warning_status, 0, "\
Warning: \\0 will disappear, use \\& instead in replacements"));
              substitute_warned = 1;
            }
          /* Fall through.  */

        case '&':
          obstack_grow (obs, victim + regs->start[0],
                        regs->end[0] - regs->start[0]);
          repl++;
          break;

        case '1': case '2': case '3': case '4': case '5': case '6':
        case '7': case '8': case '9':
          ind = ch -= '0';
          if (regs->num_regs - 1 <= ind)
            M4ERROR ((warning_status, 0,
                      "Warning: sub-expression %d not present", ch));
          else if (regs->end[ch] > 0)
            obstack_grow (obs, victim + regs->start[ch],
                          regs->end[ch] - regs->start[ch]);
          repl++;
          break;

        case '\0':
          M4ERROR ((warning_status, 0,
                    "Warning: trailing \\ ignored in replacement"));
          return;

        default:
          obstack_1grow (obs, ch);
          repl++;
          break;
        }
    }
}

/*------------------------------------------.
| Initialize regular expression variables.  |
`------------------------------------------*/

void
init_pattern_buffer (struct re_pattern_buffer *buf, struct re_registers *regs)
{
  buf->translate = NULL;
  buf->fastmap = NULL;
  buf->buffer = NULL;
  buf->allocated = 0;
  if (regs)
    {
      regs->start = NULL;
      regs->end = NULL;
    }
}

/*------------------------------------------------------------------.
| Regular expression version of index.  Given two arguments, expand |
| to the index of the first match of the second argument (a regexp) |
| in the first.  Expand to -1 if here is no match.  Given a third   |
| argument, it changes the expansion to this argument.              |
`------------------------------------------------------------------*/

static void
m4_regexp (struct obstack *obs, int argc, token_data **argv)
{
  const char *victim;           /* first argument */
  const char *regexp;           /* regular expression */
  const char *repl;             /* replacement string */

  struct re_pattern_buffer buf; /* compiled regular expression */
  struct re_registers regs;     /* for subexpression matches */
  const char *msg;              /* error message from re_compile_pattern */
  int startpos;                 /* start position of match */
  int length;                   /* length of first argument */

  if (bad_argc (argv[0], argc, 3, 4))
    {
      /* builtin(`regexp') is blank, but regexp(`abc') is 0.  */
      if (argc == 2)
        shipout_int (obs, 0);
      return;
    }

  victim = TOKEN_DATA_TEXT (argv[1]);
  regexp = TOKEN_DATA_TEXT (argv[2]);

  init_pattern_buffer (&buf, &regs);
  msg = re_compile_pattern (regexp, strlen (regexp), &buf);

  if (msg != NULL)
    {
      M4ERROR ((warning_status, 0,
                "bad regular expression: `%s': %s", regexp, msg));
      free_pattern_buffer (&buf, &regs);
      return;
    }

  length = strlen (victim);
  /* Avoid overhead of allocating regs if we won't use it.  */
  startpos = re_search (&buf, victim, length, 0, length,
                        argc == 3 ? NULL : &regs);

  if (startpos == -2)
    M4ERROR ((warning_status, 0,
               "error matching regular expression `%s'", regexp));
  else if (argc == 3)
    shipout_int (obs, startpos);
  else if (startpos >= 0)
    {
      repl = TOKEN_DATA_TEXT (argv[3]);
      substitute (obs, victim, repl, &regs);
    }

  free_pattern_buffer (&buf, &regs);
}

/*--------------------------------------------------------------------------.
| Substitute all matches of a regexp occuring in a string.  Each match of   |
| the second argument (a regexp) in the first argument is changed to the    |
| third argument, with \& substituted by the matched text, and \N           |
| substituted by the text matched by the Nth parenthesized sub-expression.  |
`--------------------------------------------------------------------------*/

static void
m4_patsubst (struct obstack *obs, int argc, token_data **argv)
{
  const char *victim;           /* first argument */
  const char *regexp;           /* regular expression */

  struct re_pattern_buffer buf; /* compiled regular expression */
  struct re_registers regs;     /* for subexpression matches */
  const char *msg;              /* error message from re_compile_pattern */
  int matchpos;                 /* start position of match */
  int offset;                   /* current match offset */
  int length;                   /* length of first argument */

  if (bad_argc (argv[0], argc, 3, 4))
    {
      /* builtin(`patsubst') is blank, but patsubst(`abc') is abc.  */
      if (argc == 2)
        obstack_grow (obs, ARG (1), strlen (ARG (1)));
      return;
    }

  regexp = TOKEN_DATA_TEXT (argv[2]);

  init_pattern_buffer (&buf, &regs);
  msg = re_compile_pattern (regexp, strlen (regexp), &buf);

  if (msg != NULL)
    {
      M4ERROR ((warning_status, 0,
                "bad regular expression `%s': %s", regexp, msg));
      free (buf.buffer);
      return;
    }

  victim = TOKEN_DATA_TEXT (argv[1]);
  length = strlen (victim);

  offset = 0;
  while (offset <= length)
    {
      matchpos = re_search (&buf, victim, length,
                            offset, length - offset, &regs);
      if (matchpos < 0)
        {

          /* Match failed -- either error or there is no match in the
             rest of the string, in which case the rest of the string is
             copied verbatim.  */

          if (matchpos == -2)
            M4ERROR ((warning_status, 0,
                      "error matching regular expression `%s'", regexp));
          else if (offset < length)
            obstack_grow (obs, victim + offset, length - offset);
          break;
        }

      /* Copy the part of the string that was skipped by re_search ().  */

      if (matchpos > offset)
        obstack_grow (obs, victim + offset, matchpos - offset);

      /* Handle the part of the string that was covered by the match.  */

      substitute (obs, victim, ARG (3), &regs);

      /* Update the offset to the end of the match.  If the regexp
         matched a null string, advance offset one more, to avoid
         infinite loops.  */

      offset = regs.end[0];
      if (regs.start[0] == regs.end[0])
        obstack_1grow (obs, victim[offset++]);
    }
  obstack_1grow (obs, '\0');

  free_pattern_buffer (&buf, &regs);
}

/* Finally, a placeholder builtin.  This builtin is not installed by
   default, but when reading back frozen files, this is associated
   with any builtin we don't recognize (for example, if the frozen
   file was created with a changeword capable m4, but is then loaded
   by a different m4 that does not support changeword).  This way, we
   can keep 'm4 -R' quiet in the common case that the user did not
   know or care about the builtin when the frozen file was created,
   while still flagging it as a potential error if an attempt is made
   to actually use the builtin.  */

/*--------------------------------------------------------------------.
| Issue a warning that this macro is a placeholder for an unsupported |
| builtin that was requested while reloading a frozen file.           |
`--------------------------------------------------------------------*/

void
m4_placeholder (struct obstack *obs M4_GNUC_UNUSED, int argc,
                token_data **argv)
{
  M4ERROR ((warning_status, 0, "\
builtin `%s' requested by frozen file is not supported", ARG (0)));
}

/*-------------------------------------------------------------------.
| This function handles all expansion of user defined and predefined |
| macros.  It is called with an obstack OBS, where the macros        |
| expansion will be placed, as an unfinished object.  SYM points to  |
| the macro definition, giving the expansion text.  ARGC and ARGV    |
| are the arguments, as usual.                                       |
`-------------------------------------------------------------------*/

void
expand_user_macro (struct obstack *obs, symbol *sym,
                   int argc, token_data **argv)
{
  const char *text = SYMBOL_TEXT (sym);
  int i;
  while (1)
    {
      const char *dollar = strchr (text, '$');
      if (!dollar)
        {
          obstack_grow (obs, text, strlen (text));
          return;
        }
      obstack_grow (obs, text, dollar - text);
      text = dollar;
      switch (*++text)
        {
        case '0': case '1': case '2': case '3': case '4':
        case '5': case '6': case '7': case '8': case '9':
          if (no_gnu_extensions)
            {
              i = *text++ - '0';
            }
          else
            {
              for (i = 0; isdigit (to_uchar (*text)); text++)
                i = i*10 + (*text - '0');
            }
          if (i < argc)
            obstack_grow (obs, TOKEN_DATA_TEXT (argv[i]),
                          strlen (TOKEN_DATA_TEXT (argv[i])));
          break;

        case '#': /* number of arguments */
          shipout_int (obs, argc - 1);
          text++;
          break;

        case '*': /* all arguments */
        case '@': /* ... same, but quoted */
          dump_args (obs, argc, argv, ",", *text == '@');
          text++;
          break;

        default:
          obstack_1grow (obs, '$');
          break;
        }
    }
}
