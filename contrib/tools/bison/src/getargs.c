/* Parse command line arguments for Bison.

   Copyright (C) 1984, 1986, 1989, 1992, 2000-2015, 2018-2020 Free
   Software Foundation, Inc.

   This file is part of Bison, the GNU Compiler Compiler.

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

#include <config.h>
#include "getargs.h"

#include "system.h"

#include <argmatch.h>
#include <c-strcase.h>
#include <configmake.h>
#include <error.h>
#include <getopt.h>
#include <progname.h>
#include <quote.h>
#include <textstyle.h>

#include "complain.h"
#include "files.h"
#include "muscle-tab.h"
#include "output.h"
#include "uniqstr.h"

bool defines_flag = false;
bool graph_flag = false;
bool xml_flag = false;
bool no_lines_flag = false;
bool token_table_flag = false;
location yacc_loc = EMPTY_LOCATION_INIT;
bool update_flag = false; /* for -u */
bool color_debug = false;

bool nondeterministic_parser = false;
bool glr_parser = false;

int feature_flag = feature_caret;
int report_flag = report_none;
int trace_flag = trace_none;

static struct bison_language const valid_languages[] = {
  /* lang,  skeleton,       ext,     hdr,     add_tab */
  { "c",    "c-skel.m4",    ".c",    ".h",    true },
  { "c++",  "c++-skel.m4",  ".cc",   ".hh",   true },
  { "d",    "d-skel.m4",    ".d",    ".d",    false },
  { "java", "java-skel.m4", ".java", ".java", false },
  { "", "", "", "", false }
};

int skeleton_prio = default_prio;
const char *skeleton = NULL;
int language_prio = default_prio;
struct bison_language const *language = &valid_languages[0];

typedef int* (xargmatch_fn) (const char *context, const char *arg);

/** Decode an option's key.
 *
 *  \param opt        option being decoded.
 *  \param xargmatch  matching function.
 *  \param all        the value of the argument 'all'.
 *  \param flags      the flags to update
 *  \param arg        the subarguments to decode.
 *                    If null, then activate all the flags.
 *  \param no         length of the potential "no-" prefix.
 *                    Can be 0 or 3. If 3, negate the action of the subargument.
 *
 *  If VALUE != 0 then KEY sets flags and no-KEY clears them.
 *  If VALUE == 0 then KEY clears all flags from \c all and no-KEY sets all
 *  flags from \c all.  Thus no-none = all and no-all = none.
 */
static void
flag_argmatch (const char *opt, xargmatch_fn xargmatch,
               int all, int *flags, char *arg, size_t no)
{
  int value = *xargmatch (opt, arg + no);

  /* -rnone == -rno-all, and -rno-none == -rall.  */
  if (!value)
    {
      value = all;
      no = !no;
    }

  if (no)
    *flags &= ~value;
  else
    *flags |= value;
}

typedef void (usage_fn) (FILE *out);

/** Decode an option's set of keys.
 *
 *  \param opt        option being decoded (e.g., --report).
 *  \param xargmatch  matching function.
 *  \param usage      function that implement --help for this option.
 *  \param all        the value of the argument 'all'.
 *  \param flags      the flags to update
 *  \param args       comma separated list of effective subarguments to decode.
 *                    If 0, then activate all the flags.
 */
static void
flags_argmatch (const char *opt,
                xargmatch_fn xargmatch,
                usage_fn usage,
                int all, int *flags, char *args)
{
  if (!args)
    *flags |= all;
  else if (STREQ (args, "help"))
    {
      usage (stdout);
      exit (EXIT_SUCCESS);
    }
  else
    for (args = strtok (args, ","); args; args = strtok (NULL, ","))
      {
        size_t no = STRPREFIX_LIT ("no-", args) ? 3 : 0;
        flag_argmatch (opt, xargmatch,
                       all, flags, args, no);
      }
}


/** Decode a set of sub arguments.
 *
 *  \param FlagName  the flag family to update.
 *  \param Args      the effective sub arguments to decode.
 *  \param All       the "all" value.
 *
 *  \arg FlagName_args   the list of keys.
 *  \arg FlagName_types  the list of values.
 *  \arg FlagName_flag   the flag to update.
 */
#define FLAGS_ARGMATCH(FlagName, Args, All)                             \
  flags_argmatch ("--" #FlagName,                                       \
                  (xargmatch_fn*) argmatch_## FlagName ## _value,        \
                  argmatch_## FlagName ## _usage,                       \
                  All, &FlagName ## _flag, Args)

/*---------------------.
| --color's handling.  |
`---------------------*/

enum color
  {
    color_always,
    color_never,
    color_auto
  };

ARGMATCH_DEFINE_GROUP (color, enum color)

static const argmatch_color_doc argmatch_color_docs[] =
{
  { "always",     N_("colorize the output") },
  { "never",      N_("don't colorize the output") },
  { "auto",       N_("colorize if the output device is a tty") },
  { NULL, NULL },
};

static const argmatch_color_arg argmatch_color_args[] =
{
  { "always",   color_always },
  { "yes",      color_always },
  { "never",    color_never },
  { "no",       color_never },
  { "auto",     color_auto },
  { "tty",      color_auto },
  { NULL, color_always },
};

const argmatch_color_group_type argmatch_color_group =
{
  argmatch_color_args,
  argmatch_color_docs,
  /* TRANSLATORS: Use the same translation for WHEN as in the
     --color=WHEN help message.  */
  N_("WHEN can be one of the following:"),
  NULL
};


/*----------------------.
| --report's handling.  |
`----------------------*/

ARGMATCH_DEFINE_GROUP (report, enum report)

static const argmatch_report_doc argmatch_report_docs[] =
{
  { "states",     N_("describe the states") },
  { "itemsets",   N_("complete the core item sets with their closure") },
  { "lookaheads", N_("explicitly associate lookahead tokens to items") },
  { "solved",     N_("describe shift/reduce conflicts solving") },
  { "all",        N_("include all the above information") },
  { "none",       N_("disable the report") },
  { NULL, NULL },
};

static const argmatch_report_arg argmatch_report_args[] =
{
  { "none",        report_none },
  { "states",      report_states },
  { "itemsets",    report_states | report_itemsets },
  { "lookaheads",  report_states | report_lookahead_tokens },
  { "solved",      report_states | report_solved_conflicts },
  { "all",         report_all },
  { NULL, report_none },
};

const argmatch_report_group_type argmatch_report_group =
{
  argmatch_report_args,
  argmatch_report_docs,
  /* TRANSLATORS: Use the same translation for THINGS as in the
     --report=THINGS help message.  */
  N_("THINGS is a list of comma separated words that can include:"),
  NULL
};

/*---------------------.
| --trace's handling.  |
`---------------------*/

ARGMATCH_DEFINE_GROUP (trace, enum trace)

static const argmatch_trace_doc argmatch_trace_docs[] =
{
  /* Meant for developers only, don't translate them.  */
  { "none",       "no traces" },
  { "locations",  "full display of the locations" },
  { "scan",       "grammar scanner traces" },
  { "parse",      "grammar parser traces" },
  { "automaton",  "construction of the automaton" },
  { "bitsets",    "use of bitsets" },
  { "closure",    "input/output of closure" },
  { "grammar",    "reading, reducing the grammar" },
  { "resource",   "memory consumption (where available)" },
  { "sets",       "grammar sets: firsts, nullable etc." },
  { "muscles",    "m4 definitions passed to the skeleton" },
  { "tools",      "m4 invocation" },
  { "m4",         "m4 traces" },
  { "skeleton",   "skeleton postprocessing" },
  { "time",       "time consumption" },
  { "ielr",       "IELR conversion" },
  { "all",        "all of the above" },
  { NULL, NULL},
};

static const argmatch_trace_arg argmatch_trace_args[] =
{
  { "none",      trace_none },
  { "locations", trace_locations },
  { "scan",      trace_scan },
  { "parse",     trace_parse },
  { "automaton", trace_automaton },
  { "bitsets",   trace_bitsets },
  { "closure",   trace_closure },
  { "grammar",   trace_grammar },
  { "resource",  trace_resource },
  { "sets",      trace_sets },
  { "muscles",   trace_muscles },
  { "tools",     trace_tools },
  { "m4",        trace_m4 },
  { "skeleton",  trace_skeleton },
  { "time",      trace_time },
  { "ielr",      trace_ielr },
  { "all",       trace_all },
  { NULL,        trace_none},
};

const argmatch_trace_group_type argmatch_trace_group =
{
  argmatch_trace_args,
  argmatch_trace_docs,
  N_("TRACES is a list of comma separated words that can include:"),
  NULL
};

/*-----------------------.
| --feature's handling.  |
`-----------------------*/

ARGMATCH_DEFINE_GROUP (feature, enum feature)

static const argmatch_feature_doc argmatch_feature_docs[] =
{
  { "caret",       N_("show errors with carets") },
  { "fixit",       N_("show machine-readable fixes") },
  { "syntax-only", N_("do not generate any file") },
  { "all",         N_("all of the above") },
  { "none",        N_("disable all of the above") },
  { NULL, NULL }
};

static const argmatch_feature_arg argmatch_feature_args[] =
{
  { "none",                          feature_none },
  { "caret",                         feature_caret },
  { "diagnostics-show-caret",        feature_caret },
  { "fixit",                         feature_fixit },
  { "diagnostics-parseable-fixits",  feature_fixit },
  { "syntax-only",                   feature_syntax_only },
  { "all",                           feature_all },
  { NULL, feature_none}
};

const argmatch_feature_group_type argmatch_feature_group =
{
  argmatch_feature_args,
  argmatch_feature_docs,
  /* TRANSLATORS: Use the same translation for FEATURES as in the
     --feature=FEATURES help message.  */
  N_("FEATURES is a list of comma separated words that can include:"),
  NULL
};

/*-------------------------------------------.
| Display the help message and exit STATUS.  |
`-------------------------------------------*/

static void usage (int) ATTRIBUTE_NORETURN;

static void
usage (int status)
{
  if (status != 0)
    fprintf (stderr, _("Try '%s --help' for more information.\n"),
             program_name);
  else
    {
      /* For ../build-aux/cross-options.pl to work, use the format:
                ^  -S, --long[=ARGS] (whitespace)
         A --long option is required.
         Otherwise, add exceptions to ../build-aux/cross-options.pl.  */

      printf (_("Usage: %s [OPTION]... FILE\n"), program_name);
      fputs (_("\
Generate a deterministic LR or generalized LR (GLR) parser employing\n\
LALR(1), IELR(1), or canonical LR(1) parser tables.\n\
\n\
"), stdout);

      fputs (_("\
Mandatory arguments to long options are mandatory for short options too.\n\
"), stdout);
      fputs (_("\
The same is true for optional arguments.\n\
"), stdout);
      putc ('\n', stdout);

      fputs (_("\
Operation Modes:\n\
  -h, --help                 display this help and exit\n\
  -V, --version              output version information and exit\n\
      --print-localedir      output directory containing locale-dependent data\n\
                             and exit\n\
      --print-datadir        output directory containing skeletons and XSLT\n\
                             and exit\n\
  -u, --update               apply fixes to the source grammar file and exit\n\
  -f, --feature[=FEATURES]   activate miscellaneous features\n\
\n\
"), stdout);

      argmatch_feature_usage (stdout);
      putc ('\n', stdout);

      fputs (_("\
Diagnostics:\n\
  -W, --warnings[=CATEGORY]  report the warnings falling in CATEGORY\n\
      --color[=WHEN]         whether to colorize the diagnostics\n\
      --style=FILE           specify the CSS FILE for colorizer diagnostics\n\
\n\
"), stdout);

      warning_usage (stdout);
      putc ('\n', stdout);

      argmatch_color_usage (stdout);
      putc ('\n', stdout);

      fputs (_("\
Tuning the Parser:\n\
  -L, --language=LANGUAGE          specify the output programming language\n\
  -S, --skeleton=FILE              specify the skeleton to use\n\
  -t, --debug                      instrument the parser for tracing\n\
                                   same as '-Dparse.trace'\n\
      --locations                  enable location support\n\
  -D, --define=NAME[=VALUE]        similar to '%define NAME VALUE'\n\
  -F, --force-define=NAME[=VALUE]  override '%define NAME VALUE'\n\
  -p, --name-prefix=PREFIX         prepend PREFIX to the external symbols\n\
                                   deprecated by '-Dapi.prefix={PREFIX}'\n\
  -l, --no-lines                   don't generate '#line' directives\n\
  -k, --token-table                include a table of token names\n\
  -y, --yacc                       emulate POSIX Yacc\n\
"), stdout);
      putc ('\n', stdout);

      /* Keep -d and --defines separate so that ../build-aux/cross-options.pl
       * won't assume that -d also takes an argument.  */
      fputs (_("\
Output Files:\n\
      --defines[=FILE]       also produce a header file\n\
  -d                         likewise but cannot specify FILE (for POSIX Yacc)\n\
  -r, --report=THINGS        also produce details on the automaton\n\
      --report-file=FILE     write report to FILE\n\
  -v, --verbose              same as '--report=state'\n\
  -b, --file-prefix=PREFIX   specify a PREFIX for output files\n\
  -o, --output=FILE          leave output to FILE\n\
  -g, --graph[=FILE]         also output a graph of the automaton\n\
  -x, --xml[=FILE]           also output an XML report of the automaton\n\
"), stdout);
      putc ('\n', stdout);

      argmatch_report_usage (stdout);
      putc ('\n', stdout);

      printf (_("Report bugs to <%s>.\n"), PACKAGE_BUGREPORT);
      printf (_("%s home page: <%s>.\n"), PACKAGE_NAME, PACKAGE_URL);
      fputs (_("General help using GNU software: "
               "<http://www.gnu.org/gethelp/>.\n"),
             stdout);

#if (defined __GLIBC__ && __GLIBC__ >= 2) && !defined __UCLIBC__
      /* Don't output this redundant message for English locales.
         Note we still output for 'C' so that it gets included in the
         man page.  */
      const char *lc_messages = setlocale (LC_MESSAGES, NULL);
      if (lc_messages && !STREQ (lc_messages, "en_"))
        /* TRANSLATORS: Replace LANG_CODE in this URL with your language
           code <http://translationproject.org/team/LANG_CODE.html> to
           form one of the URLs at http://translationproject.org/team/.
           Otherwise, replace the entire URL with your translation team's
           email address.  */
        fputs (_("Report translation bugs to "
                 "<http://translationproject.org/team/>.\n"), stdout);
#endif
      fputs (_("For complete documentation, run: info bison.\n"), stdout);
    }

  exit (status);
}


/*------------------------------.
| Display the version message.  |
`------------------------------*/

static void
version (void)
{
  /* Some efforts were made to ease the translators' task, please
     continue.  */
  printf (_("bison (GNU Bison) %s"), VERSION);
  putc ('\n', stdout);
  fputs (_("Written by Robert Corbett and Richard Stallman.\n"), stdout);
  putc ('\n', stdout);

  fprintf (stdout,
           _("Copyright (C) %d Free Software Foundation, Inc.\n"),
           PACKAGE_COPYRIGHT_YEAR);

  fputs (_("\
This is free software; see the source for copying conditions.  There is NO\n\
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\
"),
         stdout);
}


/*-------------------------------------.
| --skeleton and --language handling.  |
`--------------------------------------*/

void
skeleton_arg (char const *arg, int prio, location loc)
{
  if (prio < skeleton_prio)
    {
      skeleton_prio = prio;
      skeleton = arg;
    }
  else if (prio == skeleton_prio)
    complain (&loc, complaint,
              _("multiple skeleton declarations are invalid"));
}

void
language_argmatch (char const *arg, int prio, location loc)
{
  char const *msg = NULL;

  if (prio < language_prio)
    {
      for (int i = 0; valid_languages[i].language[0]; ++i)
        if (c_strcasecmp (arg, valid_languages[i].language) == 0)
          {
            language_prio = prio;
            language = &valid_languages[i];
            return;
          }
      msg = _("%s: invalid language");
    }
  else if (language_prio == prio)
    msg = _("multiple language declarations are invalid");

  if (msg)
    complain (&loc, complaint, msg, quotearg_colon (arg));
}

/*----------------------.
| Process the options.  |
`----------------------*/

/* Shorts options.
   Should be computed from long_options.  */
static char const short_options[] =
  "D:"
  "F:"
  "L:"
  "S:"
  "T::"
  "V"
  "W::"
  "b:"
  "d"
  "f::"
  "g::"
  "h"
  "k"
  "l"
  "o:"
  "p:"
  "r:"
  "t"
  "u"   /* --update */
  "v"
  "x::"
  "y"
  ;

/* Values for long options that do not have single-letter equivalents.  */
enum
{
  COLOR_OPTION = CHAR_MAX + 1,
  FIXED_OUTPUT_FILES_OPTION,
  LOCATIONS_OPTION,
  PRINT_DATADIR_OPTION,
  PRINT_LOCALEDIR_OPTION,
  REPORT_FILE_OPTION,
  STYLE_OPTION
};

/* In the same order as in usage(), and in the documentation.  */
static struct option const long_options[] =
{
  /* Operation modes. */
  { "help",            no_argument,       0,   'h' },
  { "version",         no_argument,       0,   'V' },
  { "print-localedir", no_argument,       0,   PRINT_LOCALEDIR_OPTION },
  { "print-datadir",   no_argument,       0,   PRINT_DATADIR_OPTION   },
  { "update",          no_argument,       0,   'u' },
  { "feature",         optional_argument, 0,   'f' },

  /* Diagnostics.  */
  { "warnings",        optional_argument,  0, 'W' },
  { "color",           optional_argument,  0,  COLOR_OPTION },
  { "style",           optional_argument,  0,  STYLE_OPTION },

  /* Tuning the Parser. */
  { "language",       required_argument,   0, 'L' },
  { "skeleton",       required_argument,   0, 'S' },
  { "debug",          no_argument,         0, 't' },
  { "locations",      no_argument,         0, LOCATIONS_OPTION },
  { "define",         required_argument,   0, 'D' },
  { "force-define",   required_argument,   0, 'F' },
  { "name-prefix",    required_argument,   0, 'p' },
  { "no-lines",       no_argument,         0, 'l' },
  { "token-table",    no_argument,         0, 'k' },
  { "yacc",           no_argument,         0, 'y' },

  /* Output Files. */
  { "defines",     optional_argument,   0,   'd' },
  { "report",      required_argument,   0,   'r' },
  { "report-file", required_argument,   0,   REPORT_FILE_OPTION },
  { "verbose",     no_argument,         0,   'v' },
  { "file-prefix", required_argument,   0,   'b' },
  { "output",      required_argument,   0,   'o' },
  { "graph",       optional_argument,   0,   'g' },
  { "xml",         optional_argument,   0,   'x' },

  /* Hidden. */
  { "fixed-output-files", no_argument,       0,  FIXED_OUTPUT_FILES_OPTION },
  { "output-file",        required_argument, 0,  'o' },
  { "trace",              optional_argument, 0,  'T' },

  {0, 0, 0, 0}
};

/* Build a location for the current command line argument. */
static
location
command_line_location (void)
{
  location res;
  /* "<command line>" is used in GCC's messages about -D. */
  boundary_set (&res.start, uniqstr_new ("<command line>"), optind - 1, -1, -1);
  res.end = res.start;
  return res;
}


/* Handle the command line options for color support.  Do it early, so
   that error messages from getargs be also colored as per the user's
   request.  This is consistent with the way GCC and Clang behave.  */

static void
getargs_colors (int argc, char *argv[])
{
  for (int i = 1; i < argc; i++)
    {
      const char *arg = argv[i];
      if (STRPREFIX_LIT ("--color=", arg))
        {
          const char *color = arg + strlen ("--color=");
          if (STREQ (color, "debug"))
            color_debug = true;
          else
            handle_color_option (color);
        }
      else if (STREQ ("--color", arg))
        handle_color_option (NULL);
      else if (STRPREFIX_LIT ("--style=", arg))
        {
          const char *style = arg + strlen ("--style=");
          handle_style_option (style);
        }
    }
  complain_init_color ();
}


void
getargs (int argc, char *argv[])
{
  getargs_colors (argc, argv);

  int c;
  while ((c = getopt_long (argc, argv, short_options, long_options, NULL))
         != -1)
  {
    location loc = command_line_location ();
    switch (c)
      {
        /* ASCII Sorting for short options (i.e., upper case then
           lower case), and then long-only options.  */

      case 0:
        /* Certain long options cause getopt_long to return 0.  */
        break;

      case 'D': /* -DNAME[=(VALUE|"VALUE"|{VALUE})]. */
      case 'F': /* -FNAME[=(VALUE|"VALUE"|{VALUE})]. */
        {
          char *name = optarg;
          char *value = strchr (optarg, '=');
          muscle_kind kind = muscle_keyword;
          if (value)
            {
              char *end = value + strlen (value) - 1;
              *value++ = 0;
              if (*value == '{' && *end == '}')
                {
                  kind = muscle_code;
                  ++value;
                  *end = 0;
                }
              else if (*value == '"' && *end == '"')
                {
                  kind = muscle_string;
                  ++value;
                  *end = 0;
                }
            }
          muscle_percent_define_insert (name, loc,
                                        kind, value ? value : "",
                                        c == 'D' ? MUSCLE_PERCENT_DEFINE_D
                                                 : MUSCLE_PERCENT_DEFINE_F);
        }
        break;

      case 'L':
        language_argmatch (optarg, command_line_prio, loc);
        break;

      case 'S':
        skeleton_arg (optarg, command_line_prio, loc);
        break;

      case 'T':
        FLAGS_ARGMATCH (trace, optarg, trace_all);
        break;

      case 'V':
        version ();
        exit (EXIT_SUCCESS);

      case 'f':
        FLAGS_ARGMATCH (feature, optarg, feature_all);
        break;

      case 'W':
        warnings_argmatch (optarg);
        break;

      case 'b':
        spec_file_prefix = optarg;
        break;

      case 'd':
        /* Here, the -d and --defines options are differentiated.  */
        defines_flag = true;
        if (optarg)
          {
            free (spec_header_file);
            spec_header_file = xstrdup (optarg);
          }
        break;

      case 'g':
        graph_flag = true;
        if (optarg)
          {
            free (spec_graph_file);
            spec_graph_file = xstrdup (optarg);
          }
        break;

      case 'h':
        usage (EXIT_SUCCESS);

      case 'k':
        token_table_flag = true;
        break;

      case 'l':
        no_lines_flag = true;
        break;

      case 'o':
        spec_outfile = optarg;
        break;

      case 'p':
        spec_name_prefix = optarg;
        break;

      case 'r':
        FLAGS_ARGMATCH (report, optarg, report_all);
        break;

      case 't':
        muscle_percent_define_insert ("parse.trace",
                                      loc,
                                      muscle_keyword, "",
                                      MUSCLE_PERCENT_DEFINE_D);
        break;

      case 'u':
        update_flag = true;
        feature_flag |= feature_syntax_only;
        break;

      case 'v':
        report_flag |= report_states;
        break;

      case 'x':
        xml_flag = true;
        if (optarg)
          {
            free (spec_xml_file);
            spec_xml_file = xstrdup (optarg);
          }
        break;

      case 'y':
        warning_argmatch ("yacc", 0, 0);
        yacc_loc = loc;
        break;

      case COLOR_OPTION:
        /* Handled in getargs_colors. */
        break;

      case FIXED_OUTPUT_FILES_OPTION:
        complain (&loc, Wdeprecated,
                  _("deprecated option: %s, use %s"),
                  quote ("--fixed-output-files"), quote_n (1, "-o y.tab.c"));
        spec_outfile = "y.tab.c";
        break;

      case LOCATIONS_OPTION:
        muscle_percent_define_ensure ("locations", loc, true);
        break;

      case PRINT_LOCALEDIR_OPTION:
        printf ("%s\n", LOCALEDIR);
        exit (EXIT_SUCCESS);

      case PRINT_DATADIR_OPTION:
        printf ("%s\n", pkgdatadir ());
        exit (EXIT_SUCCESS);

      case REPORT_FILE_OPTION:
        free (spec_verbose_file);
        spec_verbose_file = xstrdup (optarg);
        break;

      case STYLE_OPTION:
        /* Handled in getargs_colors. */
        break;

      default:
        usage (EXIT_FAILURE);
      }
  }

  if (argc - optind != 1)
    {
      if (argc - optind < 1)
        error (0, 0, _("missing operand"));
      else
        error (0, 0, _("extra operand %s"), quote (argv[optind + 1]));
      usage (EXIT_FAILURE);
    }

  grammar_file = uniqstr_new (argv[optind]);
  MUSCLE_INSERT_C_STRING ("file_name", grammar_file);
}

void
tr (char *s, char from, char to)
{
  for (; *s; s++)
    if (*s == from)
      *s = to;
}
