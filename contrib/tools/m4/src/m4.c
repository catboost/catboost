/* GNU m4 -- A simple macro processor

   Copyright (C) 1989-1994, 2004-2013 Free Software Foundation, Inc.

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

#include "m4.h"

#include <getopt.h>
#include <limits.h>
#include <signal.h>

#include "c-stack.h"
#include "ignore-value.h"
#include "progname.h"
#include "version-etc.h"

#ifdef DEBUG_STKOVF
# include "assert.h"
#endif

#define AUTHORS "Rene' Seindal"

static void usage (int) M4_GNUC_NORETURN;

/* Enable sync output for /lib/cpp (-s).  */
int sync_output = 0;

/* Debug (-d[flags]).  */
int debug_level = 0;

/* Hash table size (should be a prime) (-Hsize).  */
size_t hash_table_size = HASHMAX;

/* Disable GNU extensions (-G).  */
int no_gnu_extensions = 0;

/* Prefix all builtin functions by `m4_'.  */
int prefix_all_builtins = 0;

/* Max length of arguments in trace output (-lsize).  */
int max_debug_argument_length = 0;

/* Suppress warnings about missing arguments.  */
int suppress_warnings = 0;

/* If true, then warnings affect exit status.  */
static bool fatal_warnings = false;

/* If not zero, then value of exit status for warning diagnostics.  */
int warning_status = 0;

/* Artificial limit for expansion_level in macro.c.  */
int nesting_limit = 1024;

#ifdef ENABLE_CHANGEWORD
/* User provided regexp for describing m4 words.  */
const char *user_word_regexp = "";
#endif

/* Global catchall for any errors that should affect final error status, but
   where we try to continue execution in the meantime.  */
int retcode;

struct macro_definition
{
  struct macro_definition *next;
  int code; /* D, U, s, t, '\1', or DEBUGFILE_OPTION.  */
  const char *arg;
};
typedef struct macro_definition macro_definition;

/* Error handling functions.  */

/*-----------------------.
| Wrapper around error.  |
`-----------------------*/

void
m4_error (int status, int errnum, const char *format, ...)
{
  va_list args;
  va_start (args, format);
  verror_at_line (status, errnum, current_line ? current_file : NULL,
                  current_line, format, args);
  if (fatal_warnings && ! retcode)
    retcode = EXIT_FAILURE;
  va_end (args);
}

/*-------------------------------.
| Wrapper around error_at_line.  |
`-------------------------------*/

void
m4_error_at_line (int status, int errnum, const char *file, int line,
                  const char *format, ...)
{
  va_list args;
  va_start (args, format);
  verror_at_line (status, errnum, line ? file : NULL, line, format, args);
  if (fatal_warnings && ! retcode)
    retcode = EXIT_FAILURE;
  va_end (args);
}

#ifndef SIGBUS
# define SIGBUS SIGILL
#endif

#ifndef NSIG
# ifndef MAX
#  define MAX(a,b) ((a) < (b) ? (b) : (a))
# endif
# define NSIG (MAX (SIGABRT, MAX (SIGILL, MAX (SIGFPE,  \
                                               MAX (SIGSEGV, SIGBUS)))) + 1)
#endif

/* Pre-translated messages for program errors.  Do not translate in
   the signal handler, since gettext and strsignal are not
   async-signal-safe.  */
static const char * volatile program_error_message;
static const char * volatile signal_message[NSIG];

/* Print a nicer message about any programmer errors, then exit.  This
   must be aysnc-signal safe, since it is executed as a signal
   handler.  If SIGNO is zero, this represents a stack overflow; in
   that case, we return to allow c_stack_action to handle things.  */
static void M4_GNUC_PURE
fault_handler (int signo)
{
  if (signo)
    {
      /* POSIX states that reading static memory is, in general, not
         async-safe.  However, the static variables that we read are
         never modified once this handler is installed, so this
         particular usage is safe.  And it seems an oversight that
         POSIX claims strlen is not async-safe.  Ignore write
         failures, since we will exit with non-zero status anyway.  */
#define WRITE(f, b, l) ignore_value (write (f, b, l))
      WRITE (STDERR_FILENO, program_name, strlen (program_name));
      WRITE (STDERR_FILENO, ": ", 2);
      WRITE (STDERR_FILENO, program_error_message,
             strlen (program_error_message));
      if (signal_message[signo])
        {
          WRITE (STDERR_FILENO, ": ", 2);
          WRITE (STDERR_FILENO, signal_message[signo],
                 strlen (signal_message[signo]));
        }
      WRITE (STDERR_FILENO, "\n", 1);
#undef WRITE
      _exit (EXIT_INTERNAL_ERROR);
    }
}


/*---------------------------------------------.
| Print a usage message and exit with STATUS.  |
`---------------------------------------------*/

static void
usage (int status)
{
  if (status != EXIT_SUCCESS)
    xfprintf (stderr, "Try `%s --help' for more information.\n", program_name);
  else
    {
      xprintf ("Usage: %s [OPTION]... [FILE]...\n", program_name);
      fputs ("\
Process macros in FILEs.  If no FILE or if FILE is `-', standard input\n\
is read.\n\
", stdout);
      fputs ("\
\n\
Mandatory or optional arguments to long options are mandatory or optional\n\
for short options too.\n\
\n\
Operation modes:\n\
      --help                   display this help and exit\n\
      --version                output version information and exit\n\
", stdout);
      xprintf ("\
  -E, --fatal-warnings         once: warnings become errors, twice: stop\n\
                                 execution at first error\n\
  -i, --interactive            unbuffer output, ignore interrupts\n\
  -P, --prefix-builtins        force a `m4_' prefix to all builtins\n\
  -Q, --quiet, --silent        suppress some warnings for builtins\n\
      --warn-macro-sequence[=REGEXP]\n\
                               warn if macro definition matches REGEXP,\n\
                                 default %s\n\
", DEFAULT_MACRO_SEQUENCE);
#ifdef ENABLE_CHANGEWORD
      fputs ("\
  -W, --word-regexp=REGEXP     use REGEXP for macro name syntax\n\
", stdout);
#endif
      fputs ("\
\n\
Preprocessor features:\n\
  -D, --define=NAME[=VALUE]    define NAME as having VALUE, or empty\n\
  -I, --include=DIRECTORY      append DIRECTORY to include path\n\
  -s, --synclines              generate `#line NUM \"FILE\"' lines\n\
  -U, --undefine=NAME          undefine NAME\n\
", stdout);
      puts ("");
      xprintf (_("\
Limits control:\n\
  -g, --gnu                    override -G to re-enable GNU extensions\n\
  -G, --traditional            suppress all GNU extensions\n\
  -H, --hashsize=PRIME         set symbol lookup hash table size [509]\n\
  -L, --nesting-limit=NUMBER   change nesting limit, 0 for unlimited [%d]\n\
"), nesting_limit);
      puts ("");
      fputs ("\
Frozen state files:\n\
  -F, --freeze-state=FILE      produce a frozen state on FILE at end\n\
  -R, --reload-state=FILE      reload a frozen state from FILE at start\n\
", stdout);
      fputs ("\
\n\
Debugging:\n\
  -d, --debug[=FLAGS]          set debug level (no FLAGS implies `aeq')\n\
      --debugfile[=FILE]       redirect debug and trace output to FILE\n\
                                 (default stderr, discard if empty string)\n\
  -l, --arglength=NUM          restrict macro tracing size\n\
  -t, --trace=NAME             trace NAME when it is defined\n\
", stdout);
      fputs ("\
\n\
FLAGS is any of:\n\
  a   show actual arguments\n\
  c   show before collect, after collect and after call\n\
  e   show expansion\n\
  f   say current input file name\n\
  i   show changes in input files\n\
  l   say current input line number\n\
  p   show results of path searches\n\
  q   quote values as necessary, with a or e flag\n\
  t   trace for all macro calls, not only traceon'ed\n\
  x   add a unique macro call id, useful with c flag\n\
  V   shorthand for all of the above flags\n\
", stdout);
      fputs ("\
\n\
If defined, the environment variable `M4PATH' is a colon-separated list\n\
of directories included after any specified by `-I'.\n\
", stdout);
      fputs ("\
\n\
Exit status is 0 for success, 1 for failure, 63 for frozen file version\n\
mismatch, or whatever value was passed to the m4exit macro.\n\
", stdout);
      emit_bug_reporting_address ();
    }
  exit (status);
}

/*--------------------------------------.
| Decode options and launch execution.  |
`--------------------------------------*/

/* For long options that have no equivalent short option, use a
   non-character as a pseudo short option, starting with CHAR_MAX + 1.  */
enum
{
  DEBUGFILE_OPTION = CHAR_MAX + 1,      /* no short opt */
  DIVERSIONS_OPTION,                    /* not quite -N, because of message */
  WARN_MACRO_SEQUENCE_OPTION,           /* no short opt */

  HELP_OPTION,                          /* no short opt */
  VERSION_OPTION                        /* no short opt */
};

static const struct option long_options[] =
{
  {"arglength", required_argument, NULL, 'l'},
  {"debug", optional_argument, NULL, 'd'},
  {"define", required_argument, NULL, 'D'},
  {"error-output", required_argument, NULL, 'o'}, /* FIXME: deprecate in 2.0 */
  {"fatal-warnings", no_argument, NULL, 'E'},
  {"freeze-state", required_argument, NULL, 'F'},
  {"gnu", no_argument, NULL, 'g'},
  {"hashsize", required_argument, NULL, 'H'},
  {"include", required_argument, NULL, 'I'},
  {"interactive", no_argument, NULL, 'i'},
  {"nesting-limit", required_argument, NULL, 'L'},
  {"prefix-builtins", no_argument, NULL, 'P'},
  {"quiet", no_argument, NULL, 'Q'},
  {"reload-state", required_argument, NULL, 'R'},
  {"silent", no_argument, NULL, 'Q'},
  {"synclines", no_argument, NULL, 's'},
  {"trace", required_argument, NULL, 't'},
  {"traditional", no_argument, NULL, 'G'},
  {"undefine", required_argument, NULL, 'U'},
  {"word-regexp", required_argument, NULL, 'W'},

  {"debugfile", optional_argument, NULL, DEBUGFILE_OPTION},
  {"diversions", required_argument, NULL, DIVERSIONS_OPTION},
  {"warn-macro-sequence", optional_argument, NULL, WARN_MACRO_SEQUENCE_OPTION},

  {"help", no_argument, NULL, HELP_OPTION},
  {"version", no_argument, NULL, VERSION_OPTION},

  { NULL, 0, NULL, 0 },
};

/* Process a command line file NAME, and return true only if it was
   stdin.  */
static void
process_file (const char *name)
{
  if (STREQ (name, "-"))
    {
      /* If stdin is a terminal, we want to allow 'm4 - file -'
         to read input from stdin twice, like GNU cat.  Besides,
         there is no point closing stdin before wrapped text, to
         minimize bugs in syscmd called from wrapped text.  */
      push_file (stdin, "stdin", false);
    }
  else
    {
      char *full_name;
      FILE *fp = m4_path_search (name, &full_name);
      if (fp == NULL)
        {
          error (0, errno, _("cannot open `%s'"), name);
          /* Set the status to EXIT_FAILURE, even though we
             continue to process files after a missing file.  */
          retcode = EXIT_FAILURE;
          return;
        }
      push_file (fp, full_name, true);
      free (full_name);
    }
  expand_input ();
}

/* POSIX requires only -D, -U, and -s; and says that the first two
   must be recognized when interspersed with file names.  Traditional
   behavior also handles -s between files.  Starting OPTSTRING with
   '-' forces getopt_long to hand back file names as arguments to opt
   '\1', rather than reordering the command line.  */
#ifdef ENABLE_CHANGEWORD
#define OPTSTRING "-B:D:EF:GH:I:L:N:PQR:S:T:U:W:d::egil:o:st:"
#else
#define OPTSTRING "-B:D:EF:GH:I:L:N:PQR:S:T:U:d::egil:o:st:"
#endif

int
main (int argc, char *const *argv)
{
#if !defined(_WIN32) && !defined(_WIN64)
  struct sigaction act;
#endif
  macro_definition *head;       /* head of deferred argument list */
  macro_definition *tail;
  macro_definition *defn;
  int optchar;                  /* option character */

  macro_definition *defines;
  bool interactive = false;
  bool seen_file = false;
  const char *debugfile = NULL;
  const char *frozen_file_to_read = NULL;
  const char *frozen_file_to_write = NULL;
  const char *macro_sequence = "";

  set_program_name (argv[0]);
  retcode = EXIT_SUCCESS;
  atexit (close_stdin);

  include_init ();
  debug_init ();

  /* Stack overflow and program error handling.  Ignore failure to
     install a handler, since this is merely for improved output on
     crash, and we should never crash ;).  We install SIGBUS and
     SIGSEGV handlers prior to using the c-stack module; depending on
     the platform, c-stack will then override none, SIGSEGV, or both
     handlers.  */
  program_error_message
    = xasprintf (_("internal error detected; please report this bug to <%s>"),
                 PACKAGE_BUGREPORT);
  signal_message[SIGSEGV] = xstrdup (strsignal (SIGSEGV));
  signal_message[SIGABRT] = xstrdup (strsignal (SIGABRT));
  signal_message[SIGILL] = xstrdup (strsignal (SIGILL));
  signal_message[SIGFPE] = xstrdup (strsignal (SIGFPE));
  if (SIGBUS != SIGILL && SIGBUS != SIGSEGV)
    signal_message[SIGBUS] = xstrdup (strsignal (SIGBUS));
#if !defined(_WIN32) && !defined(_WIN64)
  // No such signals on Windows
  sigemptyset(&act.sa_mask);
  /* One-shot - if we fault while handling a fault, we want to revert
     to default signal behavior.  */
  act.sa_flags = SA_NODEFER | SA_RESETHAND;
  act.sa_handler = fault_handler;
  sigaction (SIGSEGV, &act, NULL);
  sigaction (SIGABRT, &act, NULL);
  sigaction (SIGILL, &act, NULL);
  sigaction (SIGFPE, &act, NULL);
  sigaction (SIGBUS, &act, NULL);
#endif
  if (c_stack_action (fault_handler) == 0)
    nesting_limit = 0;

#ifdef DEBUG_STKOVF
  /* Make it easier to test our fault handlers.  Exporting M4_CRASH=0
     attempts a SIGSEGV, exporting it as 1 attempts an assertion
     failure with a fallback to abort.  */
  {
    char *crash = getenv ("M4_CRASH");
    if (crash)
      {
        if (!strtol (crash, NULL, 10))
          ++*(int *) 8;
        assert (false);
        abort ();
      }
  }
#endif /* DEBUG_STKOVF */

  /* First, we decode the arguments, to size up tables and stuff.  */
  head = tail = NULL;

  while ((optchar = getopt_long (argc, (char **) argv, OPTSTRING,
                                 long_options, NULL)) != -1)
    switch (optchar)
      {
      default:
        usage (EXIT_FAILURE);

      case 'B':
      case 'S':
      case 'T':
        /* Compatibility junk: options that other implementations
           support, but which we ignore as no-ops and don't list in
           --help.  */
        error (0, 0, _("warning: `m4 -%c' may be removed in a future release"),
               optchar);
        break;

      case 'N':
      case DIVERSIONS_OPTION:
        /* -N became an obsolete no-op in 1.4.x.  */
        error (0, 0, _("warning: `m4 %s' is deprecated"),
               optchar == 'N' ? "-N" : "--diversions");
        break;

      case 'D':
      case 'U':
      case 's':
      case 't':
      case '\1':
      case DEBUGFILE_OPTION:
        /* Arguments that cannot be handled until later are accumulated.  */

        defn = (macro_definition *) xmalloc (sizeof (macro_definition));
        defn->code = optchar;
        defn->arg = optarg;
        defn->next = NULL;

        if (head == NULL)
          head = defn;
        else
          tail->next = defn;
        tail = defn;

        break;

      case 'E':
        if (! fatal_warnings)
          fatal_warnings = true;
        else
          warning_status = EXIT_FAILURE;
        break;

      case 'F':
        frozen_file_to_write = optarg;
        break;

      case 'G':
        no_gnu_extensions = 1;
        break;

      case 'H':
        hash_table_size = strtol (optarg, NULL, 10);
        if (hash_table_size == 0)
          hash_table_size = HASHMAX;
        break;

      case 'I':
        add_include_directory (optarg);
        break;

      case 'L':
        nesting_limit = strtol (optarg, NULL, 10);
        break;

      case 'P':
        prefix_all_builtins = 1;
        break;

      case 'Q':
        suppress_warnings = 1;
        break;

      case 'R':
        frozen_file_to_read = optarg;
        break;

#ifdef ENABLE_CHANGEWORD
      case 'W':
        user_word_regexp = optarg;
        break;
#endif

      case 'd':
        debug_level = debug_decode (optarg);
        if (debug_level < 0)
          {
            error (0, 0, _("bad debug flags: `%s'"), optarg);
            debug_level = 0;
          }
        break;

      case 'e':
        error (0, 0, _("warning: `m4 -e' is deprecated, use `-i' instead"));
        /* fall through */
      case 'i':
        interactive = true;
        break;

      case 'g':
        no_gnu_extensions = 0;
        break;

      case 'l':
        max_debug_argument_length = strtol (optarg, NULL, 10);
        if (max_debug_argument_length <= 0)
          max_debug_argument_length = 0;
        break;

      case 'o':
        /* -o/--error-output are deprecated synonyms of --debugfile,
           but don't issue a deprecation warning until autoconf 2.61
           or later is more widely established, as such a warning
           would interfere with all earlier versions of autoconf.  */
        /* Don't call debug_set_output here, as it has side effects.  */
        debugfile = optarg;
        break;

      case WARN_MACRO_SEQUENCE_OPTION:
         /* Don't call set_macro_sequence here, as it can exit.
            --warn-macro-sequence sets optarg to NULL (which uses the
            default regexp); --warn-macro-sequence= sets optarg to ""
            (which disables these warnings).  */
        macro_sequence = optarg;
        break;

      case VERSION_OPTION:
        version_etc (stdout, PACKAGE, PACKAGE_NAME, VERSION, AUTHORS, NULL);
        exit (EXIT_SUCCESS);
        break;

      case HELP_OPTION:
        usage (EXIT_SUCCESS);
        break;
      }

  defines = head;

  /* Do the basic initializations.  */
  if (debugfile && !debug_set_output (debugfile))
    M4ERROR ((warning_status, errno, "cannot set debug file `%s'", debugfile));

  input_init ();
  output_init ();
  symtab_init ();
  set_macro_sequence (macro_sequence);
  include_env_init ();

  if (frozen_file_to_read)
    reload_frozen_state (frozen_file_to_read);
  else
    builtin_init ();

  /* Interactive mode means unbuffered output, and interrupts ignored.  */

  if (interactive)
    {
      signal (SIGINT, SIG_IGN);
      setbuf (stdout, (char *) NULL);
    }

  /* Handle deferred command line macro definitions.  Must come after
     initialization of the symbol table.  */

  while (defines != NULL)
    {
      macro_definition *next;
      symbol *sym;

      switch (defines->code)
        {
        case 'D':
          {
            /* defines->arg is read-only, so we need a copy.  */
            char *macro_name = xstrdup (defines->arg);
            char *macro_value = strchr (macro_name, '=');
            if (macro_value)
              *macro_value++ = '\0';
            define_user_macro (macro_name, macro_value, SYMBOL_INSERT);
            free (macro_name);
          }
          break;

        case 'U':
          lookup_symbol (defines->arg, SYMBOL_DELETE);
          break;

        case 't':
          sym = lookup_symbol (defines->arg, SYMBOL_INSERT);
          SYMBOL_TRACED (sym) = true;
          break;

        case 's':
          sync_output = 1;
          break;

        case '\1':
          seen_file = true;
          process_file (defines->arg);
          break;

        case DEBUGFILE_OPTION:
          if (!debug_set_output (defines->arg))
            M4ERROR ((warning_status, errno, "cannot set debug file `%s'",
                      debugfile ? debugfile : _("stderr")));
          break;

        default:
          M4ERROR ((0, 0, "INTERNAL ERROR: bad code in deferred arguments"));
          abort ();
        }

      next = defines->next;
      free (defines);
      defines = next;
    }

  /* Handle remaining input files.  Each file is pushed on the input,
     and the input read.  Wrapup text is handled separately later.  */

  if (optind == argc && !seen_file)
    process_file ("-");
  else
    for (; optind < argc; optind++)
      process_file (argv[optind]);

  /* Now handle wrapup text.  */

  while (pop_wrapup ())
    expand_input ();

  /* Change debug stream back to stderr, to force flushing the debug
     stream and detect any errors it might have encountered.  The
     three standard streams are closed by close_stdin.  */
  debug_set_output (NULL);

  if (frozen_file_to_write)
    produce_frozen_state (frozen_file_to_write);
  else
    {
      make_diversion (0);
      undivert_all ();
    }
  output_exit ();
  free_macro_sequence ();
  exit (retcode);
}
