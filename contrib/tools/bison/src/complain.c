/* Declaration for error-reporting function for Bison.

   Copyright (C) 2000-2002, 2004-2006, 2009-2015, 2018-2020 Free
   Software Foundation, Inc.

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

/* Based on error.c and error.h,
   written by David MacKenzie <djm@gnu.ai.mit.edu>.  */

#include <config.h>
#include "system.h"

#include <argmatch.h>
#include <progname.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <textstyle.h>

#include "complain.h"
#include "files.h"
#include "fixits.h"
#include "getargs.h"
#include "quote.h"

err_status complaint_status = status_none;

bool warnings_are_errors = false;

/** Whether -Werror/-Wno-error was applied to a warning.  */
typedef enum
  {
    errority_unset = 0,     /** No explicit status.  */
    errority_disabled = 1,  /** Explicitly disabled with -Wno-error=foo.  */
    errority_enabled = 2    /** Explicitly enabled with -Werror=foo. */
  } errority;

/** For each warning type, its errority.  */
static errority errority_flag[warnings_size];

/** Diagnostics severity.  */
typedef enum
  {
    severity_disabled = 0, /**< Explicitly disabled via -Wno-foo.  */
    severity_unset = 1,    /**< Unspecified status.  */
    severity_warning = 2,  /**< A warning.  */
    severity_error = 3,    /**< An error (continue, but die soon).  */
    severity_fatal = 4     /**< Fatal error (die now).  */
  } severity;


/** For each warning type, its severity.  */
static severity warnings_flag[warnings_size];

styled_ostream_t errstream = NULL;

void
begin_use_class (const char *s, FILE *out)
{
  if (out == stderr)
    {
      if (color_debug)
        fprintf (out, "<%s>", s);
      else
        {
          styled_ostream_begin_use_class (errstream, s);
          styled_ostream_flush_to_current_style (errstream);
        }
    }
}

void
end_use_class (const char *s, FILE *out)
{
  if (out == stderr)
    {
      if (color_debug)
        fprintf (out, "</%s>", s);
      else
        {
          styled_ostream_end_use_class (errstream, s);
          styled_ostream_flush_to_current_style (errstream);
        }
    }
}

void
flush (FILE *out)
{
  if (out == stderr)
    ostream_flush (errstream, FLUSH_THIS_STREAM);
  fflush (out);
}

/*------------------------.
| --warnings's handling.  |
`------------------------*/

ARGMATCH_DEFINE_GROUP (warning, warnings)

static const argmatch_warning_doc argmatch_warning_docs[] =
{
  { "conflicts-sr",     N_("S/R conflicts (enabled by default)") },
  { "conflicts-rr",     N_("R/R conflicts (enabled by default)") },
  { "dangling-alias",   N_("string aliases not attached to a symbol") },
  { "deprecated",       N_("obsolete constructs") },
  { "empty-rule",       N_("empty rules without %empty") },
  { "midrule-values",   N_("unset or unused midrule values") },
  { "precedence",       N_("useless precedence and associativity") },
  { "yacc",             N_("incompatibilities with POSIX Yacc") },
  { "other",            N_("all other warnings (enabled by default)") },
  { "all",              N_("all the warnings except 'dangling-alias' and 'yacc'") },
  { "no-CATEGORY",      N_("turn off warnings in CATEGORY") },
  { "none",             N_("turn off all the warnings") },
  { "error[=CATEGORY]", N_("treat warnings as errors") },
  { NULL, NULL }
};

static const argmatch_warning_arg argmatch_warning_args[] =
{
  { "all",            Wall },
  { "conflicts-rr",   Wconflicts_rr },
  { "conflicts-sr",   Wconflicts_sr },
  { "dangling-alias", Wdangling_alias },
  { "deprecated",     Wdeprecated },
  { "empty-rule",     Wempty_rule },
  { "everything",     Weverything },
  { "midrule-values", Wmidrule_values },
  { "none",           Wnone },
  { "other",          Wother },
  { "precedence",     Wprecedence },
  { "yacc",           Wyacc },
  { NULL, Wnone }
};

const argmatch_warning_group_type argmatch_warning_group =
{
  argmatch_warning_args,
  argmatch_warning_docs,
  N_("Warning categories include:"),
  NULL
};

void
warning_usage (FILE *out)
{
  argmatch_warning_usage (out);
}

void
warning_argmatch (char const *arg, size_t no, size_t err)
{
  int value = *argmatch_warning_value ("--warning", arg + no + err);

  /* -Wnone == -Wno-everything, and -Wno-none == -Weverything.  */
  if (!value)
    {
      value = Weverything;
      no = !no;
    }

  for (size_t b = 0; b < warnings_size; ++b)
    if (value & 1 << b)
      {
        if (err && no)
          /* -Wno-error=foo.  */
          errority_flag[b] = errority_disabled;
        else if (err && !no)
          {
            /* -Werror=foo: enables -Wfoo. */
            errority_flag[b] = errority_enabled;
            warnings_flag[b] = severity_warning;
          }
        else if (no)
          /* -Wno-foo.  */
          warnings_flag[b] = severity_disabled;
        else
          /* -Wfoo. */
          warnings_flag[b] = severity_warning;
      }
}

/** Decode a comma-separated list of arguments from -W.
 *
 *  \param args     comma separated list of effective subarguments to decode.
 *                  If 0, then activate all the flags.
 */

void
warnings_argmatch (char *args)
{
  if (!args)
    warning_argmatch ("all", 0, 0);
  else if (STREQ (args, "help"))
    {
      warning_usage (stdout);
      exit (EXIT_SUCCESS);
    }
  else
    for (args = strtok (args, ","); args; args = strtok (NULL, ","))
      if (STREQ (args, "error"))
        warnings_are_errors = true;
      else if (STREQ (args, "no-error"))
        warnings_are_errors = false;
      else
        {
          /* The length of the possible 'no-' prefix: 3, or 0.  */
          size_t no = STRPREFIX_LIT ("no-", args) ? 3 : 0;
          /* The length of the possible 'error=' (possibly after
             'no-') prefix: 6, or 0. */
          size_t err = STRPREFIX_LIT ("error=", args + no) ? 6 : 0;

          warning_argmatch (args, no, err);
        }
}

/* Color style for this type of message.  */
static const char*
severity_style (severity s)
{
  switch (s)
    {
    case severity_disabled:
    case severity_unset:
      return "note";
    case severity_warning:
      return "warning";
    case severity_error:
    case severity_fatal:
      return "error";
    }
  abort ();
}

/* Prefix for this type of message.  */
static const char*
severity_prefix (severity s)
{
  switch (s)
    {
    case severity_disabled:
    case severity_unset:
      return "";
    case severity_warning:
      return _("warning");
    case severity_error:
      return  _("error");
    case severity_fatal:
      return _("fatal error");
    }
  abort ();
}


/*-----------.
| complain.  |
`-----------*/

void
complain_init_color (void)
{
#if HAVE_LIBTEXTSTYLE
  if (color_mode == color_yes
      || color_mode == color_html
      || (color_mode == color_tty && isatty (STDERR_FILENO)))
    {
      style_file_prepare ("BISON_STYLE", "BISON_STYLEDIR", pkgdatadir (),
                          "bison-default.css");
      /* As a fallback, use the default in the current directory.  */
      struct stat statbuf;
      if ((style_file_name == NULL || stat (style_file_name, &statbuf) < 0)
          && stat ("bison-default.css", &statbuf) == 0)
        style_file_name = "bison-default.css";
    }
  else
    /* No styling.  */
    style_file_name = NULL;
#endif

  /* Workaround clang's warning (starting at Clang 3.5) about the stub
     code of html_styled_ostream_create:

     | src/complain.c:274:7: error: code will never be executed [-Werror,-Wunreachable-code]
     |     ? html_styled_ostream_create (file_ostream_create (stderr),
     |       ^~~~~~~~~~~~~~~~~~~~~~~~~~ */
#if defined __clang__
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wunreachable-code"
#endif
  errstream =
    color_mode == color_html
    ? html_styled_ostream_create (file_ostream_create (stderr),
                                  style_file_name)
    : styled_ostream_create (STDERR_FILENO, "(stderr)", TTYCTL_AUTO,
                             style_file_name);
#if defined __clang__
# pragma clang diagnostic pop
#endif
}

void
complain_init (void)
{
  caret_init ();

  warnings warnings_default =
    Wconflicts_sr | Wconflicts_rr | Wdeprecated | Wother;

  for (size_t b = 0; b < warnings_size; ++b)
    {
      warnings_flag[b] = (1 << b & warnings_default
                          ? severity_warning
                          : severity_unset);
      errority_flag[b] = errority_unset;
    }
}

void
complain_free (void)
{
  caret_free ();
  styled_ostream_free (errstream);
}

/* A diagnostic with FLAGS is about to be issued.  With what severity?
   (severity_fatal, severity_error, severity_disabled, or
   severity_warning.) */

static severity
warning_severity (warnings flags)
{
  if (flags & fatal)
    /* Diagnostics about fatal errors.  */
    return severity_fatal;
  else if (flags & complaint)
    /* Diagnostics about errors.  */
    return severity_error;
  else
    {
      /* Diagnostics about warnings.  */
      severity res = severity_disabled;
      for (size_t b = 0; b < warnings_size; ++b)
        if (flags & 1 << b)
          {
            res = res < warnings_flag[b] ? warnings_flag[b] : res;
            /* If the diagnostic is enabled, and -Werror is enabled,
               and -Wno-error=foo was not explicitly requested, this
               is an error. */
            if (res == severity_warning
                && (errority_flag[b] == errority_enabled
                    || (warnings_are_errors
                        && errority_flag[b] != errority_disabled)))
              res = severity_error;
          }
      return res;
    }
}

bool
warning_is_unset (warnings flags)
{
  for (size_t b = 0; b < warnings_size; ++b)
    if (flags & 1 << b && warnings_flag[b] != severity_unset)
      return false;
  return true;
}

bool
warning_is_enabled (warnings flags)
{
  return severity_warning <= warning_severity (flags);
}

/** Display a "[-Wyacc]" like message on \a out.  */

static void
warnings_print_categories (warnings warn_flags, FILE *out)
{
  for (int wbit = 0; wbit < warnings_size; ++wbit)
    if (warn_flags & (1 << wbit))
      {
        warnings w = 1 << wbit;
        severity s = warning_severity (w);
        const char* style = severity_style (s);
        fputs (" [", out);
        begin_use_class (style, out);
        fprintf (out, "-W%s%s",
                 s == severity_error ? "error=" : "",
                 argmatch_warning_argument (&w));
        end_use_class (style, out);
        fputc (']', out);
        /* Display only the first match, the second is "-Wall".  */
        return;
      }
}

/** Report an error message.
 *
 * \param loc     the location, defaulting to the current file,
 *                or the program name.
 * \param indent  optional indentation for the error message.
 * \param flags   the category for this message.
 * \param sever   to decide the prefix to put before the message
 *                (e.g., "warning").
 * \param message the error message, a printf format string.  Iff it
 *                ends with ": ", then no trailing newline is printed,
 *                and the caller should print the remaining
 *                newline-terminated message to stderr.
 * \param args    the arguments of the format string.
 */
static
void
error_message (const location *loc, int *indent, warnings flags,
               severity sever, const char *message, va_list args)
{
  int pos = 0;

  if (loc)
    pos += location_print (*loc, stderr);
  else
    pos += fprintf (stderr, "%s", grammar_file ? grammar_file : program_name);
  pos += fprintf (stderr, ": ");

  if (indent)
    {
      if (*indent)
        sever = severity_disabled;
      if (!*indent)
        *indent = pos;
      else if (*indent > pos)
        fprintf (stderr, "%*s", *indent - pos, "");
    }

  const char* style = severity_style (sever);

  if (sever != severity_disabled)
    {
      begin_use_class (style, stderr);
      fprintf (stderr, "%s:", severity_prefix (sever));
      end_use_class (style, stderr);
      fputc (' ', stderr);
    }

  vfprintf (stderr, message, args);
  /* Print the type of warning, only if this is not a sub message
     (in which case the prefix is null).  */
  if (! (flags & silent) && sever != severity_disabled)
    warnings_print_categories (flags, stderr);

  {
    size_t l = strlen (message);
    if (l < 2 || message[l - 2] != ':' || message[l - 1] != ' ')
      {
        putc ('\n', stderr);
        flush (stderr);
        if (loc && feature_flag & feature_caret && !(flags & no_caret))
          location_caret (*loc, style, stderr);
      }
  }
  flush (stderr);
}

/** Raise a complaint (fatal error, error or just warning).  */

static void
complains (const location *loc, int *indent, warnings flags,
           const char *message, va_list args)
{
  severity s = warning_severity (flags);
  if ((flags & complaint) && complaint_status < status_complaint)
    complaint_status = status_complaint;

  if (severity_warning <= s)
    {
      if (severity_error <= s && ! complaint_status)
        complaint_status = status_warning_as_error;
      error_message (loc, indent, flags, s, message, args);
    }

  if (flags & fatal)
    exit (EXIT_FAILURE);
}

void
complain (location const *loc, warnings flags, const char *message, ...)
{
  va_list args;
  va_start (args, message);
  complains (loc, NULL, flags, message, args);
  va_end (args);
}

void
complain_indent (location const *loc, warnings flags, int *indent,
                 const char *message, ...)
{
  va_list args;
  va_start (args, message);
  complains (loc, indent, flags, message, args);
  va_end (args);
}

void
complain_args (location const *loc, warnings w, int *indent,
               int argc, char *argv[])
{
  switch (argc)
  {
  case 1:
    complain_indent (loc, w, indent, "%s", _(argv[0]));
    break;
  case 2:
    complain_indent (loc, w, indent, _(argv[0]), argv[1]);
    break;
  case 3:
    complain_indent (loc, w, indent, _(argv[0]), argv[1], argv[2]);
    break;
  case 4:
    complain_indent (loc, w, indent, _(argv[0]), argv[1], argv[2], argv[3]);
    break;
  case 5:
    complain_indent (loc, w, indent, _(argv[0]), argv[1], argv[2], argv[3],
                     argv[4]);
    break;
  default:
    complain (loc, fatal, "too many arguments for complains");
    break;
  }
}


void
bison_directive (location const *loc, char const *directive)
{
  complain (loc, Wyacc,
            _("POSIX Yacc does not support %s"), directive);
}

void
deprecated_directive (location const *loc, char const *old, char const *upd)
{
  if (warning_is_enabled (Wdeprecated))
    {
      complain (loc, Wdeprecated,
                _("deprecated directive: %s, use %s"),
                quote (old), quote_n (1, upd));
      if (feature_flag & feature_caret)
        location_caret_suggestion (*loc, upd, stderr);
      /* Register updates only if -Wdeprecated is enabled.  */
      fixits_register (loc, upd);
    }
}

void
duplicate_directive (char const *directive,
                     location first, location second)
{
  int i = 0;
  if (feature_flag & feature_caret)
    complain_indent (&second, Wother, &i, _("duplicate directive"));
  else
    complain_indent (&second, Wother, &i, _("duplicate directive: %s"), quote (directive));
  i += SUB_INDENT;
  complain_indent (&first, Wother, &i, _("previous declaration"));
  fixits_register (&second, "");
}

void
duplicate_rule_directive (char const *directive,
                          location first, location second)
{
  int i = 0;
  complain_indent (&second, complaint, &i,
                   _("only one %s allowed per rule"), directive);
  i += SUB_INDENT;
  complain_indent (&first, complaint, &i,
                   _("previous declaration"));
  fixits_register (&second, "");
}
