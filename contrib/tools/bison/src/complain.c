/* Declaration for error-reporting function for Bison.

   Copyright (C) 2000-2002, 2004-2006, 2009-2013 Free Software
   Foundation, Inc.

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
#include <stdarg.h>
#include <progname.h>

#include "complain.h"
#include "files.h"
#include "getargs.h"
#include "quote.h"

err_status complaint_status = status_none;

bool warnings_are_errors = false;

/** Diagnostics severity.  */
typedef enum
  {
    severity_disabled = 0,
    severity_unset = 1,
    severity_warning = 2,
    severity_error = 3,
    severity_fatal = 4
  } severity;


/** For each warning type, its severity.  */
static severity warnings_flag[warnings_size];

static unsigned *indent_ptr = 0;

/*------------------------.
| --warnings's handling.  |
`------------------------*/

static const char * const warnings_args[] =
{
  "none",
  "midrule-values",
  "yacc",
  "conflicts-sr",
  "conflicts-rr",
  "deprecated",
  "empty-rule",
  "precedence",
  "other",
  "all",
  "error",
  "everything",
  0
};

static const int warnings_types[] =
{
  Wnone,
  Wmidrule_values,
  Wyacc,
  Wconflicts_sr,
  Wconflicts_rr,
  Wdeprecated,
  Wempty_rule,
  Wprecedence,
  Wother,
  Wall,
  Werror,
  Weverything
};

ARGMATCH_VERIFY (warnings_args, warnings_types);

void
warning_argmatch (char const *arg, size_t no, size_t err)
{
  int value = XARGMATCH ("--warning", arg + no + err,
                         warnings_args, warnings_types);

  /* -Wnone == -Wno-everything, and -Wno-none == -Weverything.  */
  if (!value)
    {
      value = Weverything;
      no = !no;
    }

  if (no)
    {
      size_t b;
      for (b = 0; b < warnings_size; ++b)
        if (value & 1 << b)
          {
            if (err)
              {
                /* -Wno-error=foo: if foo enabled as an error,
                   make it a warning.  */
                if (warnings_flag[b] == severity_error)
                  warnings_flag[b] = severity_warning;
              }
            else
              /* -Wno-foo.  */
              warnings_flag[b] = severity_disabled;
          }
    }
  else
    {
      size_t b;
      for (b = 0; b < warnings_size; ++b)
        if (value & 1 << b)
          /* -Wfoo and -Werror=foo. */
          warnings_flag[b] = err ? severity_error : severity_warning;
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
  if (args)
    for (args = strtok (args, ","); args; args = strtok (NULL, ","))
      if (STREQ (args, "error"))
        warnings_are_errors = true;
      else if (STREQ (args, "no-error"))
        {
          warnings_are_errors = false;
          warning_argmatch ("no-error=everything", 3, 6);
        }
      else
        {
          size_t no = STRPREFIX_LIT ("no-", args) ? 3 : 0;
          size_t err = STRPREFIX_LIT ("error=", args + no) ? 6 : 0;

          warning_argmatch (args, no, err);
        }
  else
    warning_argmatch ("all", 0, 0);
}


/*-----------.
| complain.  |
`-----------*/

void
complain_init (void)
{
  warnings warnings_default =
    Wconflicts_sr | Wconflicts_rr | Wdeprecated | Wother;

  size_t b;
  for (b = 0; b < warnings_size; ++b)
    warnings_flag[b] = (1 << b & warnings_default
                        ? severity_warning
                        : severity_unset);
}

static severity
warning_severity (warnings flags)
{
  if (flags & fatal)
    return severity_fatal;
  else if (flags & complaint)
    return severity_error;
  else
    {
      severity res = severity_disabled;
      size_t b;
      for (b = 0; b < warnings_size; ++b)
        if (flags & 1 << b)
          res = res < warnings_flag[b] ? warnings_flag[b] : res;
      if (res == severity_warning && warnings_are_errors)
        res = severity_error;
      return res;
    }
}

bool
warning_is_unset (warnings flags)
{
  size_t b;
  for (b = 0; b < warnings_size; ++b)
    if (flags & 1 << b && warnings_flag[b] != severity_unset)
      return false;
  return true;
}

/** Display a "[-Wyacc]" like message on \a f.  */

static void
warnings_print_categories (warnings warn_flags, FILE *f)
{
  /* Display only the first match, the second is "-Wall".  */
  size_t i;
  for (i = 0; warnings_args[i]; ++i)
    if (warn_flags & warnings_types[i])
      {
        severity s = warning_severity (warnings_types[i]);
        fprintf (f, " [-W%s%s]",
                 s == severity_error ? "error=" : "",
                 warnings_args[i]);
        return;
      }
}

/** Report an error message.
 *
 * \param loc     the location, defaulting to the current file,
 *                or the program name.
 * \param flags   the category for this message.
 * \param prefix  put before the message (e.g., "warning").
 * \param message the error message, a printf format string.  Iff it
 *                ends with ": ", then no trailing newline is printed,
 *                and the caller should print the remaining
 *                newline-terminated message to stderr.
 * \param args    the arguments of the format string.
 */
static
void
error_message (const location *loc, warnings flags, const char *prefix,
               const char *message, va_list args)
{
  unsigned pos = 0;

  if (loc)
    pos += location_print (*loc, stderr);
  else
    pos += fprintf (stderr, "%s", current_file ? current_file : program_name);
  pos += fprintf (stderr, ": ");

  if (indent_ptr)
    {
      if (*indent_ptr)
        prefix = NULL;
      if (!*indent_ptr)
        *indent_ptr = pos;
      else if (*indent_ptr > pos)
        fprintf (stderr, "%*s", *indent_ptr - pos, "");
      indent_ptr = 0;
    }

  if (prefix)
    fprintf (stderr, "%s: ", prefix);

  vfprintf (stderr, message, args);
  if (! (flags & silent))
    warnings_print_categories (flags, stderr);
  {
    size_t l = strlen (message);
    if (l < 2 || message[l - 2] != ':' || message[l - 1] != ' ')
      {
        putc ('\n', stderr);
        fflush (stderr);
        if (loc && feature_flag & feature_caret && !(flags & no_caret))
          location_caret (*loc, stderr);
      }
  }
  fflush (stderr);
}

/** Raise a complaint. That can be a fatal error, an error or just a
    warning.  */

static void
complains (const location *loc, warnings flags, const char *message,
           va_list args)
{
  severity s = warning_severity (flags);
  if ((flags & complaint) && complaint_status < status_complaint)
    complaint_status = status_complaint;

  if (severity_warning <= s)
    {
      const char* prefix =
        s == severity_fatal ? _("fatal error")
        : s == severity_error ? _("error")
        : _("warning");
      if (severity_error <= s && ! complaint_status)
        complaint_status = status_warning_as_error;
      error_message (loc, flags, prefix, message, args);
    }

  if (flags & fatal)
    exit (EXIT_FAILURE);
}

void
complain (location const *loc, warnings flags, const char *message, ...)
{
  va_list args;
  va_start (args, message);
  complains (loc, flags, message, args);
  va_end (args);
}

void
complain_indent (location const *loc, warnings flags, unsigned *indent,
                 const char *message, ...)
{
  va_list args;
  indent_ptr = indent;
  va_start (args, message);
  complains (loc, flags, message, args);
  va_end (args);
}

void
complain_args (location const *loc, warnings w, unsigned *indent,
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
deprecated_directive (location const *loc, char const *old, char const *upd)
{
  if (feature_flag & feature_caret)
    complain (loc, Wdeprecated,
              _("deprecated directive, use %s"),
              quote_n (1, upd));
  else
    complain (loc, Wdeprecated,
              _("deprecated directive: %s, use %s"),
              quote (old), quote_n (1, upd));
}

void
duplicate_directive (char const *directive,
                     location first, location second)
{
  unsigned i = 0;
  complain (&second, complaint, _("only one %s allowed per rule"), directive);
  i += SUB_INDENT;
  complain_indent (&first, complaint, &i, _("previous declaration"));
}
