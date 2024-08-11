/* Declaration for error-reporting function for Bison.

   Copyright (C) 2000-2002, 2006, 2009-2015, 2018-2021 Free Software
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
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

#ifndef COMPLAIN_H_
# define COMPLAIN_H_ 1

# include <attribute.h>

# include "location.h"

/*---------------.
| Error stream.  |
`---------------*/

/** Enable a style on \a out provided it's stderr.  */
void begin_use_class (const char *style, FILE *out);

/** Disable a style on \a out provided it's stderr.  */
void end_use_class (const char *style, FILE *out);

/** Flush \a out.  */
void flush (FILE *out);

/** Whether there's styling on OUT.  */
bool is_styled (FILE *out);

/*-------------.
| --warnings.  |
`-------------*/

/** The bits assigned to each warning type.  */
typedef enum
  {
    warning_conflicts_rr,
    warning_conflicts_sr,
    warning_counterexamples,
    warning_dangling_alias,
    warning_deprecated,
    warning_empty_rule,
    warning_midrule_values,
    warning_other,
    warning_precedence,
    warning_yacc,           /**< POSIXME.  */

    warnings_size           /**< The number of warnings.  Must be last.  */
  } warning_bit;

/** Whether -Werror was set. */
extern bool warnings_are_errors;

/** Document --warning arguments.  */
void warning_usage (FILE *out);

/** Decode a single argument from -W.
 *
 *  \param arg      the subarguments to decode.
 *                  If null, then activate all the flags.
 *  \param no       length of the potential "no-" prefix.
 *                  Can be 0 or 3. If 3, negate the action of the subargument.
 *  \param err      length of a potential "error=".
 *                  Can be 0 or 6. If 6, treat the subargument as a CATEGORY.
 *
 *  If VALUE != 0 then KEY sets flags and no-KEY clears them.
 *  If VALUE == 0 then KEY clears all flags from \c all and no-KEY sets all
 *  flags from \c all.  Thus no-none = all and no-all = none.
 */
void warning_argmatch (char const *arg, size_t no, size_t err);

/** Decode a comma-separated list of arguments from -W.
 *
 *  \param args     comma separated list of effective subarguments to decode.
 *                  If 0, then activate all the flags.
 */
void warnings_argmatch (char *args);


/*-----------.
| complain.  |
`-----------*/

/** Initialize this module.  */
void complain_init (void);

/** Reclaim resources.  */
void complain_free (void);

/** Initialize support for colored messages.  */
void complain_init_color (void);

/** Flags passed to diagnostics functions.  */
typedef enum
  {
    Wnone             = 0,       /**< Issue no warnings.  */

    Wconflicts_rr     = 1 << warning_conflicts_rr,
    Wconflicts_sr     = 1 << warning_conflicts_sr,
    Wcounterexamples  = 1 << warning_counterexamples,
    Wdangling_alias   = 1 << warning_dangling_alias,
    Wdeprecated       = 1 << warning_deprecated,
    Wempty_rule       = 1 << warning_empty_rule,
    Wmidrule_values   = 1 << warning_midrule_values,
    Wother            = 1 << warning_other,
    Wprecedence       = 1 << warning_precedence,
    Wyacc             = 1 << warning_yacc,

    complaint         = 1 << 11, /**< All complaints.  */
    fatal             = 1 << 12, /**< All fatal errors.  */
    silent            = 1 << 13, /**< Do not display the warning type.  */
    no_caret          = 1 << 14, /**< Do not display caret location.  */
    note              = 1 << 15, /**< Display as a note.  */

    /**< All above warnings.  */
    Weverything       = ~complaint & ~fatal & ~silent,
    Wall              = Weverything & ~Wcounterexamples & ~Wdangling_alias & ~Wyacc
  } warnings;

/** Whether the warnings of \a flags are all unset.
    (Never enabled, never disabled). */
bool warning_is_unset (warnings flags);

/** Whether warnings of \a flags should be reported. */
bool warning_is_enabled (warnings flags);

/** Make a complaint, with maybe a location.  */
void complain (location const *loc, warnings flags, char const *message, ...)
  ATTRIBUTE_FORMAT ((__printf__, 3, 4));

/** Likewise, but with an \a argc/argv interface.  */
void complain_args (location const *loc, warnings w,
                    int argc, char *arg[]);

/** Make a subcomplain with location and note.  */
void subcomplain (location const *loc, warnings flags,
                  char const *message, ...)
  ATTRIBUTE_FORMAT ((__printf__, 3, 4));


/** GNU Bison extension not valid with POSIX Yacc.  */
void bison_directive (location const *loc, char const *directive);

/** Report an obsolete syntax, suggest the updated one.  */
void deprecated_directive (location const *loc,
                           char const *obsolete, char const *updated);

/** Report a repeated directive.  */
void duplicate_directive (char const *directive,
                          location first, location second);

/** Report a repeated directive for a rule.  */
void duplicate_rule_directive (char const *directive,
                               location first, location second);

/** Report a syntax error, where argv[0] is the unexpected
    token, and argv[1...argc] are the expected ones.  */
void syntax_error (location loc,
                   int argc, const char* argv[]);

/** Warnings treated as errors shouldn't stop the execution as regular
    errors should (because due to their nature, it is safe to go
    on). Thus, there are three possible execution statuses.  */
typedef enum
  {
    status_none,             /**< No diagnostic issued so far.  */
    status_warning_as_error, /**< A warning was issued (but no error).  */
    status_complaint         /**< An error was issued.  */
  } err_status;

/** Whether an error was reported.  */
extern err_status complaint_status;

#endif /* !COMPLAIN_H_ */
