/* Support for fixing grammar files.

   Copyright (C) 2019 Free Software Foundation, Inc.

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

#include "fixits.h"

#include "system.h"

#include "error.h"
#include "get-errno.h"
#include "getargs.h"
#include "gl_array_list.h"
#include "gl_xlist.h"
#include "progname.h"
#include "quote.h"
#include "quotearg.h"
#include "vasnprintf.h"

#include "files.h"

typedef struct
{
  location location;
  char *fix;
} fixit;

gl_list_t fixits = NULL;

static fixit *
fixit_new (location const *loc, char const* fix)
{
  fixit *res = xmalloc (sizeof *res);
  res->location = *loc;
  res->fix = xstrdup (fix);
  return res;
}


static void
fixit_free (fixit *f)
{
  free (f->fix);
  free (f);
}


/* GCC and Clang follow the same pattern.
   https://gcc.gnu.org/onlinedocs/gcc/Diagnostic-Message-Formatting-Options.html
   http://clang.llvm.org/docs/UsersManual.html#cmdoption-fdiagnostics-parseable-fixits */
static void
fixit_print (fixit const *f, FILE *out)
{
  fprintf (out, "fix-it:%s:{%d:%d-%d:%d}:%s\n",
           quotearg_n_style (1, c_quoting_style, f->location.start.file),
           f->location.start.line, f->location.start.column,
           f->location.end.line, f->location.end.column,
           quotearg_n_style (2, c_quoting_style, f->fix));
}


void
fixits_register (location const *loc, char const* fix)
{
  if (!fixits)
    fixits = gl_list_create_empty (GL_ARRAY_LIST,
                                   /* equals */ NULL,
                                   /* hashcode */ NULL,
                                   (gl_listelement_dispose_fn) fixit_free,
                                   true);
  fixit *f = fixit_new (loc, fix);
  gl_list_add_last (fixits, f);
  if (feature_flag & feature_fixit_parsable)
    fixit_print (f, stderr);
}


bool
fixits_empty (void)
{
  return !fixits;
}


void
fixits_run (void)
{
  if (!fixits)
    return;

  /* This is not unlike what is done in location_caret.  */
  uniqstr input = ((fixit *) gl_list_get_at (fixits, 0))->location.start.file;
  /* Backup the file. */
  char buf[256];
  size_t len = sizeof (buf);
  char *backup = asnprintf (buf, &len, "%s~", input);
  if (!backup)
    xalloc_die ();
  if (rename (input, backup))
    error (EXIT_FAILURE, get_errno (),
           _("%s: cannot backup"), quotearg_colon (input));
  FILE *in = xfopen (backup, "r");
  FILE *out = xfopen (input, "w");
  size_t line = 1;
  size_t offset = 1;
  fixit const *f = NULL;
  gl_list_iterator_t iter = gl_list_iterator (fixits);
  while (gl_list_iterator_next (&iter, (const void**) &f, NULL))
    {
      /* Look for the correct line. */
      while (line < f->location.start.line)
        {
          int c = getc (in);
          if (c == EOF)
            break;
          if (c == '\n')
            {
              ++line;
              offset = 1;
            }
          putc (c, out);
        }
      /* Look for the right offset. */
      while (offset < f->location.start.column)
        {
          int c = getc (in);
          if (c == EOF)
            break;
          ++offset;
          putc (c, out);
        }

      /* Paste the fix instead. */
      fputs (f->fix, out);

      /* Skip the bad input. */
      while (line < f->location.end.line)
        {
          int c = getc (in);
          if (c == EOF)
            break;
          if (c == '\n')
            {
              ++line;
              offset = 1;
            }
        }
      while (offset < f->location.end.column)
        {
          int c = getc (in);
          if (c == EOF)
            break;
          ++offset;
        }
      /* If erasing the content of a full line, also remove the
         end-of-line. */
      if (f->fix[0] == 0 && f->location.start.column == 1)
        {
          int c = getc (in);
          if (c == EOF)
            break;
          else if (c == '\n')
            {
              ++line;
              offset = 1;
            }
          else
            ungetc (c, in);
        }
    }
  /* Paste the rest of the file.  */
  {
    int c;
    while ((c = getc (in)) != EOF)
      putc (c, out);
  }

  gl_list_iterator_free (&iter);
  xfclose (out);
  xfclose (in);
  fprintf (stderr, "%s: file %s was updated (backup: %s)\n",
           program_name, quote_n (0, input), quote_n (1, backup));
  if (backup != buf)
    free (backup);
}


/* Free the registered fixits.  */
void fixits_free (void)
{
  if (fixits)
    {
      gl_list_free (fixits);
      fixits = NULL;
    }
}
