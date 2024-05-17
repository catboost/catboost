/* GNU m4 -- A simple macro processor

   Copyright (C) 1991-1994, 2004, 2006-2007, 2009-2013 Free Software
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

#include "m4.h"

#include <stdarg.h>
#include <sys/stat.h>

/* File for debugging output.  */
FILE *debug = NULL;

/* Obstack for trace messages.  */
static struct obstack trace;

static void debug_set_file (FILE *);

/*----------------------------------.
| Initialise the debugging module.  |
`----------------------------------*/

void
debug_init (void)
{
  debug_set_file (stderr);
  obstack_init (&trace);
}

/*-----------------------------------------------------------------.
| Function to decode the debugging flags OPTS.  Used by main while |
| processing option -d, and by the builtin debugmode ().           |
`-----------------------------------------------------------------*/

int
debug_decode (const char *opts)
{
  int level;

  if (opts == NULL || *opts == '\0')
    level = DEBUG_TRACE_DEFAULT;
  else
    {
      for (level = 0; *opts; opts++)
        {
          switch (*opts)
            {
            case 'a':
              level |= DEBUG_TRACE_ARGS;
              break;

            case 'e':
              level |= DEBUG_TRACE_EXPANSION;
              break;

            case 'q':
              level |= DEBUG_TRACE_QUOTE;
              break;

            case 't':
              level |= DEBUG_TRACE_ALL;
              break;

            case 'l':
              level |= DEBUG_TRACE_LINE;
              break;

            case 'f':
              level |= DEBUG_TRACE_FILE;
              break;

            case 'p':
              level |= DEBUG_TRACE_PATH;
              break;

            case 'c':
              level |= DEBUG_TRACE_CALL;
              break;

            case 'i':
              level |= DEBUG_TRACE_INPUT;
              break;

            case 'x':
              level |= DEBUG_TRACE_CALLID;
              break;

            case 'V':
              level |= DEBUG_TRACE_VERBOSE;
              break;

            default:
              return -1;
            }
        }
    }

  /* This is to avoid screwing up the trace output due to changes in the
     debug_level.  */

  obstack_free (&trace, obstack_finish (&trace));

  return level;
}

/*-----------------------------------------------------------------.
| Change the debug output stream to FP.  If the underlying file is |
| the same as stdout, use stdout instead so that debug messages    |
| appear in the correct relative position.                         |
`-----------------------------------------------------------------*/

static void
debug_set_file (FILE *fp)
{
  struct stat stdout_stat, debug_stat;

  if (debug != NULL && debug != stderr && debug != stdout
      && close_stream (debug) != 0)
    {
      M4ERROR ((warning_status, errno, "error writing to debug stream"));
      retcode = EXIT_FAILURE;
    }
  debug = fp;

  if (debug != NULL && debug != stdout)
    {
      if (fstat (STDOUT_FILENO, &stdout_stat) < 0)
        return;
      if (fstat (fileno (debug), &debug_stat) < 0)
        return;

      /* mingw has a bug where fstat on a regular file reports st_ino
         of 0.  On normal system, st_ino should never be 0.  */
      if (stdout_stat.st_ino == debug_stat.st_ino
          && stdout_stat.st_dev == debug_stat.st_dev
          && stdout_stat.st_ino != 0)
        {
          if (debug != stderr && close_stream (debug) != 0)
            {
              M4ERROR ((warning_status, errno,
                        "error writing to debug stream"));
              retcode = EXIT_FAILURE;
            }
          debug = stdout;
        }
    }
}

/*-----------------------------------------------------------.
| Serialize files.  Used before executing a system command.  |
`-----------------------------------------------------------*/

void
debug_flush_files (void)
{
  fflush (stdout);
  fflush (stderr);
  if (debug != NULL && debug != stdout && debug != stderr)
    fflush (debug);
  /* POSIX requires that if m4 doesn't consume all input, but stdin is
     opened on a seekable file, that the file pointer be left at the
     next character on exit (but places no restrictions on the file
     pointer location on a non-seekable file).  It also requires that
     fflush() followed by fseeko() on an input file set the underlying
     file pointer, and gnulib guarantees these semantics.  However,
     fflush() on a non-seekable file can lose buffered data, which we
     might otherwise want to process after syscmd.  Hence, we must
     check whether stdin is seekable.  We must also be tolerant of
     operating with stdin closed, so we don't report any failures in
     this attempt.  The stdio-safer module and friends are essential,
     so that if stdin was closed, this lseek is not on some other file
     that we have since opened.  */
  if (lseek (STDIN_FILENO, 0, SEEK_CUR) >= 0
      && fflush (stdin) == 0)
    {
      fseeko (stdin, 0, SEEK_CUR);
    }
}

/*--------------------------------------------------------------.
| Change the debug output to file NAME.  If NAME is NULL, debug |
| output is reverted to stderr, and if empty, debug output is   |
| discarded.  Return true iff the output stream was changed.    |
`--------------------------------------------------------------*/

bool
debug_set_output (const char *name)
{
  FILE *fp;

  if (name == NULL)
    debug_set_file (stderr);
  else if (*name == '\0')
    debug_set_file (NULL);
  else
    {
      fp = fopen (name, "a");
      if (fp == NULL)
        return false;

      if (set_cloexec_flag (fileno (fp), true) != 0)
        M4ERROR ((warning_status, errno,
                  "Warning: cannot protect debug file across forks"));
      debug_set_file (fp);
    }
  return true;
}

/*--------------------------------------------------------------.
| Print the header of a one-line debug message, starting by "m4 |
| debug".                                                       |
`--------------------------------------------------------------*/

void
debug_message_prefix (void)
{
  xfprintf (debug, "m4debug:");
  if (current_line)
  {
    if (debug_level & DEBUG_TRACE_FILE)
      xfprintf (debug, "%s:", current_file);
    if (debug_level & DEBUG_TRACE_LINE)
      xfprintf (debug, "%d:", current_line);
  }
  putc (' ', debug);
}

/* The rest of this file contains the functions for macro tracing output.
   All tracing output for a macro call is collected on an obstack TRACE,
   and printed whenever the line is complete.  This prevents tracing
   output from interfering with other debug messages generated by the
   various builtins.  */

/*------------------------------------------------------------------.
| Tracing output is formatted here, by a simplified                 |
| printf-to-obstack function trace_format ().  Understands only %S, |
| %s, %d, %l (optional left quote) and %r (optional right quote).   |
`------------------------------------------------------------------*/

static void
trace_format (const char *fmt, ...)
{
  va_list args;
  char ch;

  int d;
  const char *s;
  int slen;
  int maxlen;

  va_start (args, fmt);

  while (true)
    {
      while ((ch = *fmt++) != '\0' && ch != '%')
        obstack_1grow (&trace, ch);

      if (ch == '\0')
        break;

      maxlen = 0;
      switch (*fmt++)
        {
        case 'S':
          maxlen = max_debug_argument_length;
          /* fall through */

        case 's':
          s = va_arg (args, const char *);
          break;

        case 'l':
          s = (debug_level & DEBUG_TRACE_QUOTE) ? lquote.string : "";
          break;

        case 'r':
          s = (debug_level & DEBUG_TRACE_QUOTE) ? rquote.string : "";
          break;

        case 'd':
          d = va_arg (args, int);
          s = ntoa (d, 10);
          break;

        default:
          s = "";
          break;
        }

      slen = strlen (s);
      if (maxlen == 0 || maxlen > slen)
        obstack_grow (&trace, s, slen);
      else
        {
          obstack_grow (&trace, s, maxlen);
          obstack_grow (&trace, "...", 3);
        }
    }

  va_end (args);
}

/*------------------------------------------------------------------.
| Format the standard header attached to all tracing output lines.  |
`------------------------------------------------------------------*/

static void
trace_header (int id)
{
  trace_format ("m4trace:");
  if (current_line)
    {
      if (debug_level & DEBUG_TRACE_FILE)
        trace_format ("%s:", current_file);
      if (debug_level & DEBUG_TRACE_LINE)
        trace_format ("%d:", current_line);
    }
  trace_format (" -%d- ", expansion_level);
  if (debug_level & DEBUG_TRACE_CALLID)
    trace_format ("id %d: ", id);
}

/*----------------------------------------------------.
| Print current tracing line, and clear the obstack.  |
`----------------------------------------------------*/

static void
trace_flush (void)
{
  char *line;

  obstack_1grow (&trace, '\0');
  line = (char *) obstack_finish (&trace);
  DEBUG_PRINT1 ("%s\n", line);
  obstack_free (&trace, line);
}

/*-------------------------------------------------------------.
| Do pre-argument-collction tracing for macro NAME.  Used from |
| expand_macro ().                                             |
`-------------------------------------------------------------*/

void
trace_prepre (const char *name, int id)
{
  trace_header (id);
  trace_format ("%s ...", name);
  trace_flush ();
}

/*--------------------------------------------------------------.
| Format the parts of a trace line, that can be made before the |
| macro is actually expanded.  Used from expand_macro ().       |
`--------------------------------------------------------------*/

void
trace_pre (const char *name, int id, int argc, token_data **argv)
{
  int i;
  const builtin *bp;

  trace_header (id);
  trace_format ("%s", name);

  if (argc > 1 && (debug_level & DEBUG_TRACE_ARGS))
    {
      trace_format ("(");

      for (i = 1; i < argc; i++)
        {
          if (i != 1)
            trace_format (", ");

          switch (TOKEN_DATA_TYPE (argv[i]))
            {
            case TOKEN_TEXT:
              trace_format ("%l%S%r", TOKEN_DATA_TEXT (argv[i]));
              break;

            case TOKEN_FUNC:
              bp = find_builtin_by_addr (TOKEN_DATA_FUNC (argv[i]));
              if (bp == NULL)
                {
                  M4ERROR ((warning_status, 0, "\
INTERNAL ERROR: builtin not found in builtin table! (trace_pre ())"));
                  abort ();
                }
              trace_format ("<%s>", bp->name);
              break;

            case TOKEN_VOID:
            default:
              M4ERROR ((warning_status, 0,
                        "INTERNAL ERROR: bad token data type (trace_pre ())"));
              abort ();
            }

        }
      trace_format (")");
    }

  if (debug_level & DEBUG_TRACE_CALL)
    {
      trace_format (" -> ???");
      trace_flush ();
    }
}

/*-------------------------------------------------------------------.
| Format the final part of a trace line and print it all.  Used from |
| expand_macro ().                                                   |
`-------------------------------------------------------------------*/

void
trace_post (const char *name, int id, int argc, const char *expanded)
{
  if (debug_level & DEBUG_TRACE_CALL)
    {
      trace_header (id);
      trace_format ("%s%s", name, (argc > 1) ? "(...)" : "");
    }

  if (expanded && (debug_level & DEBUG_TRACE_EXPANSION))
    trace_format (" -> %l%S%r", expanded);
  trace_flush ();
}
