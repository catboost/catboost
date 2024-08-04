/* DO NOT EDIT! GENERATED AUTOMATICALLY! */
/* Dummy replacement for part of the public API of the libtextstyle library.
   Copyright (C) 2006-2007, 2019-2020 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* Written by Bruno Haible <bruno@clisp.org>, 2019.  */

/* This file is used as replacement when libtextstyle with its include file
   <textstyle.h> is not found.
   It supports the essential API and implements it in a way that does not
   provide text styling.  That is, it produces plain text output via <stdio.h>
   FILE objects.
   Thus, it allows a package to be build with or without a dependency to
   libtextstyle, with very few occurrences of '#if HAVE_LIBTEXTSTYLE'.

   Restriction:
   It assumes that freopen() is not being called on stdout and stderr.  */

#ifndef _TEXTSTYLE_H
#define _TEXTSTYLE_H

#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#if HAVE_TCDRAIN
# include <termios.h>
#endif

/* ----------------------------- From ostream.h ----------------------------- */

/* Describes the scope of a flush operation.  */
typedef enum
{
  /* Flushes buffers in this ostream_t.
     Use this value if you want to write to the underlying ostream_t.  */
  FLUSH_THIS_STREAM = 0,
  /* Flushes all buffers in the current process.
     Use this value if you want to write to the same target through a
     different file descriptor or a FILE stream.  */
  FLUSH_THIS_PROCESS = 1,
  /* Flushes buffers in the current process and attempts to flush the buffers
     in the kernel.
     Use this value so that some other process (or the kernel itself)
     may write to the same target.  */
  FLUSH_ALL = 2
} ostream_flush_scope_t;


/* An output stream is an object to which one can feed a sequence of bytes.  */

typedef FILE * ostream_t;

static inline void
ostream_write_mem (ostream_t stream, const void *data, size_t len)
{
  if (len > 0)
    fwrite (data, 1, len, stream);
}

static inline void
ostream_flush (ostream_t stream, ostream_flush_scope_t scope)
{
  fflush (stream);
  if (scope == FLUSH_ALL)
    {
      int fd = fileno (stream);
      if (fd >= 0)
        {
          /* For streams connected to a disk file:  */
          fsync (fd);
          #if HAVE_TCDRAIN
          /* For streams connected to a terminal:  */
          {
            int retval;

            do
              retval = tcdrain (fd);
            while (retval < 0 && errno == EINTR);
          }
          #endif
        }
    }
}

static inline void
ostream_free (ostream_t stream)
{
  if (stream == stdin || stream == stderr)
    fflush (stream);
  else
    fclose (stream);
}

static inline void
ostream_write_str (ostream_t stream, const char *string)
{
  ostream_write_mem (stream, string, strlen (string));
}

static inline ptrdiff_t ostream_printf (ostream_t stream,
                                        const char *format, ...)
#if (__GNUC__ == 3 && __GNUC_MINOR__ >= 1) || __GNUC__ > 3
  __attribute__ ((__format__ (__printf__, 2, 3)))
#endif
  ;
static inline ptrdiff_t
ostream_printf (ostream_t stream, const char *format, ...)
{
  va_list args;
  char *temp_string;
  ptrdiff_t ret;

  va_start (args, format);
  ret = vasprintf (&temp_string, format, args);
  va_end (args);
  if (ret >= 0)
    {
      if (ret > 0)
        ostream_write_str (stream, temp_string);
      free (temp_string);
    }
  return ret;
}

static inline ptrdiff_t ostream_vprintf (ostream_t stream,
                                         const char *format, va_list args)
#if (__GNUC__ == 3 && __GNUC_MINOR__ >= 1) || __GNUC__ > 3
  __attribute__ ((__format__ (__printf__, 2, 0)))
#endif
  ;
static inline ptrdiff_t
ostream_vprintf (ostream_t stream, const char *format, va_list args)
{
  char *temp_string;
  ptrdiff_t ret = vasprintf (&temp_string, format, args);
  if (ret >= 0)
    {
      if (ret > 0)
        ostream_write_str (stream, temp_string);
      free (temp_string);
    }
  return ret;
}

/* ------------------------- From styled-ostream.h ------------------------- */

typedef ostream_t styled_ostream_t;

#define styled_ostream_write_mem ostream_write_mem
#define styled_ostream_flush ostream_flush
#define styled_ostream_free ostream_free

static inline void
styled_ostream_begin_use_class (styled_ostream_t stream _GL_UNUSED,
                                const char *classname _GL_UNUSED)
{
}

static inline void
styled_ostream_end_use_class (styled_ostream_t stream _GL_UNUSED,
                              const char *classname _GL_UNUSED)
{
}

static inline const char *
styled_ostream_get_hyperlink_ref (styled_ostream_t stream _GL_UNUSED)
{
  return NULL;
}

static inline const char *
styled_ostream_get_hyperlink_id (styled_ostream_t stream _GL_UNUSED)
{
  return NULL;
}

static inline void
styled_ostream_set_hyperlink (styled_ostream_t stream _GL_UNUSED,
                              const char *ref _GL_UNUSED,
                              const char *id _GL_UNUSED)
{
}

static inline void
styled_ostream_flush_to_current_style (styled_ostream_t stream _GL_UNUSED)
{
}

/* -------------------------- From file-ostream.h -------------------------- */

typedef ostream_t file_ostream_t;

#define file_ostream_write_mem ostream_write_mem
#define file_ostream_flush ostream_flush
#define file_ostream_free ostream_free

static inline file_ostream_t
file_ostream_create (FILE *fp)
{
  return fp;
}

/* --------------------------- From fd-ostream.h --------------------------- */

typedef ostream_t fd_ostream_t;

#define fd_ostream_write_mem ostream_write_mem
#define fd_ostream_flush ostream_flush
#define fd_ostream_free ostream_free

static inline fd_ostream_t
fd_ostream_create (int fd, const char *filename _GL_UNUSED,
                   bool buffered _GL_UNUSED)
{
  if (fd == 1)
    return stdout;
  else if (fd == 2)
    return stderr;
  else
    return fdopen (fd, "w");
}

/* -------------------------- From term-ostream.h -------------------------- */

typedef int term_color_t;
enum
{
  COLOR_DEFAULT = -1  /* unknown */
};

typedef enum
{
  WEIGHT_NORMAL = 0,
  WEIGHT_BOLD,
  WEIGHT_DEFAULT = WEIGHT_NORMAL
} term_weight_t;

typedef enum
{
  POSTURE_NORMAL = 0,
  POSTURE_ITALIC, /* same as oblique */
  POSTURE_DEFAULT = POSTURE_NORMAL
} term_posture_t;

typedef enum
{
  UNDERLINE_OFF = 0,
  UNDERLINE_ON,
  UNDERLINE_DEFAULT = UNDERLINE_OFF
} term_underline_t;

typedef ostream_t term_ostream_t;

#define term_ostream_write_mem ostream_write_mem
#define term_ostream_flush ostream_flush
#define term_ostream_free ostream_free

static inline term_color_t
term_ostream_get_color (term_ostream_t stream _GL_UNUSED)
{
  return COLOR_DEFAULT;
}

static inline void
term_ostream_set_color (term_ostream_t stream _GL_UNUSED,
                        term_color_t color _GL_UNUSED)
{
}

static inline term_color_t
term_ostream_get_bgcolor (term_ostream_t stream _GL_UNUSED)
{
  return COLOR_DEFAULT;
}

static inline void
term_ostream_set_bgcolor (term_ostream_t stream _GL_UNUSED,
                          term_color_t color _GL_UNUSED)
{
}

static inline term_weight_t
term_ostream_get_weight (term_ostream_t stream _GL_UNUSED)
{
  return WEIGHT_DEFAULT;
}

static inline void
term_ostream_set_weight (term_ostream_t stream _GL_UNUSED,
                         term_weight_t weight _GL_UNUSED)
{
}

static inline term_posture_t
term_ostream_get_posture (term_ostream_t stream _GL_UNUSED)
{
  return POSTURE_DEFAULT;
}

static inline void
term_ostream_set_posture (term_ostream_t stream _GL_UNUSED,
                          term_posture_t posture _GL_UNUSED)
{
}

static inline term_underline_t
term_ostream_get_underline (term_ostream_t stream _GL_UNUSED)
{
  return UNDERLINE_DEFAULT;
}

static inline void
term_ostream_set_underline (term_ostream_t stream _GL_UNUSED,
                            term_underline_t underline _GL_UNUSED)
{
}

static inline const char *
term_ostream_get_hyperlink_ref (term_ostream_t stream _GL_UNUSED)
{
  return NULL;
}

static inline const char *
term_ostream_get_hyperlink_id (term_ostream_t stream _GL_UNUSED)
{
  return NULL;
}

static inline void
term_ostream_set_hyperlink (term_ostream_t stream _GL_UNUSED,
                            const char *ref _GL_UNUSED,
                            const char *id _GL_UNUSED)
{
}

static inline void
term_ostream_flush_to_current_style (term_ostream_t stream)
{
  fflush (stream);
}

typedef enum
{
  TTYCTL_AUTO = 0,  /* Automatic best-possible choice.  */
  TTYCTL_NONE,      /* No control.
                       Result: Garbled output can occur, and the terminal can
                       be left in any state when the program is interrupted.  */
  TTYCTL_PARTIAL,   /* Signal handling.
                       Result: Garbled output can occur, but the terminal will
                       be left in the default state when the program is
                       interrupted.  */
  TTYCTL_FULL       /* Signal handling and disabling echo and flush-upon-signal.
                       Result: No garbled output, and the the terminal will
                       be left in the default state when the program is
                       interrupted.  */
} ttyctl_t;

static inline term_ostream_t
term_ostream_create (int fd, const char *filename,
                     ttyctl_t tty_control _GL_UNUSED)
{
  return fd_ostream_create (fd, filename, true);
}

/* ----------------------- From term-styled-ostream.h ----------------------- */

typedef styled_ostream_t term_styled_ostream_t;

#define term_styled_ostream_write_mem ostream_write_mem
#define term_styled_ostream_flush ostream_flush
#define term_styled_ostream_free ostream_free
#define term_styled_ostream_begin_use_class styled_ostream_begin_use_class
#define term_styled_ostream_end_use_class styled_ostream_end_use_class
#define term_styled_ostream_get_hyperlink_ref styled_ostream_get_hyperlink_ref
#define term_styled_ostream_get_hyperlink_id styled_ostream_get_hyperlink_id
#define term_styled_ostream_set_hyperlink styled_ostream_set_hyperlink
#define term_styled_ostream_flush_to_current_style styled_ostream_flush_to_current_style

static inline term_styled_ostream_t
term_styled_ostream_create (int fd, const char *filename,
                            ttyctl_t tty_control _GL_UNUSED,
                            const char *css_filename _GL_UNUSED)
{
  return fd_ostream_create (fd, filename, true);
}

/* ----------------------- From html-styled-ostream.h ----------------------- */

typedef styled_ostream_t html_styled_ostream_t;

static inline html_styled_ostream_t
html_styled_ostream_create (ostream_t destination _GL_UNUSED,
                            const char *css_filename _GL_UNUSED)
{
  abort ();
  return NULL;
}

/* ------------------------------ From color.h ------------------------------ */

#define color_test_mode false

enum color_option { color_no, color_tty, color_yes, color_html };
#define color_mode color_no

#define style_file_name NULL

static inline bool
handle_color_option (const char *option _GL_UNUSED)
{
  return false;
}

static inline void
handle_style_option (const char *option _GL_UNUSED)
{
}

static inline void
print_color_test (void)
{
  abort ();
}

static inline void
style_file_prepare (const char *style_file_envvar _GL_UNUSED,
                    const char *stylesdir_envvar _GL_UNUSED,
                    const char *stylesdir_after_install _GL_UNUSED,
                    const char *default_style_file _GL_UNUSED)
{
}

/* ------------------------------ From misc.h ------------------------------ */

static inline styled_ostream_t
styled_ostream_create (int fd, const char *filename,
                       ttyctl_t tty_control _GL_UNUSED,
                       const char *css_filename _GL_UNUSED)
{
  return fd_ostream_create (fd, filename, true);
}

static inline void
libtextstyle_set_failure_exit_code (int exit_code _GL_UNUSED)
{
}

#endif /* _TEXTSTYLE_H */
